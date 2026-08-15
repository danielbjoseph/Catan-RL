"""
Trade personality presets.

Each preset wraps `heuristic_bot`: every decision that is *not* a trade
decision (placement, robber, discards, dev cards, builds, maritime trade,
end turn...) is delegated verbatim to `heuristic_bot.pick_action`, so
outcome differences across presets measure trade behavior alone.

A personality only makes two kinds of decisions:
  - Phase.TRADE_RESPONSE: ACCEPT_TRADE or DECLINE_TRADE a pending P2P offer.
  - Phase.MAIN: optionally PROPOSE_TRADE instead of falling through to the
    heuristic bot's normal action.

Value model
-----------
`value(r) = 1 / (1 + pips[r])` where `pips` is the player's own production
pip total per resource (settlements once, cities twice) -- a resource you
produce a lot of is cheap to you, one you don't produce is worth close to
1.0. A build-need bonus of `+0.5` is added on the *gain* side of a trade
margin if receiving one more unit of that resource would newly make a
build affordable (checked settlement, then city, then road -- first match
wins, no stacking).

`trade_margin(state, pid, gain, lose)` = sum(gain[r] * value(r)) -
sum(lose[r] * value(r)), where the bonus only applies to the gain-side
values. This makes an even-pips 1-for-1 trade break exactly at margin 0.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, TYPE_CHECKING

import numpy as np

from ..env.actions import ACCEPT_TRADE, Action, ActionType, DECLINE_TRADE, Resource
from ..env.board import HexType
from ..env.game_state import Phase
from ..env.player_state import BUILD_COSTS
from ..env.scoring import compute_vp
from ..env.validators import legal_actions
from ._common import group_by_type, hex_pips
from . import heuristic_bot

if TYPE_CHECKING:
    from ..env.game_state import GameState


@dataclass(frozen=True)
class TradePersonality:
    name: str
    propose_rate: float               # P(attempt a proposal) per MAIN visit
    accept_margin: float              # required value gain to accept (inf = never)
    leader_block_vp: Optional[int]    # decline proposer within this many VP of winning (None = off)
    desperation_scale: float          # accept_margin loosens by this per VP behind the leader
    max_proposals_per_turn: int
    accept_band: Optional[float] = None  # fair_dealer: accept iff |margin| <= band (overrides accept_margin)


PERSONALITIES: Dict[str, TradePersonality] = {
    "never_trader": TradePersonality(
        name="never_trader",
        propose_rate=0.0,
        accept_margin=float("inf"),
        leader_block_vp=None,
        desperation_scale=0.0,
        max_proposals_per_turn=0,
    ),
    "opportunist": TradePersonality(
        name="opportunist",
        propose_rate=0.9,
        accept_margin=-0.5,
        leader_block_vp=None,
        desperation_scale=0.0,
        max_proposals_per_turn=3,
    ),
    "stall_the_leader": TradePersonality(
        name="stall_the_leader",
        # v1 broadcast trading has no partner-choice sub-phase, so "prefers
        # partners furthest behind" (spec section 2.2) is a documented no-op
        # here: the only lever is refusing to *respond* to a leader-adjacent
        # proposer. Nothing changes what/whom it proposes to.
        propose_rate=0.4,
        accept_margin=0.0,
        leader_block_vp=2,
        desperation_scale=0.0,
        max_proposals_per_turn=3,
    ),
    "fair_dealer": TradePersonality(
        name="fair_dealer",
        propose_rate=0.4,
        accept_margin=0.0,
        leader_block_vp=None,
        desperation_scale=0.0,
        max_proposals_per_turn=2,
        accept_band=0.15,
    ),
    "desperado": TradePersonality(
        name="desperado",
        propose_rate=0.6,
        accept_margin=0.3,
        leader_block_vp=None,
        desperation_scale=0.25,
        max_proposals_per_turn=3,
    ),
}


# ---------------------------------------------------------------------------
# Value model
# ---------------------------------------------------------------------------

def resource_pips(state: "GameState", pid: int) -> np.ndarray:
    """Own production pips per resource (settlements once, cities twice)."""
    config = state.config
    geo = config.geometry
    player = state.players[pid]
    pips = np.zeros(5, dtype=float)
    for v in player.settlement_vertices:
        for h in geo.vertex_to_hexes[v]:
            if config.hex_resources[h] == HexType.DESERT:
                continue
            r = int(config.hex_resources[h].to_resource())
            pips[r] += hex_pips(config, h)
    for v in player.city_vertices:
        for h in geo.vertex_to_hexes[v]:
            if config.hex_resources[h] == HexType.DESERT:
                continue
            r = int(config.hex_resources[h].to_resource())
            pips[r] += 2 * hex_pips(config, h)
    return pips


def _has_reachable_settlement_spot(state: "GameState", pid: int) -> bool:
    """Mirrors validators._settlement_actions's reachability check for any pid."""
    player = state.players[pid]
    geo = state.config.geometry
    occupied = state.all_occupied_vertices()
    reachable = set()
    for eid in player.road_vertices:
        va, vb = geo.edge_to_vertices[eid]
        reachable.add(va)
        reachable.add(vb)
    for v in reachable:
        if v in occupied:
            continue
        if not any(nb in occupied for nb in geo.vertex_to_vertices[v]):
            return True
    return False


def _affords(resources, cost: Dict[Resource, int]) -> bool:
    return all(resources[int(r)] >= n for r, n in cost.items())


def _build_need_bonus(state: "GameState", pid: int, r: int) -> float:
    """+0.5 if gaining one unit of resource r newly affords a build.

    Checked in priority order settlement > city > road; first match wins,
    no stacking.
    """
    player = state.players[pid]
    before = list(player.resources)
    after = list(before)
    after[r] += 1

    can_place_settlement = (
        player.settlements_available > 0 and _has_reachable_settlement_spot(state, pid)
    )
    if can_place_settlement:
        cost = BUILD_COSTS["settlement"]
        if not _affords(before, cost) and _affords(after, cost):
            return 0.5

    can_place_city = player.cities_available > 0 and len(player.settlement_vertices) > 0
    if can_place_city:
        cost = BUILD_COSTS["city"]
        if not _affords(before, cost) and _affords(after, cost):
            return 0.5

    # Deliberate approximation: no legal-edge check (unlike the settlement
    # reachability check above) -- a placeable edge almost always exists, and
    # these are scripted opponents, not the trained policy.
    can_place_road = player.roads_available > 0
    if can_place_road:
        cost = BUILD_COSTS["road"]
        if not _affords(before, cost) and _affords(after, cost):
            return 0.5

    return 0.0


def trade_margin(state: "GameState", pid: int, gain: Dict[int, int], lose: Dict[int, int]) -> float:
    pips = resource_pips(state, pid)

    def base_value(r: int) -> float:
        return 1.0 / (1.0 + pips[r])

    margin = 0.0
    for r, n in gain.items():
        r = int(r)
        # The build-need bonus is a one-time +0.5 per resource ("gaining one
        # r newly affords a build"), not scaled by the quantity gained.
        margin += n * base_value(r) + _build_need_bonus(state, pid, r)
    for r, n in lose.items():
        margin -= n * base_value(int(r))
    return margin


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _leader_vp(state: "GameState") -> int:
    return max(compute_vp(pid, state) for pid in range(state.n_players))


def _is_sole_leader(state: "GameState", pid: int) -> bool:
    vps = [compute_vp(p, state) for p in range(state.n_players)]
    top = max(vps)
    return vps[pid] == top and vps.count(top) == 1


def _decide_response(state: "GameState", personality: TradePersonality) -> Action:
    pending = state.pending_trade
    proposer = pending["proposer"]
    give, get, give_n = pending["give"], pending["get"], pending["give_n"]
    me = state.current_player

    if personality.leader_block_vp is not None:
        proposer_vp = compute_vp(proposer, state)
        near_win = proposer_vp >= state.profile.win_vp - personality.leader_block_vp
        if near_win or _is_sole_leader(state, proposer):
            return DECLINE_TRADE

    gain = {give: give_n}
    lose = {get: 1}
    margin = trade_margin(state, me, gain, lose)

    if personality.accept_band is not None:
        accept = abs(margin) <= personality.accept_band
    else:
        deficit = max(0, _leader_vp(state) - compute_vp(me, state))
        threshold = personality.accept_margin - personality.desperation_scale * deficit
        accept = margin >= threshold

    return ACCEPT_TRADE if accept else DECLINE_TRADE


def _maybe_propose(
    state: "GameState", personality: TradePersonality, rng: random.Random
) -> Optional[Action]:
    if personality.propose_rate <= 0.0:
        return None
    if state.trades_proposed_this_turn >= personality.max_proposals_per_turn:
        return None

    actions = legal_actions(state)
    by_type = group_by_type(actions)
    propose_actions = by_type.get(ActionType.PROPOSE_TRADE)
    if not propose_actions:
        return None

    if rng.random() >= personality.propose_rate:
        return None

    me = state.current_player
    best_action = None
    best_margin = float("-inf")
    for a in propose_actions:
        gain = {int(a.resource2): 1}
        lose = {int(a.resource): a.give_n}
        margin = trade_margin(state, me, gain, lose)
        if margin > best_margin:
            best_margin = margin
            best_action = a

    if best_action is not None and best_margin > 0:
        return best_action
    return None


def make_personality_bot(personality: TradePersonality) -> Callable:
    """Return a pick_action(state, rng) -> Action closure for this preset."""

    def pick_action(state: "GameState", rng: random.Random | None = None) -> "Action":
        if rng is None:
            rng = random.Random()

        if state.phase == Phase.TRADE_RESPONSE:
            return _decide_response(state, personality)

        if state.phase == Phase.MAIN:
            proposal = _maybe_propose(state, personality, rng)
            if proposal is not None:
                return proposal

        return heuristic_bot.pick_action(state, rng)

    return pick_action
