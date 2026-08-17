"""
Heuristic expansion bot.

Like the greedy bot but with smarter placement valuation:
  - settlement spots scored by production pips + resource-diversity bonus
  - roads scored by the best future settlement spot they open up (2-ply)
  - robber placement weighted by opponents' victory points
  - steals from the current VP leader
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, List

from ..env.actions import Action, ActionType
from ..env.scoring import compute_vp
from ..env.validators import legal_actions, _settlement_actions
from ._common import (
    find_enabling_trade,
    group_by_type,
    most_held_resource_action,
    player_production_on_hex,
    vertex_diversity,
    vertex_production_score,
)

if TYPE_CHECKING:
    from ..env.game_state import GameState

_DIVERSITY_BONUS = 1.5


def _settlement_score(config, vertex_id: int) -> float:
    return (
        vertex_production_score(config, vertex_id)
        + _DIVERSITY_BONUS * vertex_diversity(config, vertex_id)
    )


def _is_open_spot(state: "GameState", v: int, occupied) -> bool:
    """Vertex is empty and satisfies the distance rule."""
    if v in occupied:
        return False
    geo = state.config.geometry
    return not any(nb in occupied for nb in geo.vertex_to_vertices[v])


def _road_score(state: "GameState", edge_id: int) -> float:
    """Score a candidate road by the settlement spots it opens (2-ply lookahead)."""
    geo = state.config.geometry
    occupied = state.all_occupied_vertices()
    all_roads = state.all_road_edges()
    best = 0.0
    for v in geo.edge_to_vertices[edge_id]:
        if v in occupied:
            continue
        if _is_open_spot(state, v, occupied):
            best = max(best, _settlement_score(state.config, v))
        # One road further
        for e2 in geo.vertex_to_edges[v]:
            if e2 == edge_id or e2 in all_roads:
                continue
            for w in geo.edge_to_vertices[e2]:
                if w != v and _is_open_spot(state, w, occupied):
                    best = max(best, 0.6 * _settlement_score(state.config, w))
    return best


def _robber_score(state: "GameState", hex_id: int) -> float:
    """Weight opponent production on the hex by how far ahead each opponent is."""
    me = state.current_player
    own = player_production_on_hex(state, hex_id, me)
    score = -2.0 * own
    for pid in range(state.n_players):
        if pid == me:
            continue
        vp_weight = 1.0 + compute_vp(pid, state) / 10.0
        score += vp_weight * player_production_on_hex(state, hex_id, pid)
    return score


def pick_action(state: "GameState", rng: random.Random | None = None) -> "Action":
    if rng is None:
        rng = random.Random()
    actions = legal_actions(state)
    if not actions:
        raise RuntimeError(
            f"No legal actions in phase {state.phase} for player {state.current_player}"
        )
    if len(actions) == 1:
        return actions[0]

    by_type = group_by_type(actions)
    config = state.config
    me = state.current

    if ActionType.ROLL_DICE in by_type:
        return by_type[ActionType.ROLL_DICE][0]
    if ActionType.DISCARD_RESOURCE in by_type:
        return most_held_resource_action(state, by_type[ActionType.DISCARD_RESOURCE])
    if ActionType.DECLINE_TRADE in by_type:
        # Plain bots don't reason about trades; conservatively decline
        # rather than falling through to rng.choice on trade responses.
        return by_type[ActionType.DECLINE_TRADE][0]
    if ActionType.MOVE_ROBBER in by_type:
        return max(by_type[ActionType.MOVE_ROBBER], key=lambda a: _robber_score(state, a.hex_id))
    if ActionType.CHOOSE_STEAL_TARGET in by_type:
        return max(
            by_type[ActionType.CHOOSE_STEAL_TARGET],
            key=lambda a: (compute_vp(a.player_id, state),
                           state.players[a.player_id].total_resources),
        )

    settlements = by_type.get(ActionType.BUILD_SETTLEMENT)
    roads = by_type.get(ActionType.BUILD_ROAD)

    from ..env.game_state import Phase

    if state.phase in (Phase.SETUP_SETTLEMENT_1, Phase.SETUP_SETTLEMENT_2):
        return max(settlements, key=lambda a: _settlement_score(config, a.vertex_id))
    if state.phase in (Phase.SETUP_ROAD_1, Phase.SETUP_ROAD_2,
                       Phase.ROAD_BUILDING_1, Phase.ROAD_BUILDING_2):
        if roads:
            return max(roads, key=lambda a: _road_score(state, a.edge_id))
        return by_type[ActionType.END_TURN][0]

    # --- main phase -------------------------------------------------------
    if ActionType.PLAY_VICTORY_POINT in by_type:
        return by_type[ActionType.PLAY_VICTORY_POINT][0]

    if ActionType.BUILD_CITY in by_type:
        return max(by_type[ActionType.BUILD_CITY],
                   key=lambda a: vertex_production_score(config, a.vertex_id))

    if settlements:
        return max(settlements, key=lambda a: _settlement_score(config, a.vertex_id))

    if roads and me.settlements_available > 0 and not _settlement_actions(state):
        return max(roads, key=lambda a: _road_score(state, a.edge_id))

    if ActionType.BUY_DEV_CARD in by_type:
        return by_type[ActionType.BUY_DEV_CARD][0]

    if ActionType.PLAY_KNIGHT in by_type:
        geo = config.geometry
        own_vertices = me.settlement_vertices | me.city_vertices
        if any(v in own_vertices for v in geo.hex_to_vertices[state.robber_hex]):
            return by_type[ActionType.PLAY_KNIGHT][0]

    if ActionType.MARITIME_TRADE in by_type:
        trade = find_enabling_trade(state, by_type[ActionType.MARITIME_TRADE])
        if trade is not None:
            return trade

    if ActionType.END_TURN in by_type:
        return by_type[ActionType.END_TURN][0]
    return rng.choice(actions)
