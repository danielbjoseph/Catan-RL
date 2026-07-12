"""Shared helpers for scripted bots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from ..env.actions import Action, ActionType, Resource
from ..env.board import HexType
from ..env.player_state import BUILD_COSTS

if TYPE_CHECKING:
    from ..env.board import BoardConfig
    from ..env.game_state import GameState

# Dice "pips": relative frequency of each token out of 36 rolls (7 excluded).
PIPS: Dict[int, int] = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


def hex_pips(config: "BoardConfig", hex_id: int) -> int:
    if config.hex_resources[hex_id] == HexType.DESERT:
        return 0
    return PIPS.get(config.hex_tokens[hex_id], 0)


def vertex_production_score(config: "BoardConfig", vertex_id: int) -> float:
    """Total expected production (in pips) of a vertex."""
    return float(sum(hex_pips(config, h) for h in config.geometry.vertex_to_hexes[vertex_id]))


def vertex_diversity(config: "BoardConfig", vertex_id: int) -> int:
    """Number of distinct resource types adjacent to a vertex."""
    types = {
        config.hex_resources[h]
        for h in config.geometry.vertex_to_hexes[vertex_id]
        if config.hex_resources[h] != HexType.DESERT
    }
    return len(types)


def group_by_type(actions: List[Action]) -> Dict[ActionType, List[Action]]:
    by_type: Dict[ActionType, List[Action]] = {}
    for a in actions:
        by_type.setdefault(a.action_type, []).append(a)
    return by_type


def player_production_on_hex(state: "GameState", hex_id: int, player_id: int) -> int:
    """Pips of production player_id gets from this hex (cities count double)."""
    pips = hex_pips(state.config, hex_id)
    if pips == 0:
        return 0
    p = state.players[player_id]
    total = 0
    for v in state.config.geometry.hex_to_vertices[hex_id]:
        if v in p.city_vertices:
            total += 2 * pips
        elif v in p.settlement_vertices:
            total += pips
    return total


def robber_hex_score(state: "GameState", hex_id: int) -> float:
    """Higher = better robber placement for the current player."""
    me = state.current_player
    opp = sum(
        player_production_on_hex(state, hex_id, pid)
        for pid in range(state.n_players)
        if pid != me
    )
    own = player_production_on_hex(state, hex_id, me)
    return opp - 2.0 * own


def most_held_resource_action(state: "GameState", discard_actions: List[Action]) -> Action:
    p = state.current
    return max(discard_actions, key=lambda a: p.resources[int(a.resource)])


def find_enabling_trade(state: "GameState", trade_actions: List[Action]) -> Action | None:
    """
    Return a maritime trade that makes a city or settlement affordable
    (with a legal placement available), or None.
    """
    from ..env.validators import _settlement_actions

    p = state.current
    vertices = list(p.settlement_vertices | p.city_vertices)

    can_place_city = p.cities_available > 0 and len(p.settlement_vertices) > 0
    can_place_settlement = p.settlements_available > 0 and len(_settlement_actions(state)) > 0

    def affords_after(res: List[int], cost: Dict[Resource, int]) -> bool:
        return all(res[int(r)] >= n for r, n in cost.items())

    for target, cost in (("city", BUILD_COSTS["city"]), ("settlement", BUILD_COSTS["settlement"])):
        if target == "city" and not can_place_city:
            continue
        if target == "settlement" and not can_place_settlement:
            continue
        if p.has_resources(cost):
            continue  # already affordable; no trade needed
        for a in trade_actions:
            rate = state.config.best_trade_rate(vertices, a.resource)
            res = list(p.resources)
            res[int(a.resource)] -= rate
            res[int(a.resource2)] += 1
            if res[int(a.resource)] >= 0 and affords_after(res, cost):
                return a
    return None
