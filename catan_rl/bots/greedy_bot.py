"""
Greedy build bot.

Fixed priority in the main phase:
  play VP card > build city > build settlement > expansion road
  > buy dev card > play knight (if robber hurts us) > enabling maritime trade
  > end turn.

Placements are chosen by raw production pips. Robber goes on the hex that
hurts opponents most; stealing targets the richest player; discards drop the
most-held resource.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from ..env.actions import Action, ActionType
from ..env.validators import legal_actions, _settlement_actions
from ._common import (
    find_enabling_trade,
    group_by_type,
    most_held_resource_action,
    robber_hex_score,
    vertex_production_score,
)

if TYPE_CHECKING:
    from ..env.game_state import GameState


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

    # --- forced/sub-phase decisions -------------------------------------
    if ActionType.ROLL_DICE in by_type:
        return by_type[ActionType.ROLL_DICE][0]
    if ActionType.DISCARD_RESOURCE in by_type:
        return most_held_resource_action(state, by_type[ActionType.DISCARD_RESOURCE])
    if ActionType.DECLINE_TRADE in by_type:
        # Plain bots don't reason about trades; conservatively decline
        # rather than falling through to rng.choice on trade responses.
        return by_type[ActionType.DECLINE_TRADE][0]
    if ActionType.MOVE_ROBBER in by_type:
        return max(by_type[ActionType.MOVE_ROBBER], key=lambda a: robber_hex_score(state, a.hex_id))
    if ActionType.CHOOSE_STEAL_TARGET in by_type:
        return max(
            by_type[ActionType.CHOOSE_STEAL_TARGET],
            key=lambda a: state.players[a.player_id].total_resources,
        )

    settlements = by_type.get(ActionType.BUILD_SETTLEMENT)
    roads = by_type.get(ActionType.BUILD_ROAD)

    # Setup / road-building sub-phases offer only placements
    from ..env.game_state import Phase

    if state.phase in (Phase.SETUP_SETTLEMENT_1, Phase.SETUP_SETTLEMENT_2):
        return max(settlements, key=lambda a: vertex_production_score(config, a.vertex_id))
    if state.phase in (Phase.SETUP_ROAD_1, Phase.SETUP_ROAD_2,
                       Phase.ROAD_BUILDING_1, Phase.ROAD_BUILDING_2):
        if roads:
            return rng.choice(roads)
        return by_type[ActionType.END_TURN][0]

    # --- main phase priorities -------------------------------------------
    if ActionType.PLAY_VICTORY_POINT in by_type:
        return by_type[ActionType.PLAY_VICTORY_POINT][0]

    if ActionType.BUILD_CITY in by_type:
        return max(by_type[ActionType.BUILD_CITY],
                   key=lambda a: vertex_production_score(config, a.vertex_id))

    if settlements:
        return max(settlements, key=lambda a: vertex_production_score(config, a.vertex_id))

    # Expansion road: only when no settlement spot is reachable yet
    if roads and me.settlements_available > 0 and not _settlement_actions(state):
        return rng.choice(roads)

    if ActionType.BUY_DEV_CARD in by_type:
        return by_type[ActionType.BUY_DEV_CARD][0]

    if ActionType.PLAY_KNIGHT in by_type:
        # Only bother if the robber is currently blocking one of our hexes
        geo = config.geometry
        own_vertices = me.settlement_vertices | me.city_vertices
        robber_hurts = any(
            v in own_vertices for v in geo.hex_to_vertices[state.robber_hex]
        )
        if robber_hurts:
            return by_type[ActionType.PLAY_KNIGHT][0]

    if ActionType.MARITIME_TRADE in by_type:
        trade = find_enabling_trade(state, by_type[ActionType.MARITIME_TRADE])
        if trade is not None:
            return trade

    if ActionType.END_TURN in by_type:
        return by_type[ActionType.END_TURN][0]
    return rng.choice(actions)
