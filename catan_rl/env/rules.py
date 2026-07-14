"""
State transition engine.

apply_action(state, action) -> None   (mutates state in-place)

All legality is assumed checked by the caller (use validators.legal_actions).
This module only enforces game logic, not permission checks.
"""

from __future__ import annotations

import random
from typing import Optional, TYPE_CHECKING

from .actions import Action, ActionType, Resource, DevCard
from .board import HexType
from .game_state import Phase, WIN_VP
from .player_state import BUILD_COSTS
from .scoring import update_longest_road, update_largest_army, compute_vp, check_winner

if TYPE_CHECKING:
    from .game_state import GameState


def apply_action(state: "GameState", action: Action, rng: Optional[random.Random] = None):
    """Apply action to state in-place. rng is used for dice rolls."""
    if rng is None:
        rng = random.Random()

    t = action.action_type

    if t == ActionType.ROLL_DICE:
        _roll_dice(state, rng)
    elif t == ActionType.END_TURN and state.phase in (Phase.ROAD_BUILDING_1, Phase.ROAD_BUILDING_2):
        state.phase = _main_return_phase(state)
    elif t == ActionType.END_TURN:
        _end_turn(state)
    elif t == ActionType.BUILD_ROAD:
        _build_road(state, action.edge_id)
    elif t == ActionType.BUILD_SETTLEMENT:
        _build_settlement(state, action.vertex_id)
    elif t == ActionType.BUILD_CITY:
        _build_city(state, action.vertex_id)
    elif t == ActionType.MOVE_ROBBER:
        _move_robber(state, action.hex_id)
    elif t == ActionType.CHOOSE_STEAL_TARGET:
        _steal(state, action.player_id, rng)
    elif t == ActionType.MARITIME_TRADE:
        _maritime_trade(state, action.resource, action.resource2)
    elif t == ActionType.DISCARD_RESOURCE:
        _discard(state, action.resource)
    elif t == ActionType.BUY_DEV_CARD:
        _buy_dev_card(state, rng)
    elif t == ActionType.PLAY_KNIGHT:
        _play_knight(state)
    elif t == ActionType.PLAY_ROAD_BUILDING:
        _play_road_building(state)
    elif t == ActionType.PLAY_YEAR_OF_PLENTY:
        _play_year_of_plenty(state, action.resource, action.resource2)
    elif t == ActionType.PLAY_MONOPOLY:
        _play_monopoly(state, action.resource)
    elif t == ActionType.PLAY_VICTORY_POINT:
        _play_victory_point(state)
    else:
        raise ValueError(f"Unknown action type: {t}")

    _check_win(state)


# ---------------------------------------------------------------------------
# Setup phase transitions
# ---------------------------------------------------------------------------

def _build_settlement(state: "GameState", vertex_id: int):
    player = state.current
    player.settlement_vertices.add(vertex_id)
    player.settlements_built += 1

    geo = state.config.geometry

    if state.phase == Phase.SETUP_SETTLEMENT_1:
        state.phase = Phase.SETUP_ROAD_1

    elif state.phase == Phase.SETUP_SETTLEMENT_2:
        # Give initial resources for second settlement
        for hex_id in geo.vertex_to_hexes[vertex_id]:
            hex_type = state.config.hex_resources[hex_id]
            if hex_type != HexType.DESERT:
                res = hex_type.to_resource()
                available = state.bank[int(res)]
                if available > 0:
                    player.gain(res)
                    state.bank[int(res)] -= 1
        state.phase = Phase.SETUP_ROAD_2

    else:
        # Normal build: spend resources
        player.spend(BUILD_COSTS["settlement"])
        for r, n in BUILD_COSTS["settlement"].items():
            state.bank[int(r)] += n
        update_longest_road(state)
        _check_win(state)


def _build_road(state: "GameState", edge_id: int):
    player = state.current

    if state.phase in (Phase.SETUP_ROAD_1, Phase.SETUP_ROAD_2):
        player.road_vertices.add(edge_id)
        player.roads_built += 1
        _advance_setup(state)
    elif state.phase in (Phase.ROAD_BUILDING_1, Phase.ROAD_BUILDING_2):
        player.road_vertices.add(edge_id)
        player.roads_built += 1
        update_longest_road(state)
        if state.phase == Phase.ROAD_BUILDING_1:
            from .validators import _connected_road_actions
            if player.roads_available >= 1 and _connected_road_actions(state):
                state.phase = Phase.ROAD_BUILDING_2
            else:
                state.phase = _main_return_phase(state)
        else:
            state.phase = _main_return_phase(state)
    else:
        player.spend(BUILD_COSTS["road"])
        for r, n in BUILD_COSTS["road"].items():
            state.bank[int(r)] += n
        player.road_vertices.add(edge_id)
        player.roads_built += 1
        update_longest_road(state)


def _build_city(state: "GameState", vertex_id: int):
    player = state.current
    player.settlement_vertices.discard(vertex_id)
    player.city_vertices.add(vertex_id)
    player.cities_built += 1
    player.settlements_built -= 1  # settlement piece returns to supply

    player.spend(BUILD_COSTS["city"])
    for r, n in BUILD_COSTS["city"].items():
        state.bank[int(r)] += n

    # Return settlement piece to available (cities use city pieces not settlement)
    # settlements_built tracks pieces on board; city replaces it
    # Actually settlements_available = MAX_SETTLEMENTS - settlements_built
    # Since we removed one settlement and placed a city, net settlements_built decreases by 1
    # cities_built increases by 1; both correct above.


def _advance_setup(state: "GameState"):
    """Move to next player in setup sequence."""
    n = state.n_players

    if state.phase == Phase.SETUP_ROAD_1:
        state._setup_forward_idx += 1
        if state._setup_forward_idx < n:
            state.current_player = state._setup_forward_idx
            state.phase = Phase.SETUP_SETTLEMENT_1
        else:
            # Start reverse order
            state._setup_backward_idx = n - 1
            state.current_player = state._setup_backward_idx
            state.phase = Phase.SETUP_SETTLEMENT_2

    elif state.phase == Phase.SETUP_ROAD_2:
        state._setup_backward_idx -= 1
        if state._setup_backward_idx >= 0:
            state.current_player = state._setup_backward_idx
            state.phase = Phase.SETUP_SETTLEMENT_2
        else:
            # Setup complete; player 0 starts
            state.current_player = 0
            state.phase = Phase.ROLL


# ---------------------------------------------------------------------------
# Dice / resource production
# ---------------------------------------------------------------------------

def _roll_dice(state: "GameState", rng: random.Random):
    d1 = rng.randint(1, 6)
    d2 = rng.randint(1, 6)
    state.dice = (d1, d2)
    state.rolled_this_turn = True
    total = d1 + d2

    if total == 7:
        _handle_seven(state)
    else:
        _produce_resources(state, total)
        state.phase = Phase.MAIN


def _produce_resources(state: "GameState", number: int):
    """
    Compute production for all players, then pay out per resource type
    according to the official bank-shortage rule: if the bank cannot fully
    supply all players owed a resource type, no player receives that type
    (unless exactly one player is owed it — they take what's left).
    """
    geo, config = state.config.geometry, state.config
    occupied = state.all_occupied_vertices()
    owed = [[0] * 5 for _ in range(state.n_players)]   # pid -> per-resource counts
    for hex_id in range(geo.n_hexes):
        if config.hex_tokens[hex_id] != number or hex_id == state.robber_hex:
            continue
        hex_type = config.hex_resources[hex_id]
        if hex_type == HexType.DESERT:
            continue
        r = int(hex_type.to_resource())
        for v in geo.hex_to_vertices[hex_id]:
            pid = occupied.get(v)
            if pid is None:
                continue
            owed[pid][r] += 2 if v in state.players[pid].city_vertices else 1
    for r in range(5):
        demanders = [pid for pid in range(state.n_players) if owed[pid][r] > 0]
        total = sum(owed[pid][r] for pid in demanders)
        supply = state.bank[r]
        if not demanders:
            continue
        if total <= supply:
            for pid in demanders:
                state.players[pid].resources[r] += owed[pid][r]
            state.bank[r] -= total
        elif len(demanders) == 1:
            pid = demanders[0]
            state.players[pid].resources[r] += supply
            state.bank[r] = 0
        # else: shortage with multiple demanders -> nobody paid


def _handle_seven(state: "GameState"):
    """On a 7: trigger discard for players with > 7 cards, then robber move."""
    obligations = {}
    for p in state.players:
        if p.total_resources > 7:
            obligations[p.player_id] = p.total_resources // 2

    if obligations:
        state.discard_obligations = obligations
        # Set current player to first who needs to discard
        state.current_player = min(obligations.keys())
        state.phase = Phase.DISCARD
    else:
        state.phase = Phase.ROBBER


# ---------------------------------------------------------------------------
# Robber
# ---------------------------------------------------------------------------

def _main_return_phase(state: "GameState") -> Phase:
    """Where a turn-action sub-phase (robber/steal/road-building) should
    return to once resolved: MAIN if dice have already been rolled this
    turn, otherwise back to ROLL (official rule: one dev card, including a
    knight, may be played before the roll -- the player still must roll)."""
    return Phase.MAIN if state.rolled_this_turn else Phase.ROLL


def _move_robber(state: "GameState", hex_id: int):
    state.robber_hex = hex_id
    state.pending_steal_hex = hex_id

    # Check if any opponents are adjacent
    geo = state.config.geometry
    adjacent_vertices = geo.hex_to_vertices[hex_id]
    occupied = state.all_occupied_vertices()
    targets = {occupied[v] for v in adjacent_vertices
               if v in occupied and occupied[v] != state.current_player
               and state.players[occupied[v]].total_resources > 0}

    if targets:
        state.phase = Phase.STEAL
    else:
        state.pending_steal_hex = None
        # Return to appropriate phase
        if state.phase == Phase.ROBBER:
            state.phase = _main_return_phase(state)
        # (if called from knight, knight handler sets phase)


def _steal(state: "GameState", target_player_id: int, rng: random.Random):
    target = state.players[target_player_id]
    if target.total_resources > 0:
        # Pick a random resource from target
        pool = []
        for r in Resource:
            pool.extend([r] * target.resources[int(r)])
        stolen = rng.choice(pool)
        target.resources[int(stolen)] -= 1
        state.current.gain(stolen)

    state.pending_steal_hex = None
    state.phase = _main_return_phase(state)


# ---------------------------------------------------------------------------
# Discard
# ---------------------------------------------------------------------------

def _discard(state: "GameState", resource: Resource):
    pid = state.current_player
    player = state.players[pid]
    player.resources[int(resource)] -= 1
    state.bank[int(resource)] += 1
    state.discard_obligations[pid] -= 1

    if state.discard_obligations[pid] <= 0:
        del state.discard_obligations[pid]

    if state.discard_obligations:
        # Move to next player who needs to discard
        state.current_player = min(state.discard_obligations.keys())
    else:
        # All done, proceed to robber phase
        # Restore current player to whoever rolled the 7
        # We store this in turn_number context: the player who rolled is the
        # one with turn index == turn_number % n_players
        state.current_player = state.turn_number % state.n_players
        state.phase = Phase.ROBBER


# ---------------------------------------------------------------------------
# End turn
# ---------------------------------------------------------------------------

def _end_turn(state: "GameState"):
    state.current.end_turn_refresh_dev_cards()
    state.turn_number += 1
    state.current_player = state.turn_number % state.n_players
    state.dice = None
    state.rolled_this_turn = False
    state.phase = Phase.ROLL


# ---------------------------------------------------------------------------
# Maritime trade
# ---------------------------------------------------------------------------

def _maritime_trade(state: "GameState", give: Resource, get: Resource):
    player = state.current
    geo = state.config.geometry
    player_vertices = list(player.settlement_vertices | player.city_vertices)
    rate = state.config.best_trade_rate(player_vertices, give)
    player.resources[int(give)] -= rate
    state.bank[int(give)] += rate
    player.gain(get)
    state.bank[int(get)] -= 1


# ---------------------------------------------------------------------------
# Dev cards
# ---------------------------------------------------------------------------

def _buy_dev_card(state: "GameState", rng: random.Random):
    player = state.current
    player.spend(BUILD_COSTS["dev_card"])
    for r, n in BUILD_COSTS["dev_card"].items():
        state.bank[int(r)] += n
    card = state.dev_deck.pop()
    player.receive_dev_card(card)


def _play_knight(state: "GameState"):
    player = state.current
    player.play_dev_card(DevCard.KNIGHT)
    player.army_size += 1
    update_largest_army(state)
    state.phase = Phase.ROBBER


def _play_road_building(state: "GameState"):
    from .validators import _connected_road_actions
    player = state.current
    player.play_dev_card(DevCard.ROAD_BUILDING)
    if player.roads_available >= 1 and _connected_road_actions(state):
        state.phase = Phase.ROAD_BUILDING_1


def _play_year_of_plenty(state: "GameState", res_a: Resource, res_b: Resource):
    player = state.current
    player.play_dev_card(DevCard.YEAR_OF_PLENTY)
    for r in (res_a, res_b):
        if state.bank[int(r)] > 0:
            player.gain(r)
            state.bank[int(r)] -= 1


def _play_monopoly(state: "GameState", resource: Resource):
    player = state.current
    player.play_dev_card(DevCard.MONOPOLY)
    for p in state.players:
        if p.player_id == state.current_player:
            continue
        stolen = p.resources[int(resource)]
        p.resources[int(resource)] = 0
        player.gain(resource, stolen)


def _play_victory_point(state: "GameState"):
    player = state.current
    player.play_dev_card(DevCard.VICTORY_POINT)
    # VP is counted in scoring; playing it just reveals it


# ---------------------------------------------------------------------------
# Win check
# ---------------------------------------------------------------------------

def _check_win(state: "GameState"):
    if state.phase == Phase.GAME_OVER:
        return
    winner = check_winner(state)
    if winner is not None:
        state.winner = winner
        state.phase = Phase.GAME_OVER
