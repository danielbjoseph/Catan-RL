"""
Legal action generation.

legal_actions(state) -> List[Action]

The environment is the source of truth for legality.
Returns only actions with valid catalog indices (never padded slots).
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from .actions import (
    Action, Resource, DevCard,
    ROLL_DICE, END_TURN, BUY_DEV_CARD, PLAY_KNIGHT, PLAY_ROAD_BUILDING,
    road_action, settlement_action, city_action, move_robber_action,
    steal_action, maritime_trade_action, discard_action,
    year_of_plenty_action, monopoly_action,
)
from .game_state import Phase

if TYPE_CHECKING:
    from .game_state import GameState


def legal_actions(state: "GameState") -> List[Action]:
    phase = state.phase
    if phase == Phase.SETUP_SETTLEMENT_1 or phase == Phase.SETUP_SETTLEMENT_2:
        return _setup_settlement_actions(state)
    if phase == Phase.SETUP_ROAD_1 or phase == Phase.SETUP_ROAD_2:
        return _setup_road_actions(state)
    if phase == Phase.ROLL:
        # Official rule: one dev card may be played at any time during your
        # turn, including before the roll (knight-before-roll unblocks your
        # own hexes). ROLL_DICE remains mandatory eventually, but is not the
        # only legal action while a pre-roll dev card play is available.
        return [ROLL_DICE] + _dev_card_actions(state)
    if phase == Phase.ROBBER:
        return _robber_actions(state)
    if phase == Phase.STEAL:
        return _steal_actions(state)
    if phase == Phase.DISCARD:
        return _discard_actions(state)
    if phase == Phase.ROAD_BUILDING_1 or phase == Phase.ROAD_BUILDING_2:
        actions = _road_building_actions(state)
        if not actions:
            # No placements available; treat as end of road-building
            return [END_TURN]
        return actions
    if phase == Phase.MAIN:
        return _main_actions(state)
    return []


# ---------------------------------------------------------------------------
# Setup phases
# ---------------------------------------------------------------------------

def _setup_settlement_actions(state: "GameState") -> List[Action]:
    """All vertices that are empty and satisfy distance rule."""
    occupied = state.all_occupied_vertices()
    geo = state.config.geometry
    actions = []
    for v in range(geo.n_vertices):
        if v in occupied:
            continue
        # Distance rule: no neighbor may be occupied
        if any(nb in occupied for nb in geo.vertex_to_vertices[v]):
            continue
        actions.append(settlement_action(v))
    return actions


def _setup_road_actions(state: "GameState") -> List[Action]:
    """
    During setup, roads must connect to the settlement just placed
    (the most recently placed settlement of the current player with no adjacent road).
    """
    player = state.current
    geo = state.config.geometry
    all_roads = state.all_road_edges()

    # Find the settlement placed this round (the one without any road yet in setup)
    # We find it as the settlement with no adjacent road belonging to this player
    candidate = None
    for v in player.settlement_vertices:
        adjacent_edges = geo.vertex_to_edges[v]
        has_road = any(e in player.road_vertices for e in adjacent_edges)
        if not has_road:
            candidate = v
            break

    if candidate is None:
        # Fallback: allow any road connected to any of player's settlements
        candidate_vertices = player.settlement_vertices
        actions = []
        for v in candidate_vertices:
            for e in geo.vertex_to_edges[v]:
                if e not in all_roads:
                    actions.append(road_action(e))
        return actions

    actions = []
    for e in geo.vertex_to_edges[candidate]:
        if e not in all_roads:
            actions.append(road_action(e))
    return actions


# ---------------------------------------------------------------------------
# Robber / steal
# ---------------------------------------------------------------------------

def _robber_actions(state: "GameState") -> List[Action]:
    """Can move robber to any hex except current robber hex."""
    return [move_robber_action(h) for h in range(state.config.geometry.n_hexes)
            if h != state.robber_hex]


def _steal_actions(state: "GameState") -> List[Action]:
    """
    Steal from a player with a settlement/city adjacent to the hex where
    robber was just placed (pending_steal_hex).
    """
    if state.pending_steal_hex is None:
        return []
    geo = state.config.geometry
    hex_id = state.pending_steal_hex
    adjacent_vertices = geo.hex_to_vertices[hex_id]
    occupied = state.all_occupied_vertices()
    targets = set()
    for v in adjacent_vertices:
        if v in occupied:
            pid = occupied[v]
            if pid != state.current_player:
                targets.add(pid)
    # If no targets (no opponents adjacent), skip steal by returning END_TURN-like empty
    if not targets:
        return []
    return [steal_action(pid) for pid in sorted(targets)]


# ---------------------------------------------------------------------------
# Discard
# ---------------------------------------------------------------------------

def _discard_actions(state: "GameState") -> List[Action]:
    """Current player must discard one resource at a time until obligation met."""
    player = state.players[state.current_player]
    actions = []
    for r in Resource:
        if player.resources[int(r)] > 0:
            actions.append(discard_action(r))
    return actions


# ---------------------------------------------------------------------------
# Road building dev card
# ---------------------------------------------------------------------------

def _road_building_actions(state: "GameState") -> List[Action]:
    """Legal road placements during road-building dev card sub-phase."""
    return _connected_road_actions(state)


# ---------------------------------------------------------------------------
# Main turn
# ---------------------------------------------------------------------------

def _main_actions(state: "GameState") -> List[Action]:
    actions: List[Action] = []
    player = state.current
    geo = state.config.geometry

    # Build road
    if player.can_afford_road():
        for e in _connected_road_actions(state):
            actions.append(e)

    # Build settlement
    if player.can_afford_settlement():
        for a in _settlement_actions(state):
            actions.append(a)

    # Build city
    if player.can_afford_city():
        for v in player.settlement_vertices:
            actions.append(city_action(v))

    dev_enabled = state.profile.dev_cards_enabled

    # Buy dev card
    if dev_enabled and player.can_afford_dev_card() and len(state.dev_deck) > 0:
        actions.append(BUY_DEV_CARD)

    # Maritime trades
    actions.extend(_maritime_trade_actions(state))

    # Play dev cards
    actions.extend(_dev_card_actions(state))

    actions.append(END_TURN)
    return actions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dev_card_actions(state: "GameState") -> List[Action]:
    """Dev card plays currently legal for the current player: gated by
    profile (dev_cards_enabled), the one-per-turn allowance
    (has_played_dev_card), and holding the card in `dev_cards` (cards bought
    this turn live in `dev_cards_new` and are NOT playable yet).

    Shared by both the ROLL phase (official rule: one dev card may be
    played at any time during your turn, including before the roll) and the
    MAIN phase.
    """
    actions: List[Action] = []
    player = state.current
    if not state.profile.dev_cards_enabled or player.has_played_dev_card:
        return actions

    if player.dev_cards[int(DevCard.KNIGHT)] > 0:
        actions.append(PLAY_KNIGHT)
    if player.dev_cards[int(DevCard.ROAD_BUILDING)] > 0:
        actions.append(PLAY_ROAD_BUILDING)
    if player.dev_cards[int(DevCard.YEAR_OF_PLENTY)] > 0:
        for a in Resource:
            for b in Resource:
                if b >= a:
                    actions.append(year_of_plenty_action(a, b))
    if player.dev_cards[int(DevCard.MONOPOLY)] > 0:
        for r in Resource:
            actions.append(monopoly_action(r))
    # VICTORY_POINT cards are never played (see PlayerState.hidden_vp) —
    # they count toward the win condition automatically. Catalog slot
    # 253 (PLAY_VICTORY_POINT) is intentionally never appended here and
    # stays permanently masked; the apply_action handler in rules.py is
    # kept only for backcompat with old recorded traces.

    return actions


def _connected_road_actions(state: "GameState") -> List[Action]:
    """
    Roads must connect to the current player's existing road network or settlements.
    The target edge must be unoccupied.
    Roads cannot pass through an enemy settlement/city vertex (they can start at one though).
    """
    player = state.current
    geo = state.config.geometry
    all_roads = state.all_road_edges()
    all_occ = state.all_occupied_vertices()
    enemy_vertices = {v for v, pid in all_occ.items() if pid != state.current_player}

    candidate_edges = set()

    # Edges adjacent to player's roads
    for road_edge in player.road_vertices:
        va, vb = geo.edge_to_vertices[road_edge]
        for endpoint in (va, vb):
            # Cannot extend through enemy settlement
            if endpoint in enemy_vertices:
                continue
            for adj_edge in geo.vertex_to_edges[endpoint]:
                if adj_edge not in all_roads:
                    candidate_edges.add(adj_edge)

    # Edges adjacent to player's settlements/cities
    for v in player.settlement_vertices | player.city_vertices:
        for adj_edge in geo.vertex_to_edges[v]:
            if adj_edge not in all_roads:
                candidate_edges.add(adj_edge)

    return [road_action(e) for e in sorted(candidate_edges)]


def _settlement_actions(state: "GameState") -> List[Action]:
    """
    Settlements must be on player's road network, not adjacent to any building,
    and not already occupied.
    """
    player = state.current
    geo = state.config.geometry
    all_roads = state.all_road_edges()
    occupied = state.all_occupied_vertices()

    # Vertices reachable from player's road network
    reachable = set()
    for eid in player.road_vertices:
        va, vb = geo.edge_to_vertices[eid]
        reachable.add(va)
        reachable.add(vb)

    actions = []
    for v in reachable:
        if v in occupied:
            continue
        if any(nb in occupied for nb in geo.vertex_to_vertices[v]):
            continue
        actions.append(settlement_action(v))
    return actions


def _maritime_trade_actions(state: "GameState") -> List[Action]:
    """Generate all valid maritime trade actions based on resources and port access."""
    player = state.current
    geo = state.config.geometry
    config = state.config

    # All vertices owned by this player
    player_vertices = list(player.settlement_vertices | player.city_vertices)

    actions = []
    for give in Resource:
        rate = config.best_trade_rate(player_vertices, give)
        if player.resources[int(give)] >= rate:
            for get in Resource:
                if get == give:
                    continue
                if state.bank_has(get, 1):
                    actions.append(maritime_trade_action(give, get))
    return actions
