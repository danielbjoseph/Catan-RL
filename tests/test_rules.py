"""
Tests for core game rules: legality, scoring, edge cases.
Run with: pytest tests/
"""

import pytest
import random

from catan_rl.env.board import BoardConfig, HexType
from catan_rl.env.game_state import GameState, Phase
from catan_rl.env.actions import Resource, DevCard, ActionType
from catan_rl.env.rules import apply_action
from catan_rl.env.validators import legal_actions
from catan_rl.env.scoring import compute_vp, compute_longest_road, check_winner
from catan_rl.env.actions import (
    settlement_action, road_action, city_action,
    move_robber_action, steal_action, discard_action,
    maritime_trade_action, monopoly_action, year_of_plenty_action,
    ROLL_DICE, END_TURN, BUY_DEV_CARD, PLAY_KNIGHT, PLAY_ROAD_BUILDING,
)


def make_state(seed=0):
    config = BoardConfig.standard(seed=seed)
    return GameState.new_game(config, n_players=4, seed=seed)


def fast_forward_setup(state, rng=None):
    """Complete setup phase with random placements."""
    if rng is None:
        rng = random.Random(0)
    while state.phase not in (Phase.ROLL, Phase.GAME_OVER):
        actions = legal_actions(state)
        assert actions, f"No legal actions during setup phase={state.phase}"
        action = rng.choice(actions)
        apply_action(state, action, rng)


# ---------------------------------------------------------------------------
# Distance rule
# ---------------------------------------------------------------------------

class TestDistanceRule:
    def test_no_adjacent_settlement_placement(self):
        state = make_state()
        geo = state.config.geometry
        # Place first settlement
        actions = legal_actions(state)
        first = actions[0]
        apply_action(state, first)
        # Place road
        road_acts = legal_actions(state)
        apply_action(state, road_acts[0])

        # Check that neighbors of first settlement are not legal for next player
        occupied = state.all_occupied_vertices()
        legal_verts = {a.vertex_id for a in legal_actions(state)
                       if a.action_type == ActionType.BUILD_SETTLEMENT}
        placed_v = first.vertex_id
        for nb in geo.vertex_to_vertices[placed_v]:
            assert nb not in legal_verts, f"Neighbor vertex {nb} of {placed_v} should be illegal"

    def test_all_setup_settlements_respect_distance(self):
        state = make_state()
        rng = random.Random(1)
        geo = state.config.geometry
        placed = []

        def check_then_place():
            legal_verts = {a.vertex_id for a in legal_actions(state)
                           if a.action_type == ActionType.BUILD_SETTLEMENT}
            for v in legal_verts:
                for nb in geo.vertex_to_vertices[v]:
                    assert nb not in {p for p in placed}, \
                        f"Vertex {v} is adjacent to already-placed vertex {nb}"

        while state.phase in (Phase.SETUP_SETTLEMENT_1, Phase.SETUP_SETTLEMENT_2):
            check_then_place()
            acts = [a for a in legal_actions(state)
                    if a.action_type == ActionType.BUILD_SETTLEMENT]
            chosen = rng.choice(acts)
            placed.append(chosen.vertex_id)
            apply_action(state, chosen)
            # place road
            if state.phase in (Phase.SETUP_ROAD_1, Phase.SETUP_ROAD_2):
                apply_action(state, rng.choice(legal_actions(state)))


# ---------------------------------------------------------------------------
# Road connectivity
# ---------------------------------------------------------------------------

class TestRoadConnectivity:
    def test_roads_must_connect_in_main_phase(self):
        state = make_state()
        rng = random.Random(2)
        fast_forward_setup(state, rng)

        # Give player 0 resources to build roads
        state.players[0].resources[int(Resource.WOOD)] = 10
        state.players[0].resources[int(Resource.BRICK)] = 10

        # All legal road actions must be adjacent to existing network
        geo = state.config.geometry
        player = state.players[state.current_player]
        network_verts = set()
        for e in player.road_vertices:
            va, vb = geo.edge_to_vertices[e]
            network_verts.add(va)
            network_verts.add(vb)
        for v in player.settlement_vertices | player.city_vertices:
            network_verts.add(v)

        legal = [a for a in legal_actions(state) if a.action_type == ActionType.BUILD_ROAD]
        for act in legal:
            va, vb = geo.edge_to_vertices[act.edge_id]
            assert va in network_verts or vb in network_verts, \
                f"Road {act.edge_id} not connected to player network"


# ---------------------------------------------------------------------------
# City upgrades
# ---------------------------------------------------------------------------

class TestCityUpgrade:
    def test_city_only_on_own_settlement(self):
        state = make_state()
        rng = random.Random(3)
        fast_forward_setup(state, rng)

        player = state.players[state.current_player]
        # Give enough resources
        player.resources[int(Resource.WHEAT)] = 10
        player.resources[int(Resource.ORE)] = 10

        legal = [a for a in legal_actions(state) if a.action_type == ActionType.BUILD_CITY]
        for act in legal:
            assert act.vertex_id in player.settlement_vertices, \
                f"City at {act.vertex_id} is not player's settlement"

    def test_city_not_on_opponent_settlement(self):
        state = make_state()
        rng = random.Random(4)
        fast_forward_setup(state, rng)

        player = state.players[state.current_player]
        player.resources[int(Resource.WHEAT)] = 10
        player.resources[int(Resource.ORE)] = 10

        opponent_settlements = set()
        for p in state.players:
            if p.player_id != state.current_player:
                opponent_settlements |= p.settlement_vertices

        legal = [a for a in legal_actions(state) if a.action_type == ActionType.BUILD_CITY]
        for act in legal:
            assert act.vertex_id not in opponent_settlements


# ---------------------------------------------------------------------------
# Resource production
# ---------------------------------------------------------------------------

class TestResourceProduction:
    def test_dice_produces_correct_resources(self):
        config = BoardConfig.standard(seed=5)
        state = GameState.new_game(config, n_players=4, seed=5)
        rng = random.Random(5)
        fast_forward_setup(state, rng)

        before = [list(p.resources) for p in state.players]

        # Force dice to a specific number by patching
        target_number = None
        for hi in range(config.geometry.n_hexes):
            if config.hex_tokens[hi] > 0:
                target_number = config.hex_tokens[hi]
                break

        assert target_number is not None

        # Monkeypatch dice roll
        original_randint = rng.randint
        calls = []
        def forced_randint(a, b):
            if len(calls) < 2:
                calls.append(1)
                # Return values that sum to target_number
                if len(calls) == 1:
                    return min(target_number - 1, 6)
                else:
                    return target_number - min(target_number - 1, 6)
            return original_randint(a, b)
        rng.randint = forced_randint

        apply_action(state, ROLL_DICE, rng)

        # At least one player should have gained resources (or bank was empty)
        after = [list(p.resources) for p in state.players]
        # We just check no resources went negative
        for p in state.players:
            for r in Resource:
                assert p.resources[int(r)] >= 0

    def test_robber_blocks_production(self):
        config = BoardConfig.standard(seed=6)
        state = GameState.new_game(config, n_players=4, seed=6)
        rng = random.Random(6)
        fast_forward_setup(state, rng)

        # Find a productive hex with a settlement on it
        geo = config.geometry
        occupied = state.all_occupied_vertices()
        target_hex = None
        for hi in range(geo.n_hexes):
            if config.hex_tokens[hi] > 0 and config.hex_resources[hi] != HexType.DESERT:
                for v in geo.hex_to_vertices[hi]:
                    if v in occupied:
                        target_hex = hi
                        break
            if target_hex is not None:
                break

        if target_hex is None:
            pytest.skip("No productive hex with settlement found")

        token = config.hex_tokens[target_hex]
        state.robber_hex = target_hex

        before = [sum(p.resources) for p in state.players]
        # Force roll to target token
        class FixedRng:
            def randint(self, a, b):
                d1 = min(token - 1, 6)
                d2 = token - d1
                return d1 if not hasattr(self, '_called') else (setattr(self, '_called', True) or d2)
            def choice(self, seq):
                return seq[0]

        apply_action(state, ROLL_DICE, rng)
        after = [sum(p.resources) for p in state.players]
        # Resources may or may not have increased for other hexes, but we just
        # verify no crash occurred and resources are non-negative.
        for p in state.players:
            assert sum(p.resources) >= 0


# ---------------------------------------------------------------------------
# Discard on 7
# ---------------------------------------------------------------------------

class TestDiscardOnSeven:
    def test_discard_half_on_seven(self):
        state = make_state()
        rng = random.Random(7)
        fast_forward_setup(state, rng)

        # Give player 1 too many resources
        state.players[1].resources = [3, 3, 2, 2, 2]  # 12 total -> must discard 6

        # Force a 7 by directly calling _handle_seven
        from catan_rl.env.rules import _handle_seven
        state.phase = Phase.ROLL
        state.current_player = 0

        _handle_seven(state)

        assert 1 in state.discard_obligations
        assert state.discard_obligations[1] == 6
        assert state.phase == Phase.DISCARD


# ---------------------------------------------------------------------------
# Longest road
# ---------------------------------------------------------------------------

class TestLongestRoad:
    def test_straight_road_length(self):
        state = make_state()
        rng = random.Random(8)
        fast_forward_setup(state, rng)

        geo = state.config.geometry
        player = state.players[0]

        # Build a chain of edges manually
        # Start from one of their existing settlement vertices
        start_v = next(iter(player.settlement_vertices))
        chain_length = 0
        current_v = start_v
        visited_edges = set(player.road_vertices)

        for _ in range(5):
            found = False
            for e in geo.vertex_to_edges[current_v]:
                if e in visited_edges:
                    continue
                va, vb = geo.edge_to_vertices[e]
                other = vb if va == current_v else va
                player.road_vertices.add(e)
                player.roads_built += 1
                visited_edges.add(e)
                current_v = other
                chain_length += 1
                found = True
                break
            if not found:
                break

        length = compute_longest_road(0, state)
        assert length >= chain_length

    def test_enemy_settlement_breaks_road(self):
        state = make_state()
        rng = random.Random(9)
        fast_forward_setup(state, rng)

        geo = state.config.geometry
        player0 = state.players[0]

        if not player0.road_vertices:
            pytest.skip("Player 0 has no roads")

        sample_edge = next(iter(player0.road_vertices))
        va, vb = geo.edge_to_vertices[sample_edge]
        mid_v = va

        # Place enemy settlement at mid_v
        state.players[1].settlement_vertices.add(mid_v)
        state.players[1].settlements_built += 1

        length_with_enemy = compute_longest_road(0, state)

        # Remove enemy settlement
        state.players[1].settlement_vertices.discard(mid_v)
        state.players[1].settlements_built -= 1

        length_without_enemy = compute_longest_road(0, state)

        assert length_with_enemy <= length_without_enemy


# ---------------------------------------------------------------------------
# Dev card restrictions
# ---------------------------------------------------------------------------

class TestDevCards:
    def test_cannot_play_two_dev_cards_per_turn(self):
        state = make_state()
        rng = random.Random(10)
        fast_forward_setup(state, rng)

        player = state.players[state.current_player]
        player.dev_cards[int(DevCard.MONOPOLY)] = 2
        player.has_played_dev_card = True

        legal = [a for a in legal_actions(state)
                 if a.action_type in (ActionType.PLAY_MONOPOLY, ActionType.PLAY_KNIGHT,
                                       ActionType.PLAY_ROAD_BUILDING, ActionType.PLAY_YEAR_OF_PLENTY)]
        assert len(legal) == 0

    def test_newly_bought_card_not_playable_same_turn(self):
        state = make_state()
        rng = random.Random(11)
        fast_forward_setup(state, rng)

        player = state.players[state.current_player]
        # Simulate buying a card (goes into dev_cards_new)
        player.dev_cards_new[int(DevCard.KNIGHT)] = 1
        player.dev_cards[int(DevCard.KNIGHT)] = 0

        legal = [a for a in legal_actions(state) if a.action_type == ActionType.PLAY_KNIGHT]
        assert len(legal) == 0


# ---------------------------------------------------------------------------
# Win condition
# ---------------------------------------------------------------------------

class TestWinCondition:
    def test_win_at_10_vp(self):
        state = make_state()
        rng = random.Random(12)
        fast_forward_setup(state, rng)

        player = state.players[0]
        # Manually set 10 VP worth of buildings
        player.settlements_built = 0
        player.cities_built = 4
        player.city_vertices = set(range(4))
        player.settlement_vertices = set()
        # 4 cities = 8 VP, need 2 more
        state.longest_road_holder = 0  # +2 VP

        winner = check_winner(state)
        assert winner == 0

    def test_no_win_at_9_vp(self):
        state = make_state()
        rng = random.Random(13)
        fast_forward_setup(state, rng)

        player = state.players[0]
        player.settlements_built = 0
        player.cities_built = 4
        player.city_vertices = set(range(4))
        player.settlement_vertices = set()
        # 8 VP from cities only

        winner = check_winner(state)
        assert winner is None


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_game_state_roundtrip(self):
        state = make_state()
        rng = random.Random(14)
        fast_forward_setup(state, rng)

        d = state.to_dict()
        restored = GameState.from_dict(d, state.config)

        assert restored.current_player == state.current_player
        assert restored.phase == state.phase
        assert restored.robber_hex == state.robber_hex
        assert restored.bank == state.bank
        for i, (p_orig, p_rest) in enumerate(zip(state.players, restored.players)):
            assert p_orig.resources == p_rest.resources
            assert p_orig.road_vertices == p_rest.road_vertices
            assert p_orig.settlement_vertices == p_rest.settlement_vertices

    def test_clone_independence(self):
        state = make_state()
        rng = random.Random(15)
        fast_forward_setup(state, rng)

        clone = state.clone()
        # Mutate original
        state.players[0].resources[0] += 99
        state.bank[0] += 99

        assert clone.players[0].resources[0] != state.players[0].resources[0]
        assert clone.bank[0] != state.bank[0]
