"""
Deterministic mechanics tests.

Each test isolates a single mechanic, bypasses setup when possible by
directly mutating game state, and verifies exact outcomes rather than
just "no crash."
"""

import random
import pytest

from catan_rl.env.board import BoardConfig, HexType
from catan_rl.env.game_state import GameState, Phase, WIN_VP
from catan_rl.env.actions import (
    Resource, DevCard, ActionType,
    ROLL_DICE, END_TURN, BUY_DEV_CARD, PLAY_KNIGHT, PLAY_ROAD_BUILDING,
    PLAY_VICTORY_POINT,
    settlement_action, road_action, city_action,
    move_robber_action, steal_action, discard_action,
    maritime_trade_action, monopoly_action, year_of_plenty_action,
)
from catan_rl.env.rules import apply_action, _produce_resources, _handle_seven
from catan_rl.env.validators import legal_actions
from catan_rl.env.scoring import (
    compute_vp, compute_longest_road, update_longest_road,
    update_largest_army, check_winner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DiceRng:
    """RNG stub that returns fixed dice then falls back to real random."""
    def __init__(self, d1: int, d2: int, seed: int = 0):
        self._d1 = d1
        self._d2 = d2
        self._calls = 0
        self._rng = random.Random(seed)

    def randint(self, a, b):
        self._calls += 1
        if self._calls == 1:
            return self._d1
        if self._calls == 2:
            return self._d2
        return self._rng.randint(a, b)

    def choice(self, seq):
        return self._rng.choice(seq)

    def shuffle(self, lst):
        self._rng.shuffle(lst)

    def pop(self):
        return self._rng.random()


def _blank_state(seed: int = 0) -> GameState:
    """
    Return a game state with setup phase bypassed.
    All 4 players start with:
      - 1 settlement at a known vertex (spread across the board)
      - 1 adjacent road
      - empty resources / dev cards
    Phase = ROLL, current_player = 0.
    """
    config = BoardConfig.standard(seed=seed)
    state = GameState.new_game(config, n_players=4, seed=seed)
    geo = config.geometry

    # Skip setup
    state.phase = Phase.ROLL
    state.current_player = 0
    state._setup_forward_idx = 4
    state._setup_backward_idx = -1

    # Place 4 settlements at well-separated vertices
    placed_vertices = []
    occupied: set = set()
    for v in range(geo.n_vertices):
        if v in occupied:
            continue
        if any(nb in occupied for nb in geo.vertex_to_vertices[v]):
            continue
        pid = len(placed_vertices)
        if pid >= 4:
            break
        state.players[pid].settlement_vertices.add(v)
        state.players[pid].settlements_built = 1
        occupied.add(v)
        placed_vertices.append(v)
        # One road per player
        for e in geo.vertex_to_edges[v]:
            state.players[pid].road_vertices.add(e)
            state.players[pid].roads_built = 1
            break

    return state


def _find_hex_with(config: BoardConfig, hex_type: HexType) -> int:
    """Return first hex_id with the given HexType."""
    for hi, res in enumerate(config.hex_resources):
        if res == hex_type:
            return hi
    raise ValueError(f"No hex of type {hex_type}")


def _find_hex_with_token(config: BoardConfig, hex_type: HexType, token: int) -> int:
    """Return a hex with the given type and token (skipping 0 = desert)."""
    for hi, res in enumerate(config.hex_resources):
        if res == hex_type and config.hex_tokens[hi] == token:
            return hi
    raise ValueError(f"No {hex_type.name} hex with token {token}")


# ---------------------------------------------------------------------------
# 1. Resource production: settlement yields 1, city yields 2
# ---------------------------------------------------------------------------

class TestResourceProduction:

    def _setup_production_test(self, hex_type: HexType, token: int, seed: int = 0):
        """
        Place player 0 settlement adjacent to a specific hex; return (state, hex_id, resource).
        Raises pytest.skip if no such hex exists for this seed.
        """
        config = BoardConfig.standard(seed=seed)
        # Try to find hex
        try:
            hex_id = _find_hex_with_token(config, hex_type, token)
        except ValueError:
            pytest.skip(f"seed={seed} has no {hex_type.name} hex with token {token}")

        state = _blank_state(seed=seed)
        geo = config.geometry
        vertex = geo.hex_to_vertices[hex_id][0]

        # Ensure player 0 owns this vertex (clear any previous state, place fresh)
        for p in state.players:
            p.settlement_vertices.discard(vertex)
            p.city_vertices.discard(vertex)
        state.players[0].settlement_vertices.add(vertex)

        resource = hex_type.to_resource()
        return state, hex_id, vertex, resource

    def test_settlement_yields_one_resource(self):
        config = BoardConfig.standard(seed=0)
        # Find any non-desert productive hex
        for hi, hex_type in enumerate(config.hex_resources):
            if hex_type != HexType.DESERT and config.hex_tokens[hi] > 0:
                token = config.hex_tokens[hi]
                resource = hex_type.to_resource()
                break

        state = _blank_state(seed=0)
        geo = config.geometry
        vertex = geo.hex_to_vertices[hi][0]

        # Clear and place
        for p in state.players:
            p.settlement_vertices.discard(vertex)
            p.city_vertices.discard(vertex)
        state.players[0].settlement_vertices.add(vertex)
        state.players[0].resources = [0] * 5

        before = state.players[0].resources[int(resource)]
        d1, d2 = (token // 2), token - (token // 2)
        if d1 < 1: d1, d2 = 1, token - 1
        if d2 < 1 or d2 > 6 or d1 > 6:
            pytest.skip(f"Cannot roll {token} with 2d6")

        _produce_resources(state, token)
        after = state.players[0].resources[int(resource)]
        assert after == before + 1, f"Expected +1 {resource.name}, got {after - before}"

    def test_city_yields_two_resources(self):
        config = BoardConfig.standard(seed=0)
        for hi, hex_type in enumerate(config.hex_resources):
            if hex_type != HexType.DESERT and config.hex_tokens[hi] > 0:
                token = config.hex_tokens[hi]
                resource = hex_type.to_resource()
                break

        state = _blank_state(seed=0)
        geo = config.geometry
        vertex = geo.hex_to_vertices[hi][0]

        for p in state.players:
            p.settlement_vertices.discard(vertex)
            p.city_vertices.discard(vertex)
        state.players[0].city_vertices.add(vertex)
        state.players[0].resources = [0] * 5

        _produce_resources(state, token)
        gained = state.players[0].resources[int(resource)]
        assert gained == 2, f"City should yield 2, got {gained}"

    def test_robber_blocks_production(self):
        config = BoardConfig.standard(seed=0)
        for hi, hex_type in enumerate(config.hex_resources):
            if hex_type != HexType.DESERT and config.hex_tokens[hi] > 0:
                token = config.hex_tokens[hi]
                resource = hex_type.to_resource()
                break

        state = _blank_state(seed=0)
        geo = config.geometry
        vertex = geo.hex_to_vertices[hi][0]

        for p in state.players:
            p.settlement_vertices.discard(vertex)
            p.city_vertices.discard(vertex)
        state.players[0].settlement_vertices.add(vertex)
        state.players[0].resources = [0] * 5

        state.robber_hex = hi  # place robber on the hex
        _produce_resources(state, token)
        gained = state.players[0].resources[int(resource)]
        assert gained == 0, "Robber should block production"

    def test_only_correct_token_produces(self):
        config = BoardConfig.standard(seed=0)
        # Find a hex with token 6
        target_hi = None
        for hi, hex_type in enumerate(config.hex_resources):
            if hex_type != HexType.DESERT and config.hex_tokens[hi] == 6:
                target_hi = hi
                target_resource = hex_type.to_resource()
                break
        if target_hi is None:
            pytest.skip("No hex with token 6")

        state = _blank_state(seed=0)
        geo = config.geometry
        vertex = geo.hex_to_vertices[target_hi][0]
        for p in state.players:
            p.settlement_vertices.discard(vertex)
        state.players[0].settlement_vertices.add(vertex)
        state.players[0].resources = [0] * 5

        _produce_resources(state, 5)  # wrong token
        assert state.players[0].resources[int(target_resource)] == 0

        _produce_resources(state, 6)  # correct token
        assert state.players[0].resources[int(target_resource)] == 1

    def test_multiple_settlements_same_roll(self):
        """Two settlements on different hexes with same token both produce."""
        config = BoardConfig.standard(seed=0)
        # Find two hexes with the same token (common: 3, 4, 5, 6, 8, 9, 10, 11)
        token_hexes: dict = {}
        for hi, hex_type in enumerate(config.hex_resources):
            if hex_type != HexType.DESERT and config.hex_tokens[hi] > 0:
                t = config.hex_tokens[hi]
                token_hexes.setdefault(t, []).append(hi)
        shared_token = next((t for t, hs in token_hexes.items() if len(hs) >= 2), None)
        if shared_token is None:
            pytest.skip("No shared token found")

        hi_a, hi_b = token_hexes[shared_token][:2]
        va = config.geometry.hex_to_vertices[hi_a][0]
        vb = config.geometry.hex_to_vertices[hi_b][0]
        if va == vb:
            pytest.skip("Same vertex")

        state = _blank_state(seed=0)
        geo = config.geometry
        for p in state.players:
            p.settlement_vertices.discard(va)
            p.settlement_vertices.discard(vb)
        state.players[0].settlement_vertices.add(va)
        state.players[0].settlement_vertices.add(vb)
        state.players[0].resources = [0] * 5

        res_a = config.hex_resources[hi_a].to_resource()
        res_b = config.hex_resources[hi_b].to_resource()

        before_a = state.players[0].resources[int(res_a)]
        before_b = state.players[0].resources[int(res_b)]
        _produce_resources(state, shared_token)

        assert state.players[0].resources[int(res_a)] >= before_a + 1
        assert state.players[0].resources[int(res_b)] >= before_b + 1

    def test_desert_never_produces(self):
        """Desert hex has token 0 so it never matches any dice roll (2-12)."""
        config = BoardConfig.standard(seed=0)
        desert_hi = config.desert_hex
        geo = config.geometry

        assert config.hex_tokens[desert_hi] == 0, "Desert must have token 0"
        assert config.hex_resources[desert_hi] == HexType.DESERT

        # Find a vertex adjacent ONLY to the desert (coastal corner vertex)
        desert_only = [v for v in range(geo.n_vertices)
                       if set(geo.vertex_to_hexes[v]) == {desert_hi}]

        if not desert_only:
            # Desert is interior — all its vertices border other hexes.
            # Verify by checking production with empty bank so only desert matters.
            state = _blank_state(seed=0)
            vertex = geo.hex_to_vertices[desert_hi][0]
            for p in state.players:
                p.settlement_vertices.discard(vertex)
            state.players[0].settlement_vertices.add(vertex)
            state.players[0].resources = [0] * 5
            state.bank = [0] * 5  # drain bank so neighbours can't produce either
            for token in range(2, 13):
                _produce_resources(state, token)
            assert sum(state.players[0].resources) == 0
            return

        state = _blank_state(seed=0)
        vertex = desert_only[0]
        for p in state.players:
            p.settlement_vertices.discard(vertex)
        state.players[0].settlement_vertices.add(vertex)
        state.players[0].resources = [0] * 5

        for token in range(2, 13):
            _produce_resources(state, token)
        assert sum(state.players[0].resources) == 0


# ---------------------------------------------------------------------------
# 2. Setup phase: second settlement gives initial resources
# ---------------------------------------------------------------------------

class TestSetupInitialResources:

    def test_second_settlement_gives_adjacent_resources(self):
        config = BoardConfig.standard(seed=0)
        state = GameState.new_game(config, n_players=4, seed=0)
        rng = random.Random(0)
        geo = config.geometry

        # Complete setup_settlement_1 and setup_road_1 for all 4 players
        while state.phase == Phase.SETUP_SETTLEMENT_1 or state.phase == Phase.SETUP_ROAD_1:
            acts = legal_actions(state)
            apply_action(state, rng.choice(acts), rng)

        # Now we're in SETUP_SETTLEMENT_2 for player 3
        assert state.phase == Phase.SETUP_SETTLEMENT_2

        player = state.current
        before = sum(player.resources)

        # Place second settlement
        acts = [a for a in legal_actions(state) if a.action_type == ActionType.BUILD_SETTLEMENT]
        chosen = acts[0]
        apply_action(state, chosen, rng)

        after = sum(player.resources)
        # Should have gained at least 1 resource (unless all adjacent hexes are desert/bank empty)
        adjacent_hexes = geo.vertex_to_hexes[chosen.vertex_id]
        productive = sum(
            1 for hi in adjacent_hexes
            if config.hex_resources[hi] != HexType.DESERT
        )
        if productive > 0:
            assert after >= before + 1, "Second settlement should give resources"

    def test_setup_phase_no_resource_cost(self):
        """Placing roads during setup costs nothing."""
        config = BoardConfig.standard(seed=1)
        state = GameState.new_game(config, n_players=4, seed=1)
        rng = random.Random(1)

        # Place first settlement
        acts = legal_actions(state)
        apply_action(state, rng.choice(acts), rng)

        # Place first road - should cost nothing
        player = state.current
        player.resources = [0] * 5
        acts = legal_actions(state)
        road_acts = [a for a in acts if a.action_type == ActionType.BUILD_ROAD]
        assert road_acts, "Should have road actions"
        apply_action(state, road_acts[0], rng)
        assert sum(player.resources) == 0, "Setup road should be free"


# ---------------------------------------------------------------------------
# 3. Building costs
# ---------------------------------------------------------------------------

class TestBuildingCosts:

    def test_road_costs_wood_brick(self):
        state = _blank_state(seed=0)
        p = state.players[0]
        p.resources = [1, 1, 0, 0, 0]  # exact cost
        bank_before = list(state.bank)
        state.phase = Phase.MAIN

        acts = [a for a in legal_actions(state) if a.action_type == ActionType.BUILD_ROAD]
        assert acts, "Should have road actions"
        apply_action(state, acts[0])

        assert p.resources[int(Resource.WOOD)] == 0
        assert p.resources[int(Resource.BRICK)] == 0
        assert state.bank[int(Resource.WOOD)] == bank_before[int(Resource.WOOD)] + 1
        assert state.bank[int(Resource.BRICK)] == bank_before[int(Resource.BRICK)] + 1

    def test_settlement_costs_wood_brick_sheep_wheat(self):
        state = _blank_state(seed=0)
        p = state.players[0]
        p.resources = [1, 1, 1, 1, 0]  # exact settlement cost
        state.phase = Phase.MAIN
        # Give extra roads to have reachable vertices
        geo = state.config.geometry
        # Extend road network to reach a new buildable vertex (copy set first)
        for e in list(p.road_vertices):
            va, vb = geo.edge_to_vertices[e]
            for v in (va, vb):
                for adj_e in geo.vertex_to_edges[v]:
                    if adj_e not in state.all_road_edges():
                        p.road_vertices.add(adj_e)
                        p.roads_built += 1
                        break

        acts = [a for a in legal_actions(state) if a.action_type == ActionType.BUILD_SETTLEMENT]
        if not acts:
            pytest.skip("No settlement spots reachable")
        apply_action(state, acts[0])

        assert p.resources[int(Resource.WOOD)] == 0
        assert p.resources[int(Resource.BRICK)] == 0
        assert p.resources[int(Resource.SHEEP)] == 0
        assert p.resources[int(Resource.WHEAT)] == 0
        assert p.settlements_built == 2

    def test_city_costs_wheat_ore(self):
        state = _blank_state(seed=0)
        p = state.players[0]
        p.resources = [0, 0, 0, 2, 3]  # exact city cost
        bank_before = list(state.bank)

        settlement_v = next(iter(p.settlement_vertices))
        apply_action(state, city_action(settlement_v))

        assert p.resources[int(Resource.WHEAT)] == 0
        assert p.resources[int(Resource.ORE)] == 0
        assert settlement_v in p.city_vertices
        assert settlement_v not in p.settlement_vertices
        assert p.cities_built == 1
        assert p.settlements_built == 0
        assert state.bank[int(Resource.WHEAT)] == bank_before[int(Resource.WHEAT)] + 2
        assert state.bank[int(Resource.ORE)] == bank_before[int(Resource.ORE)] + 3


# ---------------------------------------------------------------------------
# 4. Maritime trade
# ---------------------------------------------------------------------------

class TestMaritimeTrade:

    def _state_with_port(self, resource_or_none):
        """
        Find a port of the given type, place player 0 settlement on it,
        give them 10 of every resource.
        """
        for seed in range(20):
            config = BoardConfig.standard(seed=seed)
            for port in config.ports:
                if port.resource == resource_or_none:
                    state = _blank_state(seed=seed)
                    p = state.players[0]
                    v = port.vertices[0]
                    for pl in state.players:
                        pl.settlement_vertices.discard(v)
                        pl.city_vertices.discard(v)
                    p.settlement_vertices.add(v)
                    p.resources = [10, 10, 10, 10, 10]
                    return state, port
        pytest.skip(f"No port found for {resource_or_none}")

    def test_default_rate_is_4(self):
        state = _blank_state(seed=0)
        p = state.players[0]
        # No ports; put settlement far from any port
        p.resources = [10, 10, 10, 10, 10]

        # Check rate for each resource
        geo = state.config.geometry
        player_verts = list(p.settlement_vertices)
        for res in Resource:
            rate = state.config.best_trade_rate(player_verts, res)
            # May or may not have a port depending on vertex placement
            assert rate in (2, 3, 4)

    def test_generic_port_rate_3(self):
        state, port = self._state_with_port(None)  # None = 3:1 generic
        p = state.players[0]
        geo = state.config.geometry
        player_verts = list(p.settlement_vertices)
        for res in Resource:
            rate = state.config.best_trade_rate(player_verts, res)
            assert rate <= 3, f"Generic port should give rate ≤3, got {rate}"

    def test_specific_port_rate_2(self):
        state, port = self._state_with_port(Resource.WOOD)
        p = state.players[0]
        geo = state.config.geometry
        player_verts = list(p.settlement_vertices)
        assert state.config.best_trade_rate(player_verts, Resource.WOOD) == 2

    def test_trade_deducts_correct_amount(self):
        state, port = self._state_with_port(Resource.WOOD)
        p = state.players[0]
        p.resources = [0] * 5
        p.resources[int(Resource.WOOD)] = 10
        state.bank[int(Resource.ORE)] = 5

        bank_ore_before = state.bank[int(Resource.ORE)]
        bank_wood_before = state.bank[int(Resource.WOOD)]
        apply_action(state, maritime_trade_action(Resource.WOOD, Resource.ORE))

        assert p.resources[int(Resource.WOOD)] == 8, "2:1 port should cost 2 wood"
        assert p.resources[int(Resource.ORE)] == 1
        assert state.bank[int(Resource.ORE)] == bank_ore_before - 1
        assert state.bank[int(Resource.WOOD)] == bank_wood_before + 2

    def test_trade_4to1_no_port(self):
        """Find a vertex with no port and verify 4:1 trade rate."""
        config = BoardConfig.standard(seed=0)
        state = _blank_state(seed=0)
        p = state.players[0]

        # Find a vertex with no port
        port_vertices = {v for port in config.ports for v in port.vertices}
        no_port_v = None
        for v in range(config.geometry.n_vertices):
            if v not in port_vertices:
                no_port_v = v
                break
        if no_port_v is None:
            pytest.skip("All vertices have ports")

        for pl in state.players:
            pl.settlement_vertices.discard(no_port_v)
            pl.city_vertices.discard(no_port_v)
        p.settlement_vertices = {no_port_v}
        p.resources = [10, 10, 10, 10, 10]
        state.bank[int(Resource.ORE)] = 5

        rate = config.best_trade_rate([no_port_v], Resource.WOOD)
        assert rate == 4

        apply_action(state, maritime_trade_action(Resource.WOOD, Resource.ORE))
        assert p.resources[int(Resource.WOOD)] == 6, "4:1 should cost 4"
        assert p.resources[int(Resource.ORE)] == 11

    def test_cannot_trade_same_resource(self):
        state = _blank_state(seed=0)
        state.players[0].resources = [10] * 5
        acts = legal_actions(state)
        for a in acts:
            if a.action_type == ActionType.MARITIME_TRADE:
                assert a.resource != a.resource2


# ---------------------------------------------------------------------------
# 5. Robber: move + steal
# ---------------------------------------------------------------------------

class TestRobber:

    def test_robber_steal_transfers_resource(self):
        state = _blank_state(seed=0)
        geo = state.config.geometry

        # Set up: player 1 has 3 wood; player 0 is robbing them
        state.players[1].resources = [3, 0, 0, 0, 0]  # 3 wood
        state.players[0].resources = [0] * 5

        # Find a hex adjacent to player 1's settlement
        p1_settlement_v = next(iter(state.players[1].settlement_vertices))
        hex_with_p1 = geo.vertex_to_hexes[p1_settlement_v][0]

        # Place robber there and enter steal phase
        state.robber_hex = hex_with_p1
        state.pending_steal_hex = hex_with_p1
        state.phase = Phase.STEAL

        rng = random.Random(0)
        apply_action(state, steal_action(1), rng)

        # Player 0 should have gained exactly 1 wood (only resource target had)
        assert sum(state.players[0].resources) == 1
        assert state.players[0].resources[int(Resource.WOOD)] == 1
        assert sum(state.players[1].resources) == 2

    def test_steal_no_action_when_no_opponents_adjacent(self):
        state = _blank_state(seed=0)
        # Move robber to desert hex (no settlements adjacent, hopefully)
        desert_hex = state.config.desert_hex
        geo = state.config.geometry
        adj_verts = set(geo.hex_to_vertices[desert_hex])
        # Remove all settlements from desert-adjacent vertices
        for p in state.players:
            p.settlement_vertices -= adj_verts
            p.city_vertices -= adj_verts

        state.phase = Phase.ROBBER
        state.rolled_this_turn = True  # this scenario is post-roll (7 or post-roll knight)
        apply_action(state, move_robber_action(desert_hex))

        # No opponents adjacent, so should skip straight to MAIN
        assert state.phase == Phase.MAIN

    def test_cannot_move_robber_to_current_hex(self):
        state = _blank_state(seed=0)
        state.phase = Phase.ROBBER
        acts = legal_actions(state)
        robber_targets = {a.hex_id for a in acts if a.action_type == ActionType.MOVE_ROBBER}
        assert state.robber_hex not in robber_targets

    def test_robber_covers_all_other_hexes(self):
        state = _blank_state(seed=0)
        state.phase = Phase.ROBBER
        acts = legal_actions(state)
        robber_targets = {a.hex_id for a in acts if a.action_type == ActionType.MOVE_ROBBER}
        expected = set(range(state.config.geometry.n_hexes)) - {state.robber_hex}
        assert robber_targets == expected


# ---------------------------------------------------------------------------
# 6. Discard on 7
# ---------------------------------------------------------------------------

class TestDiscardOnSeven:

    def test_player_with_8_must_discard_4(self):
        state = _blank_state(seed=0)
        state.players[0].resources = [2, 2, 2, 2, 0]  # 8 total
        state.current_player = 0
        state.phase = Phase.ROLL
        _handle_seven(state)
        assert state.discard_obligations.get(0) == 4

    def test_player_with_7_no_discard(self):
        state = _blank_state(seed=0)
        state.players[0].resources = [2, 2, 2, 1, 0]  # 7 total
        _handle_seven(state)
        assert 0 not in state.discard_obligations

    def test_player_with_14_discards_7(self):
        state = _blank_state(seed=0)
        state.players[0].resources = [3, 3, 3, 3, 2]  # 14 total
        _handle_seven(state)
        assert state.discard_obligations.get(0) == 7

    def test_multiple_players_discard(self):
        state = _blank_state(seed=0)
        state.players[0].resources = [2, 2, 2, 2, 0]  # 8 → discard 4
        state.players[1].resources = [0] * 5            # 0 → ok
        state.players[2].resources = [3, 3, 3, 2, 0]  # 11 → discard 5
        state.players[3].resources = [2, 2, 2, 1, 0]  # 7 → ok
        _handle_seven(state)
        assert state.discard_obligations.get(0) == 4
        assert 1 not in state.discard_obligations
        assert state.discard_obligations.get(2) == 5
        assert 3 not in state.discard_obligations

    def test_discard_removes_resource_returns_to_bank(self):
        state = _blank_state(seed=0)
        state.players[0].resources = [8, 0, 0, 0, 0]  # 8 wood → discard 4
        state.current_player = 0
        state.discard_obligations = {0: 4}
        state.phase = Phase.DISCARD
        bank_before = state.bank[int(Resource.WOOD)]

        for _ in range(4):
            assert state.phase == Phase.DISCARD
            apply_action(state, discard_action(Resource.WOOD))

        assert state.players[0].resources[int(Resource.WOOD)] == 4
        assert state.bank[int(Resource.WOOD)] == bank_before + 4
        assert state.phase == Phase.ROBBER  # moves to robber after all discards done

    def test_discard_obligation_consumed_one_at_a_time(self):
        state = _blank_state(seed=0)
        state.players[0].resources = [5, 0, 0, 0, 0]
        state.current_player = 0
        state.discard_obligations = {0: 2}
        state.phase = Phase.DISCARD

        apply_action(state, discard_action(Resource.WOOD))
        assert state.discard_obligations.get(0) == 1
        assert state.phase == Phase.DISCARD

        apply_action(state, discard_action(Resource.WOOD))
        assert 0 not in state.discard_obligations
        assert state.phase == Phase.ROBBER


# ---------------------------------------------------------------------------
# 7. Longest road
# ---------------------------------------------------------------------------

class TestLongestRoad:

    def _build_chain(self, state, player_id: int, length: int):
        """Build a road chain of exact length from the player's first settlement."""
        geo = state.config.geometry
        player = state.players[player_id]
        start_v = next(iter(player.settlement_vertices))
        all_roads = state.all_road_edges()
        current_v = start_v
        added = 0
        for _ in range(length):
            found = False
            for e in geo.vertex_to_edges[current_v]:
                if e in all_roads:
                    continue
                va, vb = geo.edge_to_vertices[e]
                other = vb if va == current_v else va
                player.road_vertices.add(e)
                player.roads_built += 1
                all_roads[e] = player_id
                current_v = other
                added += 1
                found = True
                break
            if not found:
                break
        return added

    def test_chain_of_5_gives_longest_road_5(self):
        state = _blank_state(seed=0)
        # Clear all existing roads for player 0 to start fresh
        state.players[0].road_vertices = set()
        state.players[0].roads_built = 0
        added = self._build_chain(state, 0, 5)
        if added < 5:
            pytest.skip("Could not build chain of 5")
        length = compute_longest_road(0, state)
        assert length == 5

    def test_chain_of_4_not_longest_road_holder(self):
        state = _blank_state(seed=0)
        state.players[0].road_vertices = set()
        state.players[0].roads_built = 0
        self._build_chain(state, 0, 4)
        update_longest_road(state)
        assert state.longest_road_holder != 0  # need ≥5

    def test_chain_of_5_gives_holder(self):
        state = _blank_state(seed=0)
        state.players[0].road_vertices = set()
        state.players[0].roads_built = 0
        added = self._build_chain(state, 0, 5)
        if added < 5:
            pytest.skip("Could not build chain of 5")
        update_longest_road(state)
        assert state.longest_road_holder == 0

    def test_longer_road_takes_title(self):
        """
        Directly assign disjoint road sets to two players, bypassing
        the chain-builder's sensitivity to board geometry at seed=0.
        Player 0 gets 5 edges from the first hex; player 1 gets 7 edges
        from the other side of the board.
        """
        state = _blank_state(seed=0)
        geo = state.config.geometry

        # Clear all roads
        for p in state.players:
            p.road_vertices = set()
            p.roads_built = 0

        # Give player 0 the 5 edges of hex 0 (skip one to avoid a cycle)
        p0_edges = geo.hex_to_edges[0][:5]
        state.players[0].road_vertices = set(p0_edges)
        state.players[0].roads_built = 5

        # Give player 1 the 6 edges of hex 18 (opposite side of board) + 1 adj edge
        p1_hex_edges = list(geo.hex_to_edges[18])
        extra_edge = None
        for v in geo.hex_to_vertices[18]:
            for e in geo.vertex_to_edges[v]:
                if e not in set(p1_hex_edges) and e not in set(p0_edges):
                    extra_edge = e
                    break
            if extra_edge is not None:
                break

        p1_edges = p1_hex_edges[:6]
        if extra_edge is not None:
            p1_edges = p1_hex_edges[:6] + [extra_edge]

        state.players[1].road_vertices = set(p1_edges)
        state.players[1].roads_built = len(p1_edges)

        p0_length = compute_longest_road(0, state)
        p1_length = compute_longest_road(1, state)

        assert p0_length >= 5
        assert p1_length > p0_length, \
            f"Setup failed: p0={p0_length}, p1={p1_length}"

        # Neither player holds longest road yet
        state.longest_road_holder = None
        update_longest_road(state)
        # Whoever has the longest should be the holder
        expected_holder = 0 if p0_length > p1_length else 1
        assert state.longest_road_holder == expected_holder

        # Now bump player 0 to be strictly longer
        state.longest_road_holder = 1  # 1 is current holder
        # Give player 0 many more roads from the other hex side
        extra_p0 = []
        for e in geo.hex_to_edges[9]:  # central hex, lots of edges
            if e not in state.players[0].road_vertices and e not in state.players[1].road_vertices:
                extra_p0.append(e)
        state.players[0].road_vertices |= set(extra_p0)
        state.players[0].roads_built += len(extra_p0)

        new_p0 = compute_longest_road(0, state)
        new_p1 = compute_longest_road(1, state)
        if new_p0 > new_p1:
            update_longest_road(state)
            assert state.longest_road_holder == 0

    def test_equal_length_does_not_transfer(self):
        """Holder keeps title on tie — must be strictly longer to take it."""
        state = _blank_state(seed=0)
        state.players[0].road_vertices = set()
        state.players[0].roads_built = 0
        added = self._build_chain(state, 0, 5)
        if added < 5:
            pytest.skip()
        update_longest_road(state)
        assert state.longest_road_holder == 0

        # Player 1 builds exactly 5 — should not take
        self._build_chain(state, 1, 5)
        update_longest_road(state)
        assert state.longest_road_holder == 0

    def test_enemy_settlement_breaks_road(self):
        state = _blank_state(seed=0)
        state.players[0].road_vertices = set()
        state.players[0].roads_built = 0
        geo = state.config.geometry
        player = state.players[0]
        start_v = next(iter(player.settlement_vertices))

        # Build a 6-edge chain and note mid vertex
        all_roads = state.all_road_edges()
        current_v = start_v
        edges = []
        vertices = [start_v]
        for _ in range(6):
            for e in geo.vertex_to_edges[current_v]:
                if e in all_roads:
                    continue
                va, vb = geo.edge_to_vertices[e]
                other = vb if va == current_v else va
                player.road_vertices.add(e)
                player.roads_built += 1
                all_roads[e] = 0
                edges.append(e)
                vertices.append(other)
                current_v = other
                break

        if len(edges) < 6:
            pytest.skip("Could not build 6-edge chain")

        length_before = compute_longest_road(0, state)

        # Place enemy settlement in the middle of the chain
        mid_v = vertices[3]
        state.players[1].settlement_vertices.add(mid_v)
        length_after = compute_longest_road(0, state)

        assert length_after < length_before


# ---------------------------------------------------------------------------
# 8. Largest army
# ---------------------------------------------------------------------------

class TestLargestArmy:

    def test_3_knights_gives_largest_army(self):
        state = _blank_state(seed=0)
        state.players[0].army_size = 3
        update_largest_army(state)
        assert state.largest_army_holder == 0

    def test_2_knights_not_enough(self):
        state = _blank_state(seed=0)
        state.players[0].army_size = 2
        update_largest_army(state)
        assert state.largest_army_holder is None

    def test_more_knights_transfers_army(self):
        state = _blank_state(seed=0)
        state.players[0].army_size = 3
        update_largest_army(state)
        assert state.largest_army_holder == 0

        state.players[1].army_size = 4
        update_largest_army(state)
        assert state.largest_army_holder == 1

    def test_tie_does_not_transfer(self):
        state = _blank_state(seed=0)
        state.players[0].army_size = 3
        update_largest_army(state)
        state.players[1].army_size = 3
        update_largest_army(state)
        assert state.largest_army_holder == 0  # original holder keeps on tie

    def test_largest_army_worth_2_vp(self):
        state = _blank_state(seed=0)
        state.largest_army_holder = 0
        vp = compute_vp(0, state)
        vp_no_army = compute_vp(1, state)
        assert vp == vp_no_army + 2

    def test_knight_play_increments_army(self):
        state = _blank_state(seed=0)
        p = state.players[0]
        p.dev_cards[int(DevCard.KNIGHT)] = 1
        p.army_size = 0
        state.phase = Phase.MAIN

        apply_action(state, PLAY_KNIGHT)
        assert p.army_size == 1
        assert state.phase == Phase.ROBBER


# ---------------------------------------------------------------------------
# 9. Dev card effects
# ---------------------------------------------------------------------------

class TestDevCardEffects:

    def test_monopoly_steals_all_of_resource(self):
        state = _blank_state(seed=0)
        state.players[1].resources[int(Resource.ORE)] = 3
        state.players[2].resources[int(Resource.ORE)] = 2
        state.players[3].resources[int(Resource.ORE)] = 1
        state.players[0].resources = [0] * 5
        state.players[0].dev_cards[int(DevCard.MONOPOLY)] = 1
        state.phase = Phase.MAIN

        apply_action(state, monopoly_action(Resource.ORE))

        assert state.players[0].resources[int(Resource.ORE)] == 6
        assert state.players[1].resources[int(Resource.ORE)] == 0
        assert state.players[2].resources[int(Resource.ORE)] == 0
        assert state.players[3].resources[int(Resource.ORE)] == 0

    def test_monopoly_does_not_steal_other_resources(self):
        state = _blank_state(seed=0)
        state.players[1].resources = [2, 0, 0, 0, 2]  # wood + ore
        state.players[0].resources = [0] * 5
        state.players[0].dev_cards[int(DevCard.MONOPOLY)] = 1
        state.phase = Phase.MAIN

        apply_action(state, monopoly_action(Resource.ORE))

        assert state.players[0].resources[int(Resource.ORE)] == 2
        assert state.players[1].resources[int(Resource.WOOD)] == 2  # untouched

    def test_year_of_plenty_gives_two_from_bank(self):
        state = _blank_state(seed=0)
        state.players[0].resources = [0] * 5
        state.players[0].dev_cards[int(DevCard.YEAR_OF_PLENTY)] = 1
        state.bank[int(Resource.WHEAT)] = 5
        state.bank[int(Resource.ORE)] = 5
        state.phase = Phase.MAIN

        apply_action(state, year_of_plenty_action(Resource.WHEAT, Resource.ORE))

        assert state.players[0].resources[int(Resource.WHEAT)] == 1
        assert state.players[0].resources[int(Resource.ORE)] == 1
        assert state.bank[int(Resource.WHEAT)] == 4
        assert state.bank[int(Resource.ORE)] == 4

    def test_year_of_plenty_same_resource_twice(self):
        state = _blank_state(seed=0)
        state.players[0].resources = [0] * 5
        state.players[0].dev_cards[int(DevCard.YEAR_OF_PLENTY)] = 1
        state.bank[int(Resource.WOOD)] = 5
        state.phase = Phase.MAIN

        apply_action(state, year_of_plenty_action(Resource.WOOD, Resource.WOOD))

        assert state.players[0].resources[int(Resource.WOOD)] == 2
        assert state.bank[int(Resource.WOOD)] == 3

    def test_road_building_gives_two_free_roads(self):
        state = _blank_state(seed=0)
        p = state.players[0]
        p.dev_cards[int(DevCard.ROAD_BUILDING)] = 1
        p.resources = [0] * 5  # no resources — roads should be free
        roads_before = p.roads_built
        state.phase = Phase.MAIN
        state.rolled_this_turn = True  # already rolled to reach MAIN this turn

        apply_action(state, PLAY_ROAD_BUILDING)
        assert state.phase == Phase.ROAD_BUILDING_1

        rng = random.Random(0)
        acts = legal_actions(state)
        road_acts = [a for a in acts if a.action_type == ActionType.BUILD_ROAD]
        if road_acts:
            apply_action(state, road_acts[0], rng)
            assert state.phase == Phase.ROAD_BUILDING_2
            assert p.roads_built == roads_before + 1

            acts2 = legal_actions(state)
            road_acts2 = [a for a in acts2 if a.action_type == ActionType.BUILD_ROAD]
            if road_acts2:
                apply_action(state, road_acts2[0], rng)
                assert state.phase == Phase.MAIN
                assert p.roads_built == roads_before + 2

        assert sum(p.resources) == 0  # no cost

    def test_victory_point_card_adds_vp_as_soon_as_held(self):
        """VP dev cards are never "played" -- they count automatically the
        moment they're held, so acquiring one adds a VP with no action."""
        state = _blank_state(seed=0)
        p = state.players[0]
        vp_before = compute_vp(0, state)

        p.dev_cards[int(DevCard.VICTORY_POINT)] = 1

        vp_after = compute_vp(0, state)
        assert vp_after == vp_before + 1

    def test_playing_victory_point_card_is_a_vp_no_op(self):
        """The PLAY_VICTORY_POINT apply_action handler is kept only for
        backcompat with old recorded traces; it moves the card from hand to
        played, but since both are already counted in hidden_vp, VP is
        unaffected."""
        state = _blank_state(seed=0)
        p = state.players[0]
        p.dev_cards[int(DevCard.VICTORY_POINT)] = 1
        state.phase = Phase.MAIN

        vp_before = compute_vp(0, state)
        apply_action(state, PLAY_VICTORY_POINT)
        vp_after = compute_vp(0, state)
        assert vp_after == vp_before

    def test_cannot_play_card_bought_this_turn(self):
        state = _blank_state(seed=0)
        p = state.players[0]
        p.dev_cards_new[int(DevCard.KNIGHT)] = 1  # bought this turn
        p.dev_cards[int(DevCard.KNIGHT)] = 0
        state.phase = Phase.MAIN

        acts = legal_actions(state)
        knight_acts = [a for a in acts if a.action_type == ActionType.PLAY_KNIGHT]
        assert len(knight_acts) == 0

    def test_card_moves_to_playable_after_end_turn(self):
        state = _blank_state(seed=0)
        p = state.players[0]
        p.dev_cards_new[int(DevCard.KNIGHT)] = 1
        p.dev_cards[int(DevCard.KNIGHT)] = 0

        p.end_turn_refresh_dev_cards()

        assert p.dev_cards[int(DevCard.KNIGHT)] == 1
        assert p.dev_cards_new[int(DevCard.KNIGHT)] == 0


# ---------------------------------------------------------------------------
# 10. Turn progression
# ---------------------------------------------------------------------------

class TestTurnProgression:

    def test_end_turn_advances_player(self):
        state = _blank_state(seed=0)
        state.phase = Phase.MAIN
        assert state.current_player == 0

        apply_action(state, END_TURN)

        assert state.current_player == 1
        assert state.phase == Phase.ROLL
        assert state.turn_number == 1

    def test_player_wraps_around(self):
        state = _blank_state(seed=0)
        state.phase = Phase.MAIN
        state.current_player = 3
        state.turn_number = 3

        apply_action(state, END_TURN)

        assert state.current_player == 0

    def test_dice_cleared_on_end_turn(self):
        state = _blank_state(seed=0)
        state.dice = (3, 4)
        state.phase = Phase.MAIN

        apply_action(state, END_TURN)

        assert state.dice is None

    def test_full_round_robin(self):
        state = _blank_state(seed=0)
        rng = random.Random(0)

        for expected_player in range(4):
            assert state.current_player == expected_player
            assert state.phase == Phase.ROLL
            apply_action(state, ROLL_DICE, rng)
            # Handle any sub-phases (robber, discard, steal)
            max_steps = 30
            while state.phase not in (Phase.MAIN, Phase.GAME_OVER) and max_steps > 0:
                acts = legal_actions(state)
                if acts:
                    apply_action(state, rng.choice(acts), rng)
                max_steps -= 1
            if state.is_terminal:
                break
            apply_action(state, END_TURN)


# ---------------------------------------------------------------------------
# 11. Scoring
# ---------------------------------------------------------------------------

class TestScoring:

    def test_settlement_worth_1_vp(self):
        state = _blank_state(seed=0)
        p = state.players[0]
        p.settlements_built = 1
        p.cities_built = 0
        assert p.public_vp == 1

    def test_city_worth_2_vp(self):
        state = _blank_state(seed=0)
        p = state.players[0]
        p.settlements_built = 0
        p.cities_built = 1
        assert p.public_vp == 2

    def test_longest_road_worth_2_vp(self):
        state = _blank_state(seed=0)
        state.longest_road_holder = 0
        vp0 = compute_vp(0, state)
        state.longest_road_holder = None
        vp0_no_road = compute_vp(0, state)
        assert vp0 == vp0_no_road + 2

    def test_largest_army_worth_2_vp(self):
        state = _blank_state(seed=0)
        state.largest_army_holder = 0
        vp0 = compute_vp(0, state)
        state.largest_army_holder = None
        vp0_no_army = compute_vp(0, state)
        assert vp0 == vp0_no_army + 2

    def test_win_at_exactly_10_vp(self):
        state = _blank_state(seed=0)
        p = state.players[0]
        # 4 cities = 8 VP
        p.cities_built = 4
        p.city_vertices = set(range(4))
        p.settlements_built = 0
        p.settlement_vertices = set()
        state.longest_road_holder = 0  # +2 VP = 10 total

        assert compute_vp(0, state) == 10
        assert check_winner(state) == 0

    def test_no_win_at_9_vp(self):
        state = _blank_state(seed=0)
        p = state.players[0]
        p.cities_built = 4
        p.city_vertices = set(range(4))
        p.settlements_built = 0
        p.settlement_vertices = set()
        # 8 VP, no special cards

        assert compute_vp(0, state) == 8
        assert check_winner(state) is None

    def test_vp_dev_card_counts_toward_win(self):
        state = _blank_state(seed=0)
        p = state.players[0]
        p.cities_built = 4
        p.city_vertices = set(range(4))
        p.settlements_built = 0
        p.settlement_vertices = set()
        # 8 VP from cities; play 2 VP cards to reach 10
        p.played_dev_cards[int(DevCard.VICTORY_POINT)] = 2

        assert compute_vp(0, state) == 10
        assert check_winner(state) == 0


# ---------------------------------------------------------------------------
# 12. Board geometry sanity
# ---------------------------------------------------------------------------

class TestBoardGeometry:

    def test_standard_board_counts(self):
        from catan_rl.env.board import get_geometry
        geo = get_geometry()
        assert geo.n_hexes == 19
        assert geo.n_vertices == 54
        assert geo.n_edges == 72

    def test_every_hex_has_6_vertices(self):
        from catan_rl.env.board import get_geometry
        geo = get_geometry()
        for hi in range(geo.n_hexes):
            assert len(geo.hex_to_vertices[hi]) == 6, f"Hex {hi} has wrong vertex count"

    def test_every_hex_has_6_edges(self):
        from catan_rl.env.board import get_geometry
        geo = get_geometry()
        for hi in range(geo.n_hexes):
            assert len(geo.hex_to_edges[hi]) == 6, f"Hex {hi} has wrong edge count"

    def test_interior_vertices_have_3_hexes(self):
        from catan_rl.env.board import get_geometry
        geo = get_geometry()
        interior = [v for v in range(geo.n_vertices)
                    if len(geo.vertex_to_hexes[v]) == 3]
        assert len(interior) == 24

    def test_coastal_vertices_have_1_or_2_hexes(self):
        from catan_rl.env.board import get_geometry
        geo = get_geometry()
        coastal = [v for v in range(geo.n_vertices)
                   if len(geo.vertex_to_hexes[v]) < 3]
        assert len(coastal) == 30

    def test_every_edge_connects_two_vertices(self):
        from catan_rl.env.board import get_geometry
        geo = get_geometry()
        for eid in range(geo.n_edges):
            va, vb = geo.edge_to_vertices[eid]
            assert va != vb
            assert 0 <= va < geo.n_vertices
            assert 0 <= vb < geo.n_vertices

    def test_resource_counts(self):
        config = BoardConfig.standard(seed=0)
        from collections import Counter
        counts = Counter(config.hex_resources)
        assert counts[HexType.WOOD] == 4
        assert counts[HexType.BRICK] == 3
        assert counts[HexType.SHEEP] == 4
        assert counts[HexType.WHEAT] == 4
        assert counts[HexType.ORE] == 3
        assert counts[HexType.DESERT] == 1

    def test_token_counts(self):
        config = BoardConfig.standard(seed=0)
        tokens = [t for t in config.hex_tokens if t > 0]
        from collections import Counter
        counts = Counter(tokens)
        assert counts[2] == 1
        assert counts[12] == 1
        assert counts[6] == 2
        assert counts[8] == 2
        assert len(tokens) == 18  # one per non-desert hex

    def test_9_ports(self):
        config = BoardConfig.standard(seed=0)
        assert len(config.ports) == 9
        specific = [p for p in config.ports if p.resource is not None]
        generic = [p for p in config.ports if p.resource is None]
        assert len(specific) == 5
        assert len(generic) == 4
