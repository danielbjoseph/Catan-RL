"""
Official-rulebook regression tests for catan_rl.env.rules.

These tests encode specific rules from the official Catan rulebook that are
easy to get subtly wrong in an engine implementation (e.g. "first come, first
served" bank depletion vs. the official "nobody gets it if the bank can't
cover everyone" rule). This file grows across the rules-audit task series.
"""

from catan_rl.env.board import BoardConfig, HexType
from catan_rl.env.game_state import GameState, Phase
from catan_rl.env.rules import _produce_resources


def _clean_vertex(geo, config, v, hex_id, token):
    """True if `v` isn't also adjacent to a *different* hex sharing `token`.

    Standard board generation doesn't forbid two same-numbered hexes from
    touching the same vertex, so without this check a test that places a
    settlement on `v` could accidentally draw production from an unrelated
    hex and contaminate the resource counts being asserted on.
    """
    return all(
        hx == hex_id or config.hex_tokens[hx] != token
        for hx in geo.vertex_to_hexes[v]
    )


def _find_clean_production_site(config: BoardConfig, min_vertices: int = 2):
    """
    Find a non-desert hex whose token has at least `min_vertices` vertices
    that are "clean" (see `_clean_vertex`), so a test can place settlements
    there and know exactly what resource/count is owed. Returns
    (hex_id, token, resource, clean_vertices).
    """
    geo = config.geometry
    for hi, hex_type in enumerate(config.hex_resources):
        if hex_type == HexType.DESERT:
            continue
        token = config.hex_tokens[hi]
        if token == 0:
            continue
        clean = [v for v in geo.hex_to_vertices[hi] if _clean_vertex(geo, config, v, hi, token)]
        if len(clean) >= min_vertices:
            return hi, token, hex_type.to_resource(), clean
    raise AssertionError("No hex found with enough clean (non-token-colliding) vertices")


def _make_state(seed: int = 0):
    config = BoardConfig.standard(seed=seed)
    state = GameState.new_game(config, n_players=4, seed=seed)
    state.phase = Phase.ROLL
    state.current_player = 0
    # Setup phase bookkeeping isn't relevant to production tests.
    state._setup_forward_idx = 4
    state._setup_backward_idx = -1
    # Start every player with a clean slate; tests place settlements manually.
    for p in state.players:
        p.settlement_vertices = set()
        p.city_vertices = set()
        p.resources = [0] * 5
    return state, config


class TestBankShortageOnProduction:
    """Official rule: if the bank cannot fully supply *all* players owed a
    resource on a roll, *no* player receives that resource type — unless
    exactly one player is owed it, in which case they take what's left."""

    def test_two_players_owed_more_than_bank_has_both_get_zero(self):
        """(a) Two players owed wood, bank has only 1 -> both get 0, bank unchanged."""
        state, config = _make_state(seed=0)
        hex_id, token, resource, vertices = _find_clean_production_site(config, min_vertices=2)

        v0, v1 = vertices[0], vertices[1]
        state.players[0].settlement_vertices.add(v0)
        state.players[1].settlement_vertices.add(v1)

        state.bank = [19, 19, 19, 19, 19]
        state.bank[int(resource)] = 1  # not enough for two players each owed 1

        bank_before = list(state.bank)
        _produce_resources(state, token)

        assert state.players[0].resources[int(resource)] == 0
        assert state.players[1].resources[int(resource)] == 0
        assert state.bank == bank_before

    def test_single_demander_takes_remaining_supply(self):
        """(b) One player owed 3 (city + settlement), bank has 2 -> gets 2, bank 0."""
        state, config = _make_state(seed=0)
        hex_id, token, resource, vertices = _find_clean_production_site(config, min_vertices=2)

        v0, v1 = vertices[0], vertices[1]
        # Player 0 has a city on v0 (2) and a settlement on v1 (1) => owed 3.
        state.players[0].city_vertices.add(v0)
        state.players[0].settlement_vertices.add(v1)

        state.bank = [19, 19, 19, 19, 19]
        state.bank[int(resource)] = 2  # less than the 3 owed

        _produce_resources(state, token)

        assert state.players[0].resources[int(resource)] == 2
        assert state.bank[int(resource)] == 0

    def test_sufficient_supply_unchanged_behavior(self):
        """(c) Bank has plenty -> every demander gets exactly what they're owed."""
        state, config = _make_state(seed=0)
        hex_id, token, resource, vertices = _find_clean_production_site(config, min_vertices=2)

        v0, v1 = vertices[0], vertices[1]
        state.players[0].settlement_vertices.add(v0)
        state.players[1].city_vertices.add(v1)

        state.bank = [19, 19, 19, 19, 19]
        bank_before = list(state.bank)

        _produce_resources(state, token)

        assert state.players[0].resources[int(resource)] == 1
        assert state.players[1].resources[int(resource)] == 2
        assert state.bank[int(resource)] == bank_before[int(resource)] - 3
