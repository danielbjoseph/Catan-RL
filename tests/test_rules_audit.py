"""
Official-rulebook regression tests for catan_rl.env.rules.

These tests encode specific rules from the official Catan rulebook that are
easy to get subtly wrong in an engine implementation (e.g. "first come, first
served" bank depletion vs. the official "nobody gets it if the bank can't
cover everyone" rule). This file grows across the rules-audit task series.
"""

import random

from catan_rl.env.action_mask import legal_action_mask
from catan_rl.env.actions import (
    ActionType, DevCard, END_TURN, PLAY_KNIGHT, ROLL_DICE,
    move_robber_action, steal_action,
)
from catan_rl.env.board import BoardConfig, HexType
from catan_rl.env.game_state import GameState, Phase
from catan_rl.env.rules import _produce_resources, apply_action
from catan_rl.env.rules_profile import SIMPLIFIED_V1
from catan_rl.env.scoring import check_winner
from catan_rl.env.validators import legal_actions
from catan_rl.bots.random_bot import pick_action


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


class TestVictoryPointDevCardsAutoCount:
    """Official rule: VP dev cards are never "played" — they count toward
    the win condition automatically as soon as they're held, while staying
    excluded from public VP (visible building-based score)."""

    def test_newly_bought_vp_card_detected_as_winner_on_next_win_check(self):
        """A player at 9 public VP who receives a VP card (still in
        dev_cards_new, i.e. bought this turn and not yet "playable") should
        be declared the winner the next time a win check runs, without ever
        playing the card."""
        state, config = _make_state(seed=0)
        state.phase = Phase.MAIN
        p = state.players[0]
        p.settlements_built = 1
        p.settlement_vertices = {0}
        p.cities_built = 4
        p.city_vertices = {1, 2, 3, 4}
        assert p.public_vp == 9

        assert check_winner(state) is None

        p.receive_dev_card(DevCard.VICTORY_POINT)  # lands in dev_cards_new
        assert p.dev_cards_new[int(DevCard.VICTORY_POINT)] == 1

        apply_action(state, END_TURN)

        assert state.winner == 0
        assert state.phase == Phase.GAME_OVER

    def test_play_victory_point_slot_never_legal_over_random_game(self):
        """Catalog slot 253 (PLAY_VICTORY_POINT) must never appear as a
        legal action across a full random-legal standard-profile game."""
        rng = random.Random(0)
        config = BoardConfig.standard(seed=0)
        state = GameState.new_game(config, n_players=4, seed=0)

        for _ in range(200):
            mask = legal_action_mask(state)
            assert not mask[253], "PLAY_VICTORY_POINT (slot 253) must be permanently masked"
            if state.phase == Phase.GAME_OVER:
                break
            action = pick_action(state, rng)
            apply_action(state, action, rng)

    def test_public_vp_excludes_vp_cards(self):
        """public_vp must reflect only settlements/cities; VP dev cards
        (held, newly bought, or historically played) live in hidden_vp."""
        state, config = _make_state(seed=0)
        p = state.players[0]
        p.settlements_built = 2
        p.cities_built = 1
        p.dev_cards[int(DevCard.VICTORY_POINT)] = 3
        p.dev_cards_new[int(DevCard.VICTORY_POINT)] = 2
        p.played_dev_cards[int(DevCard.VICTORY_POINT)] = 1

        assert p.public_vp == 4
        assert p.hidden_vp == 6
        assert p.total_vp == 10


class TestDevCardBeforeRoll:
    """Official rule: one dev card may be played at any time during your
    turn, including before the roll. Knight-before-roll is the strategically
    important case (unblocks your own hexes before production happens)."""

    def _no_target_hex(self, state) -> int:
        """A hex the robber isn't already on (safe to move to; _make_state
        clears all settlements so no hex has adjacent occupied vertices)."""
        return 0 if state.robber_hex != 0 else 1

    def test_knight_legal_in_roll_phase(self):
        """A player holding a knight (standard profile) sees PLAY_KNIGHT
        legal while still in the ROLL phase, before rolling."""
        state, config = _make_state(seed=0)
        p0 = state.players[0]
        p0.dev_cards[int(DevCard.KNIGHT)] = 1

        assert state.phase == Phase.ROLL
        acts = legal_actions(state)
        assert any(a.action_type == ActionType.PLAY_KNIGHT for a in acts)

        mask = legal_action_mask(state)
        assert mask[231], "PLAY_KNIGHT (catalog slot 231) must be legal pre-roll"

    def test_knight_preroll_no_steal_target_returns_to_roll(self):
        """Playing knight pre-roll with no adjacent opponents: ROLL ->
        (play knight) -> ROBBER -> back to ROLL, and ROLL_DICE is still
        required/legal afterward."""
        state, config = _make_state(seed=0)
        p0 = state.players[0]
        p0.dev_cards[int(DevCard.KNIGHT)] = 1

        apply_action(state, PLAY_KNIGHT)
        assert state.phase == Phase.ROBBER
        assert p0.has_played_dev_card
        assert p0.army_size == 1

        apply_action(state, move_robber_action(self._no_target_hex(state)))
        assert state.phase == Phase.ROLL

        acts = legal_actions(state)
        assert any(a.action_type == ActionType.ROLL_DICE for a in acts)
        assert ROLL_DICE in acts

        # Must still be able to roll to finish the turn.
        apply_action(state, ROLL_DICE, random.Random(0))
        assert state.rolled_this_turn is True

    def test_knight_preroll_with_steal_target_returns_to_roll(self):
        """Playing knight pre-roll with an adjacent opponent: ROLL ->
        ROBBER -> STEAL -> back to ROLL (not MAIN)."""
        state, config = _make_state(seed=1)
        geo = config.geometry
        p0 = state.players[0]
        p1 = state.players[1]
        p0.dev_cards[int(DevCard.KNIGHT)] = 1
        p1.resources = [1, 0, 0, 0, 0]

        hex_id = self._no_target_hex(state)
        v = geo.hex_to_vertices[hex_id][0]
        p1.settlement_vertices.add(v)

        apply_action(state, PLAY_KNIGHT)
        assert state.phase == Phase.ROBBER

        apply_action(state, move_robber_action(hex_id))
        assert state.phase == Phase.STEAL

        apply_action(state, steal_action(1), random.Random(0))
        assert state.phase == Phase.ROLL

    def test_knight_postroll_returns_to_main(self):
        """Playing knight after rolling still returns to MAIN, as before."""
        state, config = _make_state(seed=0)
        p0 = state.players[0]
        p0.dev_cards[int(DevCard.KNIGHT)] = 1
        state.rolled_this_turn = True
        state.phase = Phase.MAIN

        apply_action(state, PLAY_KNIGHT)
        assert state.phase == Phase.ROBBER

        apply_action(state, move_robber_action(self._no_target_hex(state)))
        assert state.phase == Phase.MAIN

    def test_card_bought_this_turn_not_playable_preroll(self):
        """A knight bought this turn (still in dev_cards_new) is not
        playable pre-roll, same gating as MAIN."""
        state, config = _make_state(seed=0)
        p0 = state.players[0]
        p0.dev_cards_new[int(DevCard.KNIGHT)] = 1

        acts = legal_actions(state)
        assert not any(a.action_type == ActionType.PLAY_KNIGHT for a in acts)
        assert not legal_action_mask(state)[231]

    def test_preroll_dev_card_play_consumes_one_per_turn_allowance(self):
        """Playing a dev card pre-roll uses up the one-per-turn allowance:
        nothing playable in MAIN afterward that same turn."""
        state, config = _make_state(seed=0)
        p0 = state.players[0]
        p0.dev_cards[int(DevCard.KNIGHT)] = 1
        p0.dev_cards[int(DevCard.MONOPOLY)] = 1

        apply_action(state, PLAY_KNIGHT)
        assert p0.has_played_dev_card
        apply_action(state, move_robber_action(self._no_target_hex(state)))
        assert state.phase == Phase.ROLL

        # Simulate finishing the roll (already exercised end-to-end above);
        # here we just need to land in MAIN with the allowance already spent.
        state.rolled_this_turn = True
        state.phase = Phase.MAIN

        acts = legal_actions(state)
        dev_play_acts = [
            a for a in acts
            if a.action_type in (
                ActionType.PLAY_KNIGHT, ActionType.PLAY_ROAD_BUILDING,
                ActionType.PLAY_YEAR_OF_PLENTY, ActionType.PLAY_MONOPOLY,
            )
        ]
        assert dev_play_acts == []

    def test_simplified_v1_roll_phase_only_roll_dice(self):
        """simplified_v1 has dev cards disabled entirely: ROLL phase offers
        only ROLL_DICE, even if a card is (artificially) present."""
        config = BoardConfig.standard(seed=0)
        state = GameState.new_game(config, n_players=4, seed=0, profile=SIMPLIFIED_V1)
        state.phase = Phase.ROLL
        state.current_player = 0
        state._setup_forward_idx = 4
        state._setup_backward_idx = -1
        state.players[0].dev_cards[int(DevCard.KNIGHT)] = 1

        acts = legal_actions(state)
        assert acts == [ROLL_DICE]
