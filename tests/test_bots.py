"""Tests for scripted bots: greedy and heuristic."""

import random

import pytest

from catan_rl.bots import greedy_bot, heuristic_bot, random_bot
from catan_rl.env.board import BoardConfig
from catan_rl.env.game_state import GameState
from catan_rl.env.rules import apply_action
from catan_rl.env.rules_profile import SIMPLIFIED_V1
from catan_rl.env.validators import legal_actions

MAX_PLIES = 5000


def play_match(seat_bots, seed, profile=SIMPLIFIED_V1, max_plies=MAX_PLIES):
    """Play one game with one pick_action fn per seat. Returns final state."""
    rng = random.Random(seed)
    config = BoardConfig.standard(seed=seed)
    state = GameState.new_game(config, n_players=4, seed=seed, profile=profile)
    plies = 0
    while not state.is_terminal and plies < max_plies:
        action = seat_bots[state.current_player](state, rng)
        apply_action(state, action, rng)
        plies += 1
    return state


@pytest.mark.parametrize("bot", [greedy_bot, heuristic_bot], ids=["greedy", "heuristic"])
class TestBotBasics:
    def test_completes_games(self, bot):
        for seed in range(3):
            state = play_match([bot.pick_action] * 4, seed=seed)
            assert state.is_terminal, f"game with seed {seed} did not finish"
            assert state.winner is not None

    def test_only_legal_actions(self, bot):
        rng = random.Random(11)
        config = BoardConfig.standard(seed=11)
        state = GameState.new_game(config, n_players=4, seed=11, profile=SIMPLIFIED_V1)
        for _ in range(300):
            if state.is_terminal:
                break
            action = bot.pick_action(state, rng)
            legal_idx = {a.catalog_index for a in legal_actions(state)}
            assert action.catalog_index in legal_idx, (
                f"illegal action {action} in phase {state.phase}"
            )
            apply_action(state, action, rng)

    def test_beats_random(self, bot):
        """Bot in a rotating seat vs 3 randoms should win >40% (random baseline: 25%)."""
        wins = 0
        n_games = 16
        for i in range(n_games):
            seat = i % 4
            bots = [random_bot.pick_action] * 4
            bots[seat] = bot.pick_action
            state = play_match(bots, seed=100 + i)
            if state.winner == seat:
                wins += 1
        assert wins / n_games > 0.4, f"only {wins}/{n_games} wins vs random"


def test_standard_profile_games_complete():
    """Bots must also handle dev cards under the standard profile."""
    from catan_rl.env.rules_profile import STANDARD
    for seed in (0, 1):
        state = play_match([greedy_bot.pick_action] * 4, seed=seed, profile=STANDARD)
        assert state.is_terminal
