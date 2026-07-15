"""Tests for the evaluation harness."""

import torch

from catan_rl.bots import random_bot
from catan_rl.env.observation import obs_dim_for_mode
from catan_rl.env.rules_profile import RulesProfile
from catan_rl.rl.evaluate import evaluate_vs_bots, evaluate_vs_checkpoint
from catan_rl.rl.checkpointing import save_checkpoint
from catan_rl.rl.models import ActorCritic

FAST = RulesProfile(name="fast", dev_cards_enabled=False, win_vp=8)


def _policy(seed=0, obs_mode="self_play"):
    torch.manual_seed(seed)
    return ActorCritic(obs_dim=obs_dim_for_mode(obs_mode), hidden_sizes=(32, 32))


def test_evaluate_vs_bots_basic():
    result = evaluate_vs_bots(
        _policy(), random_bot.pick_action, n_games=4,
        rules_profile=FAST, seed=5, max_turns=400,
    )
    assert 0.0 <= result["win_rate"] <= 1.0
    assert result["mean_turns"] > 0
    assert result["mean_vp"] > 0
    assert result["games"] == 4


def test_evaluate_vs_bots_seat_rotation():
    """With 4 games the policy must have occupied each seat once."""
    result = evaluate_vs_bots(
        _policy(), random_bot.pick_action, n_games=4,
        rules_profile=FAST, seed=1, max_turns=400,
    )
    assert sorted(result["seats_played"]) == [0, 1, 2, 3]


def test_evaluate_vs_checkpoint(tmp_path):
    old = _policy(seed=1)
    opt = torch.optim.Adam(old.parameters())
    ckpt = save_checkpoint(tmp_path, old, opt, 0, {}, {})

    current = _policy(seed=2)
    result = evaluate_vs_checkpoint(
        current, ckpt, n_games=2, rules_profile=FAST, seed=9, max_turns=400,
    )
    assert 0.0 <= result["win_rate"] <= 1.0
    assert result["games"] == 2


def test_evaluate_vs_bots_realistic_mode():
    """Exercises the shared-tracker path: a policy expecting realistic-mode
    observations must be handed a table-view BeliefTracker per game."""
    result = evaluate_vs_bots(
        _policy(seed=6, obs_mode="realistic"), random_bot.pick_action, n_games=2,
        rules_profile=FAST, seed=3, max_turns=400, obs_mode="realistic",
    )
    assert 0.0 <= result["win_rate"] <= 1.0
    assert result["games"] == 2


def test_evaluate_vs_checkpoint_mixed_modes(tmp_path):
    """The old checkpoint's own stored obs_mode is used for its seats, while
    the current policy uses whatever obs_mode is passed in."""
    old = _policy(seed=1, obs_mode="self_play")
    opt = torch.optim.Adam(old.parameters())
    ckpt = save_checkpoint(tmp_path, old, opt, 0, {}, {}, obs_mode="self_play")

    current = _policy(seed=2, obs_mode="realistic")
    result = evaluate_vs_checkpoint(
        current, ckpt, n_games=2, rules_profile=FAST, seed=9, max_turns=400,
        obs_mode="realistic",
    )
    assert 0.0 <= result["win_rate"] <= 1.0
    assert result["games"] == 2
