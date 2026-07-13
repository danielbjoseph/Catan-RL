"""Tests for the evaluation harness."""

import torch

from catan_rl.bots import random_bot
from catan_rl.env.rules_profile import RulesProfile
from catan_rl.rl.evaluate import evaluate_vs_bots, evaluate_vs_checkpoint
from catan_rl.rl.checkpointing import save_checkpoint
from catan_rl.rl.models import ActorCritic

FAST = RulesProfile(name="fast", dev_cards_enabled=False, win_vp=8)


def _policy(seed=0):
    torch.manual_seed(seed)
    return ActorCritic(hidden_sizes=(32, 32))


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
