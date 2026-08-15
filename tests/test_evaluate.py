"""Tests for the evaluation harness."""

import sys
from pathlib import Path

import torch

from catan_rl.bots import PERSONALITIES, random_bot, resolve_bot
from catan_rl.env.observation import obs_dim_for_mode
from catan_rl.env.rules_profile import RulesProfile
from catan_rl.rl.evaluate import (
    evaluate_policy_vs_policy,
    evaluate_vs_bots,
    evaluate_vs_checkpoint,
)
from catan_rl.rl.checkpointing import load_policy, save_checkpoint
from catan_rl.rl.models import ActorCritic
from catan_rl.rl.self_play import eval_personality_names

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.evaluate_checkpoints import eval_kwargs_from_meta

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


def test_evaluate_policy_vs_policy_cross_mode():
    """Two random-init policies in different obs modes (global vs realistic)
    should complete games with sane, bounded win rates."""
    policy_a = _policy(seed=1, obs_mode="global")
    policy_b = _policy(seed=2, obs_mode="realistic")
    result = evaluate_policy_vs_policy(
        policy_a, "global", policy_b, "realistic",
        n_games=2, rules_profile=FAST, seed=7, max_turns=400,
    )
    assert set(result.keys()) == {"win_rate_a", "win_rate_b", "draws", "games"}
    assert result["win_rate_a"] + result["win_rate_b"] <= 1.0
    assert result["games"] == 2


def test_evaluate_policy_vs_policy_deterministic():
    """Same seed and inputs must yield identical results across runs. This
    also indirectly exercises seat rotation: game 0 gives A seats {0, 2}
    and game 1 gives A seats {1, 3}, matching evaluate_vs_checkpoint's
    convention, and determinism confirms the rotation logic is stable
    rather than randomized."""
    policy_a = _policy(seed=3, obs_mode="self_play")
    policy_b = _policy(seed=4, obs_mode="self_play")

    result1 = evaluate_policy_vs_policy(
        policy_a, "self_play", policy_b, "self_play",
        n_games=2, rules_profile=FAST, seed=11, max_turns=400,
    )
    result2 = evaluate_policy_vs_policy(
        policy_a, "self_play", policy_b, "self_play",
        n_games=2, rules_profile=FAST, seed=11, max_turns=400,
    )
    assert result1 == result2


def test_eval_kwargs_from_meta_dispatches_realistic_obs_mode(tmp_path):
    """A checkpoint trained in realistic mode must be evaluated with
    obs_mode='realistic' and a noise_cfg (defaulting to 0.25/0.5 when the
    checkpoint's own config doesn't specify belief_blend/belief_noise),
    otherwise evaluate_vs_bots crashes with a shape mismatch (1548 vs 1577)
    because it silently defaults to self_play."""
    policy = _policy(seed=1, obs_mode="realistic")
    opt = torch.optim.Adam(policy.parameters())
    ckpt = save_checkpoint(tmp_path, policy, opt, 0, {}, {}, obs_mode="realistic")

    loaded_policy, meta = load_policy(ckpt)
    kwargs = eval_kwargs_from_meta(meta, seed=3)
    assert kwargs["obs_mode"] == "realistic"
    assert kwargs["noise_cfg"] == {
        "belief_blend": 0.25,
        "belief_noise": 0.5,
        "seed": 3,
    }

    # Exercise the same call path the script uses: must run without crashing.
    result = evaluate_vs_bots(
        loaded_policy, random_bot.pick_action, n_games=1,
        rules_profile=FAST, seed=3, max_turns=5,
        obs_mode=kwargs["obs_mode"], noise_cfg=kwargs["noise_cfg"],
    )
    assert result["games"] == 1


def test_eval_kwargs_from_meta_uses_ckpts_own_belief_config(tmp_path):
    """When the checkpoint's stored training config carries custom
    belief_blend/belief_noise values, those must be used instead of the
    0.25/0.5 defaults."""
    policy = _policy(seed=1, obs_mode="realistic")
    opt = torch.optim.Adam(policy.parameters())
    ckpt = save_checkpoint(
        tmp_path, policy, opt, 0,
        config={"belief_blend": 0.1, "belief_noise": 0.9},
        metrics={}, obs_mode="realistic",
    )
    _, meta = load_policy(ckpt)
    kwargs = eval_kwargs_from_meta(meta, seed=5)
    assert kwargs["noise_cfg"] == {
        "belief_blend": 0.1,
        "belief_noise": 0.9,
        "seed": 5,
    }


def test_eval_kwargs_from_meta_self_play_has_no_noise_cfg(tmp_path):
    policy = _policy(seed=1, obs_mode="self_play")
    opt = torch.optim.Adam(policy.parameters())
    ckpt = save_checkpoint(tmp_path, policy, opt, 0, {}, {}, obs_mode="self_play")
    _, meta = load_policy(ckpt)
    kwargs = eval_kwargs_from_meta(meta)
    assert kwargs == {"obs_mode": "self_play", "noise_cfg": None}


def test_old_checkpoint_plays_in_trading_env(tmp_path):
    """A Package-A-shaped checkpoint (1520-obs/256-action, predating the
    trade catalog/observation extensions) must still be usable as an eval
    opponent inside the current 1548-obs/512-action trading env: obs/mask
    are exact prefixes (Task 4 invariant), so `policy_action` slicing via
    `act_prefix_sliced` must complete the game without a shape error, with
    the old policy auto-declining any trade response it can't represent."""
    old = ActorCritic(obs_dim=1520, n_actions=256, hidden_sizes=(32, 32))
    opt = torch.optim.Adam(old.parameters())
    ckpt = save_checkpoint(tmp_path, old, opt, 0, {}, {}, obs_mode="self_play")
    policy, _ = load_policy(ckpt)

    result = evaluate_vs_bots(
        policy, resolve_bot("opportunist"), n_games=1,
        rules_profile="standard_trading", max_turns=60,
    )
    assert result["games"] == 1


def test_old_checkpoint_plays_via_evaluate_vs_checkpoint(tmp_path):
    """Same backcompat contract as test_old_checkpoint_plays_in_trading_env,
    but exercised directly through evaluate_vs_checkpoint and
    evaluate_policy_vs_policy rather than incidentally via evaluate_vs_bots
    -- these share _policy_actor/policy_action, but a future change could
    special-case one of the three eval functions without the others."""
    old = ActorCritic(obs_dim=1520, n_actions=256, hidden_sizes=(32, 32))
    opt = torch.optim.Adam(old.parameters())
    ckpt = save_checkpoint(tmp_path, old, opt, 0, {}, {}, obs_mode="self_play")

    current = _policy(seed=1)
    result = evaluate_vs_checkpoint(
        current, ckpt, n_games=2, rules_profile="standard_trading", max_turns=60,
    )
    assert result["games"] == 2

    old_policy, _ = load_policy(ckpt)
    result = evaluate_policy_vs_policy(
        old_policy, "self_play", current, "self_play", n_games=2,
        rules_profile="standard_trading", max_turns=60,
    )
    assert result["games"] == 2


def test_eval_personalities_default_gating():
    """`eval_personalities: None` (the default) resolves to no personality
    opponents under a non-trading profile, and all 5 presets under a
    trading-enabled profile. An explicit list is passed through as-is."""
    non_trading = RulesProfile.get("simplified_v1")
    trading = RulesProfile.get("standard_trading")

    assert eval_personality_names({"eval_personalities": None}, non_trading) == []
    assert set(eval_personality_names({"eval_personalities": None}, trading)) == set(PERSONALITIES)

    explicit = ["opportunist", "desperado"]
    assert eval_personality_names({"eval_personalities": explicit}, non_trading) == explicit
    assert eval_personality_names({"eval_personalities": []}, trading) == []


def test_evaluate_policy_vs_policy_draws_on_truncation():
    """With max_turns=2 no game can reach a winner, so every game is a draw
    and both win rates are 0."""
    policy_a = _policy(seed=5, obs_mode="self_play")
    policy_b = _policy(seed=6, obs_mode="self_play")
    result = evaluate_policy_vs_policy(
        policy_a, "self_play", policy_b, "self_play",
        n_games=3, rules_profile=FAST, seed=13, max_turns=2,
    )
    assert result["draws"] == 3
    assert result["win_rate_a"] == 0.0
    assert result["win_rate_b"] == 0.0
    assert result["games"] == 3
