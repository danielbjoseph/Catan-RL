"""Tests for checkpoint save/load round-trips."""

import json

import pytest
import torch

from catan_rl.rl.checkpointing import (
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    load_policy,
    save_checkpoint,
    widen_policy,
)
from catan_rl.rl.models import ActorCritic


def _make(seed=0):
    torch.manual_seed(seed)
    policy = ActorCritic(obs_dim=32, hidden_sizes=(16, 16))
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    return policy, opt


def test_save_load_round_trip(tmp_path):
    policy, opt = _make(seed=1)
    path = save_checkpoint(
        tmp_path, policy, opt, iteration=7,
        config={"lr": 1e-3}, metrics={"win_rate": 0.5},
    )
    assert path.exists()

    obs = torch.randn(3, 32)
    with torch.no_grad():
        logits_before, value_before = policy(obs)

    fresh, _ = _make(seed=2)  # different weights
    meta = load_checkpoint(path, fresh)
    with torch.no_grad():
        logits_after, value_after = fresh(obs)

    assert torch.allclose(logits_before, logits_after)
    assert torch.allclose(value_before, value_after)
    assert meta["iteration"] == 7


def test_metadata_sidecar(tmp_path):
    policy, opt = _make()
    path = save_checkpoint(tmp_path, policy, opt, 3, {"gamma": 0.99}, {"elo": 1000})
    sidecar = path.with_suffix(".json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["iteration"] == 3
    assert meta["config"]["gamma"] == 0.99
    assert meta["metrics"]["elo"] == 1000
    assert meta["arch"]["obs_dim"] == 32


def test_load_policy_reconstructs_arch(tmp_path):
    policy, opt = _make(seed=3)
    path = save_checkpoint(tmp_path, policy, opt, 1, {}, {})
    loaded, meta = load_policy(path)
    assert loaded.obs_dim == 32
    obs = torch.randn(2, 32)
    with torch.no_grad():
        a, _ = policy(obs)
        b, _ = loaded(obs)
    assert torch.allclose(a, b)


def test_obs_mode_metadata_round_trip(tmp_path):
    policy, opt = _make(seed=4)
    path = save_checkpoint(
        tmp_path, policy, opt, iteration=2,
        config={}, metrics={}, obs_mode="realistic",
    )
    _, meta = load_policy(path)
    assert meta["obs_mode"] == "realistic"
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["obs_mode"] == "realistic"


def test_obs_mode_defaults_to_self_play(tmp_path):
    policy, opt = _make(seed=5)
    path = save_checkpoint(tmp_path, policy, opt, 3, {}, {})
    _, meta = load_policy(path)
    assert meta["obs_mode"] == "self_play"


def test_latest_and_list(tmp_path):
    policy, opt = _make()
    p1 = save_checkpoint(tmp_path, policy, opt, 1, {}, {})
    p10 = save_checkpoint(tmp_path, policy, opt, 10, {}, {})
    p2 = save_checkpoint(tmp_path, policy, opt, 2, {}, {})
    assert list_checkpoints(tmp_path) == [p1, p2, p10]
    assert latest_checkpoint(tmp_path) == p10
    assert latest_checkpoint(tmp_path / "empty") is None


def test_widen_preserves_old_function_exactly():
    old = ActorCritic(obs_dim=1520, n_actions=256, hidden_sizes=(32, 32))
    new = widen_policy(old, 1548, 512)

    x_old = torch.randn(3, 1520)
    x_new = torch.cat([x_old, torch.zeros(3, 28)], dim=1)
    with torch.no_grad():
        lo, vo = old(x_old)
        ln, vn = new(x_new)

    assert new.obs_dim == 1548
    assert new.n_actions == 512
    assert torch.allclose(lo, ln[:, :256], atol=1e-6)
    assert torch.allclose(vo, vn, atol=1e-6)
    assert torch.all(ln[:, 256:] == -4.0)  # zero weights + bias -4


def test_widen_rejects_shrink_or_hidden_mismatch():
    old = ActorCritic(obs_dim=1520, n_actions=256, hidden_sizes=(32, 32))

    with pytest.raises(ValueError):
        widen_policy(old, 1500, 512)  # obs_dim shrink

    with pytest.raises(ValueError):
        widen_policy(old, 1548, 200)  # n_actions shrink

    with pytest.raises(ValueError):
        widen_policy(old, 1548, 512, new_hidden_sizes=(64, 64))  # hidden_sizes mismatch


def test_init_from_flag_smoke(tmp_path):
    from catan_rl.rl.self_play import SelfPlayTrainer

    old_policy = ActorCritic(obs_dim=1520, n_actions=256, hidden_sizes=(32, 32))
    old_opt = torch.optim.Adam(old_policy.parameters(), lr=1e-3)
    # Give the old optimizer real Adam moment state, so the "fresh optimizer"
    # assertion below is meaningful rather than trivially true because
    # nothing was ever stepped.
    logits, value = old_policy(torch.randn(4, 1520))
    (logits.sum() + value.sum()).backward()
    old_opt.step()
    assert len(old_opt.state) > 0
    ckpt_path = save_checkpoint(
        tmp_path / "old_run" / "checkpoints", old_policy, old_opt,
        iteration=100, config={}, metrics={},
    )

    trainer = SelfPlayTrainer(
        {"obs_mode": "self_play", "hidden_sizes": [32, 32]},
        run_dir=tmp_path / "new_run",
        device="cpu",
        init_from=ckpt_path,
    )
    try:
        assert trainer.policy.obs_dim == 1548
        assert trainer.policy.n_actions == 512
        # init_from must not carry over the old checkpoint's optimizer
        # state (Adam moments/step count) -- warm-starting reuses only
        # the weights, never the optimizer.
        assert len(trainer.trainer.optimizer.state) == 0
    finally:
        trainer.close()


def test_init_from_already_at_target_dims_still_validates_hidden_sizes(tmp_path):
    """A checkpoint already at the current run's target obs/action dims
    must still have its hidden_sizes validated against the run's config,
    not silently adopted as-is."""
    from catan_rl.rl.self_play import SelfPlayTrainer

    full_dim_policy = ActorCritic(obs_dim=1548, n_actions=512, hidden_sizes=(32, 32))
    opt = torch.optim.Adam(full_dim_policy.parameters(), lr=1e-3)
    ckpt_path = save_checkpoint(
        tmp_path / "old_run" / "checkpoints", full_dim_policy, opt,
        iteration=100, config={}, metrics={},
    )

    with pytest.raises(ValueError):
        SelfPlayTrainer(
            {"obs_mode": "self_play", "hidden_sizes": [64, 64]},  # mismatched
            run_dir=tmp_path / "new_run",
            device="cpu",
            init_from=ckpt_path,
        )
