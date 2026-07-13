"""Tests for checkpoint save/load round-trips."""

import json

import torch

from catan_rl.rl.checkpointing import (
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    load_policy,
    save_checkpoint,
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


def test_latest_and_list(tmp_path):
    policy, opt = _make()
    p1 = save_checkpoint(tmp_path, policy, opt, 1, {}, {})
    p10 = save_checkpoint(tmp_path, policy, opt, 10, {}, {})
    p2 = save_checkpoint(tmp_path, policy, opt, 2, {}, {})
    assert list_checkpoints(tmp_path) == [p1, p2, p10]
    assert latest_checkpoint(tmp_path) == p10
    assert latest_checkpoint(tmp_path / "empty") is None
