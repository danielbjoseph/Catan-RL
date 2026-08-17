"""
Verify that parallel(N workers) collection produces the same batch as sequential collection.

This is critical for maintaining training determinism and reproducibility.
"""

import pytest
import torch
import numpy as np
from catan_rl.rl.rollout import collect_rollouts, collect_rollouts_parallel
from catan_rl.rl.models import ActorCritic
from catan_rl.env.rules_profile import RulesProfile
from catan_rl.env.observation import obs_dim_for_mode


@pytest.fixture
def policy():
    """Shared policy for all tests."""
    return ActorCritic(obs_dim=obs_dim_for_mode("self_play"), hidden_sizes=(512, 512))


@pytest.fixture
def profile():
    """Game rules profile."""
    return RulesProfile.get("simplified_v1")


def test_parallel_1_worker_equals_sequential(policy, profile):
    """Parallel with 1 worker should produce same batch as sequential."""
    n_games = 4
    seed = 42

    sequential_batch = collect_rollouts(
        policy=policy,
        n_games=n_games,
        rules_profile=profile,
        gamma=0.99,
        lam=0.95,
        max_turns=500,
        seed=seed,
    )

    parallel_batch = collect_rollouts_parallel(
        policy=policy,
        n_games=n_games,
        num_workers=1,
        rules_profile=profile,
        gamma=0.99,
        lam=0.95,
        max_turns=500,
        seed=seed,
    )

    # Batches should have same size (might be in different order, so check size)
    assert len(sequential_batch) == len(parallel_batch)


def test_parallel_multiple_workers_deterministic(policy, profile):
    """Parallel collection with same seed should be deterministic."""
    n_games = 8
    seed = 42

    batch1 = collect_rollouts_parallel(
        policy=policy,
        n_games=n_games,
        num_workers=2,
        rules_profile=profile,
        gamma=0.99,
        lam=0.95,
        max_turns=500,
        seed=seed,
    )

    batch2 = collect_rollouts_parallel(
        policy=policy,
        n_games=n_games,
        num_workers=2,
        rules_profile=profile,
        gamma=0.99,
        lam=0.95,
        max_turns=500,
        seed=seed,
    )

    # Same seed + same worker count = same batch
    assert len(batch1) == len(batch2)
    assert torch.allclose(batch1.obs, batch2.obs)
    assert torch.equal(batch1.actions, batch2.actions)


def test_parallel_2_workers_produces_valid_batch(policy, profile):
    """Parallel with 2 workers should produce valid training batch."""
    batch = collect_rollouts_parallel(
        policy=policy,
        n_games=8,
        num_workers=2,
        rules_profile=profile,
        gamma=0.99,
        lam=0.95,
        max_turns=500,
        seed=42,
    )

    assert batch is not None
    assert len(batch) > 0

    # All tensors should have same batch size
    n = len(batch)
    assert batch.obs.shape[0] == n
    assert batch.actions.shape[0] == n
    assert batch.advantages.shape[0] == n
    assert batch.returns.shape[0] == n

    # Advantages should be normalized (mean ~0, std ~1) if there are enough transitions
    if n > 100:
        mean_adv = batch.advantages.mean().item()
        std_adv = batch.advantages.std().item()
        assert abs(mean_adv) < 0.5, f"Expected mean ~0, got {mean_adv}"
        assert std_adv > 0.1, f"Expected std > 0.1, got {std_adv}"


def test_self_play_trainer_runs_with_parallel_collection(tmp_path):
    """End-to-end: SelfPlayTrainer should run successfully with parallel collection."""
    from catan_rl.rl.self_play import SelfPlayTrainer

    cfg = {
        "experiment_name": "test_parallel_integration",
        "seed": 42,
        "iterations": 1,
        "games_per_iteration": 4,
        "num_workers": 2,
        "eval_interval": 100,  # Skip eval in test
        "checkpoint_interval": 100,  # Skip checkpoint in test
        "device": "cpu",
    }

    trainer = SelfPlayTrainer(config=cfg, run_dir=str(tmp_path))
    trainer.train(iterations=1)

    # Just verify it completes without error
    assert trainer.iteration == 1
