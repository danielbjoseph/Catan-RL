"""Tests for parallel rollout worker module."""

import torch
import pytest

from catan_rl.env.observation import OBS_DIM
from catan_rl.env.rules_profile import RulesProfile
from catan_rl.rl.models import ActorCritic
from catan_rl.rl.rollout import Batch
from catan_rl.rl.parallel_rollout import _worker_collect_games, ParallelRolloutConfig


FAST_PROFILE = RulesProfile(name="fast", dev_cards_enabled=False, win_vp=8)


class TestWorkerCollectGames:
    def test_worker_collect_games_returns_batch(self):
        """Verify worker returns valid Batch with expected structure."""
        torch.manual_seed(0)
        policy = ActorCritic(hidden_sizes=(64, 64))

        batch = _worker_collect_games(
            worker_id=0,
            n_games=1,
            policy=policy,
            rules_profile=FAST_PROFILE,
            gamma=0.999,
            lam=0.95,
            max_turns=500,
            seed_base=123,
        )

        # Verify it's a Batch
        assert isinstance(batch, Batch)

        # Verify it has data
        assert len(batch) > 0

        # Verify tensor shapes are consistent
        n = batch.obs.shape[0]
        assert batch.obs.shape == (n, OBS_DIM)
        assert batch.actions.shape == (n,)
        assert batch.logprobs.shape == (n,)
        assert batch.values.shape == (n,)
        assert batch.advantages.shape == (n,)
        assert batch.returns.shape == (n,)
        assert batch.seat_ids.shape == (n,)
        assert batch.episode_ids.shape == (n,)

        # Verify stats
        assert batch.stats is not None
        assert batch.stats["games_completed"] == 1


class TestParallelRolloutConfig:
    def test_config_defaults(self):
        """Verify ParallelRolloutConfig has correct defaults."""
        config = ParallelRolloutConfig()
        assert config.num_workers is None
        assert config.fallback_to_sequential is True

    def test_config_custom_values(self):
        """Verify ParallelRolloutConfig can be initialized with custom values."""
        config = ParallelRolloutConfig(num_workers=4, fallback_to_sequential=False)
        assert config.num_workers == 4
        assert config.fallback_to_sequential is False
