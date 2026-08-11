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


class TestAggregateBatches:
    def test_aggregate_batches_concatenates_tensors(self):
        """Verify _aggregate_batches concatenates tensors along dim=0."""
        from catan_rl.rl.rollout import _aggregate_batches

        # Create two simple batches
        batch1 = Batch(
            obs=torch.randn(10, OBS_DIM),
            masks=torch.ones(10, 121, dtype=torch.bool),
            actions=torch.randint(0, 121, (10,)),
            logprobs=torch.randn(10),
            values=torch.randn(10),
            advantages=torch.randn(10),
            returns=torch.randn(10),
            seat_ids=torch.zeros(10, dtype=torch.long),
            episode_ids=torch.zeros(10, dtype=torch.long),
            stats={"games_completed": 2},
        )
        batch2 = Batch(
            obs=torch.randn(5, OBS_DIM),
            masks=torch.ones(5, 121, dtype=torch.bool),
            actions=torch.randint(0, 121, (5,)),
            logprobs=torch.randn(5),
            values=torch.randn(5),
            advantages=torch.randn(5),
            returns=torch.randn(5),
            seat_ids=torch.ones(5, dtype=torch.long),
            episode_ids=torch.ones(5, dtype=torch.long),
            stats={"games_completed": 1},
        )

        # Aggregate
        aggregated = _aggregate_batches([batch1, batch2])

        # Verify shapes are concatenated
        assert aggregated.obs.shape == (15, OBS_DIM)
        assert aggregated.actions.shape == (15,)
        assert aggregated.logprobs.shape == (15,)
        assert aggregated.values.shape == (15,)
        assert aggregated.advantages.shape == (15,)
        assert aggregated.returns.shape == (15,)
        assert aggregated.seat_ids.shape == (15,)
        assert aggregated.episode_ids.shape == (15,)
        assert len(aggregated) == 15


class TestAggregateStats:
    def test_aggregate_stats_sums_numeric_values(self):
        """Verify _aggregate_stats sums numeric values."""
        from catan_rl.rl.rollout import _aggregate_stats

        stats_list = [
            {
                "games_completed": 10,
                "truncated_games": 2,
                "mean_episode_length": 100.0,
            },
            {
                "games_completed": 20,
                "truncated_games": 4,
                "mean_episode_length": 110.0,
            },
        ]

        aggregated = _aggregate_stats(stats_list)

        # Numeric values should be summed
        assert aggregated["games_completed"] == 30
        assert aggregated["truncated_games"] == 6
        assert aggregated["mean_episode_length"] == 210.0


class TestCollectRolloutsParallel:
    def test_collect_rollouts_parallel_fallback_to_sequential_when_workers_one(self):
        """Verify parallel collection falls back to sequential when num_workers <= 1."""
        from catan_rl.rl.rollout import collect_rollouts_parallel

        torch.manual_seed(0)
        policy = ActorCritic(hidden_sizes=(64, 64))

        batch = collect_rollouts_parallel(
            policy=policy,
            n_games=1,
            num_workers=1,
            rules_profile=FAST_PROFILE,
            gamma=0.999,
            lam=0.95,
            max_turns=500,
            seed=123,
        )

        assert isinstance(batch, Batch)
        assert len(batch) > 0
        assert batch.stats["games_completed"] == 1

    def test_collect_rollouts_parallel_with_explicit_workers(self):
        """Verify parallel collection with explicit num_workers."""
        from catan_rl.rl.rollout import collect_rollouts_parallel

        torch.manual_seed(0)
        policy = ActorCritic(hidden_sizes=(64, 64))

        batch = collect_rollouts_parallel(
            policy=policy,
            n_games=2,
            num_workers=2,
            rules_profile=FAST_PROFILE,
            gamma=0.999,
            lam=0.95,
            max_turns=500,
            seed=123,
        )

        assert isinstance(batch, Batch)
        assert len(batch) > 0
        assert batch.stats["games_completed"] == 2
