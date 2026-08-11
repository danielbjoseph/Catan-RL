"""Parallel rollout worker module for CPU-based parallelization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from catan_rl.env.rules_profile import RulesProfile
from catan_rl.rl.models import ActorCritic
from catan_rl.rl.rollout import Batch, collect_rollouts


@dataclass
class ParallelRolloutConfig:
    """Configuration for parallel rollout collection.

    Args:
        num_workers: Number of worker processes. None = auto-detect CPU count.
        fallback_to_sequential: If parallelization fails, fall back to sequential mode.
    """

    num_workers: Optional[int] = None
    fallback_to_sequential: bool = True


def _worker_collect_games(
    worker_id: int,
    n_games: int,
    policy: ActorCritic,
    rules_profile: RulesProfile,
    gamma: float,
    lam: float,
    max_turns: int,
    seed_base: int,
    opponent_pool: Optional[List[Dict]] = None,
    cfg: Optional[Dict] = None,
) -> Batch:
    """Collect games in a worker process.

    Each worker derives a deterministic per-worker seed from seed_base and worker_id,
    ensuring reproducibility across multiple workers while maintaining different trajectories.

    Args:
        worker_id: Unique worker identifier (0-indexed).
        n_games: Number of games to collect in this worker.
        policy: The ActorCritic policy to use for collecting rollouts.
        rules_profile: Game rules profile to use.
        gamma: Discount factor for GAE.
        lam: Lambda parameter for GAE.
        max_turns: Maximum turns per game.
        seed_base: Base seed for reproducibility.
        opponent_pool: Optional opponent pool spec (format: {"pool": [...]}).
        cfg: Optional additional configuration (reserved for future use).

    Returns:
        A Batch containing the collected game data.
    """
    # Derive per-worker seed deterministically from worker_id and seed_base.
    # The XOR with (worker_id * 0x12345) ensures different seeds for different workers.
    per_worker_seed = seed_base ^ (worker_id * 0x12345)

    # Collect rollouts for this worker.
    batch = collect_rollouts(
        policy,
        n_games,
        rules_profile=rules_profile,
        gamma=gamma,
        lam=lam,
        max_turns=max_turns,
        seed=per_worker_seed,
        opponents=opponent_pool,
    )

    return batch
