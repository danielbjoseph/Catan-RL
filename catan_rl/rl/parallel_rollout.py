"""Parallel rollout worker module for CPU-based parallelization."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from catan_rl.env.rules_profile import RulesProfile
from catan_rl.rl.logger import StructuredLogger
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
    rules_profile: Union[str, RulesProfile, None],
    gamma: float,
    lam: float,
    max_turns: int,
    seed_base: Optional[int],
    device: str,
    obs_mode: str,
    reward_win: float,
    reward_loss: float,
    belief_blend: float,
    belief_noise: float,
    trace_dir: Optional[Union[str, Path]],
    trace_every: Optional[int],
    trace_prefix: str,
    opponents: Optional[Dict],
    n_policy_seats: int,
    logger: Optional[dict] = None,
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
        device: Device to use for policy inference.
        obs_mode: Observation mode ("self_play" or "realistic").
        reward_win: Reward for winning.
        reward_loss: Penalty for losing.
        belief_blend: Belief tracker blending factor for realistic mode.
        belief_noise: Belief tracker noise level for realistic mode.
        trace_dir: Directory to save game traces.
        trace_every: Interval for recording game traces.
        trace_prefix: Prefix for trace filenames.
        opponents: Opponent pool specification (format: {"pool": [...]}).
        n_policy_seats: Number of seats controlled by the policy.
        logger: Optional dict with 'run_id' and 'log_dir' to initialize logger in worker.

    Returns:
        A Batch containing the collected game data.
    """
    # Initialize logger in this worker process
    worker_logger = None
    if logger is not None:
        worker_logger = StructuredLogger(logger["run_id"], logger.get("log_dir"))

    # Derive per-worker seed deterministically from worker_id and seed_base.
    # The XOR with (worker_id * 0x12345) ensures different seeds for different workers.
    per_worker_seed = None if seed_base is None else seed_base ^ (worker_id * 0x12345)

    # Seed the RNG for numpy and torch to ensure determinism in worker process
    if per_worker_seed is not None:
        np.random.seed(per_worker_seed)
        torch.manual_seed(per_worker_seed)
        torch.set_num_threads(1)  # Prevent oversubscription with N workers × N threads

    start_time = time.time()

    # Collect rollouts for this worker, passing through all parameters.
    batch = collect_rollouts(
        policy,
        n_games,
        rules_profile=rules_profile,
        gamma=gamma,
        lam=lam,
        max_turns=max_turns,
        seed=per_worker_seed,
        device=device,
        obs_mode=obs_mode,
        reward_win=reward_win,
        reward_loss=reward_loss,
        belief_blend=belief_blend,
        belief_noise=belief_noise,
        trace_dir=trace_dir,
        trace_every=trace_every,
        trace_prefix=trace_prefix,
        opponents=opponents,
        n_policy_seats=n_policy_seats,
    )

    elapsed = time.time() - start_time

    if worker_logger is not None:
        worker_logger.log_event(
            "rollout_complete",
            worker_id=worker_id,
            n_games=n_games,
            batch_size=len(batch),
            elapsed_sec=elapsed,
            games_per_sec=n_games / elapsed if elapsed > 0 else 0,
        )

    return batch
