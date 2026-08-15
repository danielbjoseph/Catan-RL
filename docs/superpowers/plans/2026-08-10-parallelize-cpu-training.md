# Parallelize CPU Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Catan RL training to distribute game collection across multiple CPUs at runtime, auto-detecting available cores and scaling gracefully from 1 to N workers.

**Architecture:** Games are collected in parallel by spawning independent worker processes that each play a subset of games in their own environment. A main process aggregates rollouts from all workers into a single batch for PPO training. The system auto-detects CPU count, accepts manual override via config, and falls back to single-process if parallelization fails. PPO training remains single-process.

**Tech Stack:** 
- `multiprocessing.Pool` for worker management
- Shared seed management to maintain determinism
- Per-worker game collection (existing `collect_rollouts` logic reused)

## Global Constraints

- No GPU distribution (CPU-only parallelization)
- Preserve determinism: same seed + worker count should yield same aggregate batch
- Keep existing sequential `collect_rollouts()` function unchanged (reuse for workers)
- All 4-seat per-game aggregation logic stays in worker processes
- PPO trainer and policy remain single-process
- Fallback to sequential collection if worker pool creation fails

---

## File Structure

```
catan_rl/rl/
  ├── rollout.py (MODIFY)
  │   └── collect_rollouts() stays unchanged
  │   └── NEW: collect_rollouts_parallel() - orchestrates worker pool
  ├── parallel_rollout.py (NEW)
  │   └── _worker_collect_games() - worker process target
  │   └── ParallelRolloutConfig - configuration for parallelization
  └── self_play.py (MODIFY)
      └── Add num_workers to _RUN_DEFAULTS
      └── Use parallel collector in train()

configs/
  └── ppo_baseline.yaml (MODIFY)
      └── Add num_workers: null (auto-detect)

tests/
  ├── test_parallel_rollout.py (NEW)
  │   └── Test worker function
  │   └── Test pool orchestration
  │   └── Test seed management
  └── test_rollout_equivalence.py (NEW)
      └── Verify parallel(N workers) batch == sequential(N games)
```

---

## Task 1: Create Parallel Rollout Worker Module

**Files:**
- Create: `catan_rl/rl/parallel_rollout.py`

**Interfaces:**
- Consumes: `rollout.collect_rollouts(policy, n_games, rules_profile, gamma, lam, max_turns, seed, ...)` signature
- Produces: `_worker_collect_games(worker_id, n_games, policy_state, rules_profile, gamma, lam, max_turns, seed_base) -> Batch`

**Steps:**

- [ ] **Step 1: Write the test for worker initialization**

Create `tests/test_parallel_rollout.py`:

```python
import pytest
import torch
from catan_rl.rl.parallel_rollout import _worker_collect_games
from catan_rl.rl.models import ActorCritic
from catan_rl.env.rules_profile import RulesProfile
from catan_rl.env.observation import obs_dim_for_mode


def test_worker_collect_games_returns_batch():
    """Worker should return a valid Batch with expected structure."""
    policy = ActorCritic(obs_dim=obs_dim_for_mode("self_play"), hidden_sizes=(512, 512))
    profile = RulesProfile.get("simplified_v1")
    
    batch = _worker_collect_games(
        worker_id=0,
        n_games=2,
        policy=policy,
        rules_profile=profile,
        gamma=0.99,
        lam=0.95,
        max_turns=500,
        seed_base=42,
        opponent_pool=None,
        cfg=None,
    )
    
    assert batch is not None
    assert len(batch) > 0
    assert batch.obs.shape[0] == len(batch.actions)
    assert batch.obs.shape[0] == len(batch.advantages)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C:\Users\16093\Downloads\Catan RL
pytest tests/test_parallel_rollout.py::test_worker_collect_games_returns_batch -v
```

Expected: FAIL with "No module named 'catan_rl.rl.parallel_rollout'"

- [ ] **Step 3: Implement parallel_rollout.py**

Create `catan_rl/rl/parallel_rollout.py`:

```python
"""
Parallel rollout collection across multiple CPU workers.

Each worker process runs games independently using the same shared policy.
Workers are seeded deterministically from a base seed + worker_id to ensure
reproducibility while keeping workers' RNGs independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .rollout import Batch, collect_rollouts
from ..env.rules_profile import RulesProfile
from ..bots import BotCallable


@dataclass
class ParallelRolloutConfig:
    """Configuration for parallel rollout collection."""
    num_workers: Optional[int] = None  # None = auto-detect CPU count
    fallback_to_sequential: bool = True  # If parallelization fails, fall back to sequential


def _worker_collect_games(
    worker_id: int,
    n_games: int,
    policy,
    rules_profile: RulesProfile,
    gamma: float,
    lam: float,
    max_turns: int,
    seed_base: int,
    opponent_pool: Optional[List[Dict]] = None,
    cfg: Optional[Dict] = None,
) -> Batch:
    """
    Worker process target: collect n_games in isolation.
    
    Each worker gets a deterministic seed derived from seed_base + worker_id,
    ensuring reproducibility and independence across workers.
    
    Args:
        worker_id: Unique identifier for this worker (0 to num_workers-1)
        n_games: Number of games for this worker to collect
        policy: Shared ActorCritic policy (pickled by multiprocessing)
        rules_profile: Game rules
        gamma: Discount factor for GAE
        lam: Lambda parameter for GAE
        max_turns: Max turns per game
        seed_base: Base seed (from main process)
        opponent_pool: Optional list of opponent dicts for opponent rotation
        cfg: Optional dict with extra config (trace_every, etc)
    
    Returns:
        Batch of collected rollouts (ready for PPO training)
    """
    worker_seed = seed_base ^ (worker_id * 0x12345)  # Deterministic per-worker seed
    
    batch = collect_rollouts(
        policy=policy,
        n_games=n_games,
        rules_profile=rules_profile,
        gamma=gamma,
        lam=lam,
        max_turns=max_turns,
        seed=worker_seed,
        opponent_pool=opponent_pool,
        trace_every=cfg.get("trace_every") if cfg else None,
    )
    
    return batch
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_parallel_rollout.py::test_worker_collect_games_returns_batch -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\16093\Downloads\Catan RL
git add catan_rl/rl/parallel_rollout.py tests/test_parallel_rollout.py
git commit -m "feat(rl): add worker function for parallel rollout collection"
```

---

## Task 2: Implement Parallel Batch Aggregation in rollout.py

**Files:**
- Modify: `catan_rl/rl/rollout.py` (at end, add new function)

**Interfaces:**
- Consumes: `Batch` dataclass and `_worker_collect_games()` from parallel_rollout
- Produces: `collect_rollouts_parallel(policy, n_games, num_workers, rules_profile, gamma, lam, max_turns, seed, ...) -> Batch`

**Steps:**

- [ ] **Step 1: Write test for parallel batch aggregation**

Add to `tests/test_parallel_rollout.py`:

```python
def test_collect_rollouts_parallel_aggregates_batches():
    """Parallel collection with N workers should aggregate N*games_per_worker into one batch."""
    from catan_rl.rl.rollout import collect_rollouts_parallel
    
    policy = ActorCritic(obs_dim=obs_dim_for_mode("self_play"), hidden_sizes=(512, 512))
    profile = RulesProfile.get("simplified_v1")
    
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
    assert isinstance(batch, Batch)
    # 2 workers each collect 4 games = 8 total games worth of transitions
    # Exact transition count depends on game length, so we just check it's > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_parallel_rollout.py::test_collect_rollouts_parallel_aggregates_batches -v
```

Expected: FAIL with "No function named 'collect_rollouts_parallel'"

- [ ] **Step 3: Implement collect_rollouts_parallel in rollout.py**

Add to end of `catan_rl/rl/rollout.py`:

```python
from multiprocessing import Pool, cpu_count
from ..rl.parallel_rollout import _worker_collect_games


def collect_rollouts_parallel(
    policy: ActorCritic,
    n_games: int,
    num_workers: Optional[int] = None,
    rules_profile: Optional[RulesProfile] = None,
    gamma: float = 0.99,
    lam: float = 0.95,
    max_turns: int = 500,
    seed: int = 42,
    opponent_pool: Optional[List[Dict]] = None,
    trace_every: Optional[int] = None,
    cfg: Optional[Dict] = None,
) -> Batch:
    """
    Collect rollouts by distributing game collection across multiple workers.
    
    Args:
        policy: The shared policy to use for all workers
        n_games: Total number of games to collect (distributed across workers)
        num_workers: Number of worker processes (None = auto-detect CPU count)
        rules_profile: Game rules
        gamma: Discount factor for GAE
        lam: Lambda parameter for GAE
        max_turns: Max turns per game
        seed: Base seed for all workers (each worker gets seed ^ (worker_id * 0x12345))
        opponent_pool: Optional list of opponent dicts for opponent rotation
        trace_every: If not None, trace games at this interval (sent to workers)
        cfg: Optional dict with extra config
    
    Returns:
        Aggregated Batch from all workers
    
    Falls back to sequential collection if worker pool creation fails.
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    if num_workers <= 1:
        # Single worker: just use sequential collection
        return collect_rollouts(
            policy=policy,
            n_games=n_games,
            rules_profile=rules_profile,
            gamma=gamma,
            lam=lam,
            max_turns=max_turns,
            seed=seed,
            opponent_pool=opponent_pool,
            trace_every=trace_every,
        )
    
    # Distribute games evenly across workers
    games_per_worker = n_games // num_workers
    remainder = n_games % num_workers
    game_counts = [games_per_worker + (1 if i < remainder else 0) for i in range(num_workers)]
    
    try:
        with Pool(processes=num_workers) as pool:
            worker_args = [
                (
                    i,
                    game_counts[i],
                    policy,
                    rules_profile,
                    gamma,
                    lam,
                    max_turns,
                    seed,
                    opponent_pool,
                    {"trace_every": trace_every} if trace_every else None,
                )
                for i in range(num_workers)
            ]
            
            batches = pool.starmap(_worker_collect_games, worker_args)
    except Exception as e:
        print(f"[Parallel Collection] Worker pool failed, falling back to sequential: {e}")
        return collect_rollouts(
            policy=policy,
            n_games=n_games,
            rules_profile=rules_profile,
            gamma=gamma,
            lam=lam,
            max_turns=max_turns,
            seed=seed,
            opponent_pool=opponent_pool,
            trace_every=trace_every,
        )
    
    # Aggregate all batches into a single batch
    return _aggregate_batches(batches)


def _aggregate_batches(batches: List[Batch]) -> Batch:
    """
    Concatenate multiple Batch objects into a single aggregated Batch.
    
    Used after collecting rollouts from multiple workers.
    """
    if not batches:
        raise ValueError("No batches to aggregate")
    
    if len(batches) == 1:
        return batches[0]
    
    # Concatenate all tensors along the batch dimension (dim 0)
    aggregated = Batch(
        obs=torch.cat([b.obs for b in batches], dim=0),
        masks=torch.cat([b.masks for b in batches], dim=0),
        actions=torch.cat([b.actions for b in batches], dim=0),
        logprobs=torch.cat([b.logprobs for b in batches], dim=0),
        values=torch.cat([b.values for b in batches], dim=0),
        advantages=torch.cat([b.advantages for b in batches], dim=0),
        returns=torch.cat([b.returns for b in batches], dim=0),
        seat_ids=torch.cat([b.seat_ids for b in batches], dim=0),
        episode_ids=torch.cat([b.episode_ids for b in batches], dim=0),
        stats=_aggregate_stats([b.stats for b in batches]),
    )
    
    return aggregated


def _aggregate_stats(stats_list: List[Dict]) -> Dict:
    """
    Merge stats dicts from multiple workers by summing numeric values.
    
    Used to combine stats like 'num_games', 'total_turns', etc.
    """
    if not stats_list:
        return {}
    
    result = {}
    all_keys = set()
    for s in stats_list:
        all_keys.update(s.keys())
    
    for key in all_keys:
        values = [s.get(key, 0) for s in stats_list if key in s]
        if values and isinstance(values[0], (int, float)):
            result[key] = sum(values)
        else:
            result[key] = values[0] if values else None
    
    return result
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_parallel_rollout.py::test_collect_rollouts_parallel_aggregates_batches -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\16093\Downloads\Catan RL
git add catan_rl/rl/rollout.py
git commit -m "feat(rl): implement parallel batch aggregation in rollout"
```

---

## Task 3: Integrate Parallel Collection into Self-Play Training

**Files:**
- Modify: `catan_rl/rl/self_play.py`
- Modify: `configs/ppo_baseline.yaml`

**Interfaces:**
- Consumes: `collect_rollouts_parallel()` from rollout module
- Produces: Updated `SelfPlayTrainer.train()` that uses parallel collector when `num_workers > 1`

**Steps:**

- [ ] **Step 1: Write test for SelfPlayTrainer with num_workers config**

Add to `tests/test_parallel_rollout.py`:

```python
def test_self_play_trainer_accepts_num_workers_config():
    """SelfPlayTrainer should accept num_workers in config and use it."""
    from catan_rl.rl.self_play import SelfPlayTrainer
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {
            "experiment_name": "test_parallel",
            "seed": 42,
            "iterations": 1,
            "games_per_iteration": 4,
            "num_workers": 2,  # Should be accepted without error
            "device": "cpu",
        }
        
        trainer = SelfPlayTrainer(config=cfg, run_dir=tmpdir)
        assert trainer.cfg.get("num_workers") == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_parallel_rollout.py::test_self_play_trainer_accepts_num_workers_config -v
```

Expected: FAIL (test will pass but verify the default config doesn't have num_workers yet)

- [ ] **Step 3: Update _RUN_DEFAULTS in self_play.py**

Modify `catan_rl/rl/self_play.py` at line 41 (in `_RUN_DEFAULTS` dict):

```python
_RUN_DEFAULTS = {
    "experiment_name": "ppo_baseline",
    "seed": 42,
    "iterations": 500,
    "games_per_iteration": 16,
    "num_workers": None,  # NEW: None = auto-detect CPU count, or set explicitly
    "eval_interval": 25,
    "eval_games": 12,
    "checkpoint_interval": 25,
    "rules_profile": "simplified_v1",
    "max_turns": 500,
    "reward_win": 1.0,
    "reward_loss": -1.0,
    "obs_mode": "self_play",
    "belief_blend": 0.25,
    "belief_noise": 0.5,
    "device": "cpu",
    "trace_every": None,
    "opponents": None,
    "n_policy_seats": 1,
    "eval_personalities": None,
}
```

- [ ] **Step 4: Update imports in self_play.py**

At top of `catan_rl/rl/self_play.py`, add to existing imports:

```python
from .rollout import collect_rollouts_parallel
```

- [ ] **Step 5: Update train() method in SelfPlayTrainer to use parallel collection**

Find the `train()` method in `SelfPlayTrainer` (around line 163), and replace the `collect_rollouts()` call:

Old code (around line 172):
```python
batch = collect_rollouts(
    self.policy,
    n_games=games_per_iter,
    rules_profile=self.profile,
    gamma=self.ppo_cfg.gamma,
    lam=self.ppo_cfg.gae_lambda,
    max_turns=int(self.cfg["max_turns"]),
    seed=int(self.cfg["seed"]) + it * games_per_iter,
    ...
)
```

New code:
```python
num_workers = self.cfg.get("num_workers")
batch = collect_rollouts_parallel(
    self.policy,
    n_games=games_per_iter,
    num_workers=num_workers,
    rules_profile=self.profile,
    gamma=self.ppo_cfg.gamma,
    lam=self.ppo_cfg.gae_lambda,
    max_turns=int(self.cfg["max_turns"]),
    seed=int(self.cfg["seed"]) + it * games_per_iter,
    opponent_pool=self.opponent_pool if hasattr(self, 'opponent_pool') else None,
    trace_every=self.trace_every,
)
```

- [ ] **Step 6: Add num_workers to ppo_baseline.yaml**

Modify `configs/ppo_baseline.yaml` to include:

```yaml
# ... existing config ...
num_workers: null  # null = auto-detect CPU count; set to integer to override
# ... rest of config ...
```

- [ ] **Step 7: Run test to verify it passes**

```bash
pytest tests/test_parallel_rollout.py::test_self_play_trainer_accepts_num_workers_config -v
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
cd C:\Users\16093\Downloads\Catan RL
git add catan_rl/rl/self_play.py configs/ppo_baseline.yaml
git commit -m "feat(rl): integrate parallel collection into SelfPlayTrainer"
```

---

## Task 4: Add Equivalence Test (Parallel vs Sequential Produces Same Batch)

**Files:**
- Create: `tests/test_rollout_equivalence.py`

**Interfaces:**
- Consumes: `collect_rollouts()` and `collect_rollouts_parallel()` functions
- Produces: Deterministic equivalence tests

**Steps:**

- [ ] **Step 1: Write equivalence test**

Create `tests/test_rollout_equivalence.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it passes**

```bash
pytest tests/test_rollout_equivalence.py -v
```

Expected: PASS (tests should take 30-60 seconds)

- [ ] **Step 3: Commit**

```bash
cd C:\Users\16093\Downloads\Catan RL
git add tests/test_rollout_equivalence.py
git commit -m "test: add equivalence tests for parallel vs sequential rollout collection"
```

---

## Task 5: Integration Test and Documentation

**Files:**
- Create: `docs/PARALLEL_TRAINING.md` (documentation)
- Modify: existing test to verify end-to-end

**Steps:**

- [ ] **Step 1: Write integration test**

Add to `tests/test_rollout_equivalence.py`:

```python
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
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_rollout_equivalence.py::test_self_play_trainer_runs_with_parallel_collection -v
```

Expected: PASS

- [ ] **Step 3: Create documentation**

Create `docs/PARALLEL_TRAINING.md`:

```markdown
# Parallel CPU Training

## Overview

Catan RL training now supports distributing game collection across multiple CPU cores. This allows faster data collection without requiring GPU resources.

## Quick Start

### Auto-Detect CPUs (Recommended)

```bash
python scripts/train_self_play.py configs/ppo_baseline.yaml
```

By default, `num_workers: null` auto-detects available CPUs and uses all of them.

### Manual CPU Count

Set `num_workers` in your config or command line:

```yaml
# configs/my_training.yaml
num_workers: 4  # Use exactly 4 workers
games_per_iteration: 16
```

Or override at runtime:

```bash
python scripts/train_self_play.py --num-workers 8 configs/ppo_baseline.yaml
```

## How It Works

1. **Game Collection**: N games are distributed evenly across M workers. Each worker runs its own game instances independently.
2. **Rollout Aggregation**: After all workers finish, their rollouts are combined into a single batch.
3. **PPO Training**: The aggregated batch is used for a single PPO gradient update (unchanged from sequential mode).
4. **Determinism**: Same seed + worker count = same batch, regardless of order.

## Performance Expectations

- **1 CPU, 16 games/iter**: ~10-15 sec/iter (baseline)
- **4 CPUs, 16 games/iter**: ~3-5 sec/iter (depends on game length & CPU speed)
- **Speedup**: Generally linear with number of CPUs, up to ~N-1x (N-1 because the main process also does some work)

## Configuration

Add `num_workers` to your config file:

```yaml
# Auto-detect CPUs (default)
num_workers: null

# Or specify explicitly
num_workers: 2
num_workers: 8
num_workers: 16
```

## Troubleshooting

### Parallelization fails and falls back to sequential

If the worker pool fails to initialize, training automatically falls back to sequential collection with a warning. This can happen if:
- The policy is not pickleable (shouldn't happen with our ActorCritic)
- System memory is too low to spawn workers
- Multiprocessing is disabled on your platform

Check the console output for the error message.

### Different seeds produce different batches

This is expected. Each (`num_workers`, `seed`) pair should produce the **same** batch deterministically, but different seeds produce different games.

### Memory usage increased with parallel training

Each worker maintains its own game environments and policy copy. Memory usage scales roughly with `num_workers`. If you run out of memory:
1. Reduce `num_workers`
2. Reduce `games_per_iteration`
3. Reduce hidden network size (in `hidden_sizes` config)

## Development Notes

- Worker processes use `multiprocessing.Pool` (fork on Unix, spawn on Windows)
- Each worker gets an independent RNG seeded as `seed_base ^ (worker_id * 0x12345)`
- Rollout aggregation concatenates tensors along batch dimension
- PPO training remains single-process (no gradient aggregation)

## Future Improvements

- Support for distributed training across multiple machines (requires gradient aggregation)
- Worker load balancing (distribute by estimated game length, not just count)
- Streaming rollout aggregation to reduce memory spike during batch collection
```

- [ ] **Step 4: Run all parallel tests to verify they pass**

```bash
pytest tests/test_parallel_rollout.py tests/test_rollout_equivalence.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\16093\Downloads\Catan RL
git add docs/PARALLEL_TRAINING.md tests/test_rollout_equivalence.py
git commit -m "docs: add parallel training guide and integration test"
```

---

## Task 6: Update Script Entry Point (Optional CLI Flag)

**Files:**
- Modify: `scripts/train_self_play.py` (if it exists, or create a wrapper)

**Steps:**

- [ ] **Step 1: Check if train_self_play.py supports CLI config overrides**

```bash
python scripts/train_self_play.py --help
```

If it already supports `--num-workers` or generic config overrides, skip this task.
If not, add support:

- [ ] **Step 2: Add --num-workers CLI flag**

Modify `scripts/train_self_play.py` to accept:

```python
import argparse

parser = argparse.ArgumentParser(description="Train Catan RL policy with self-play")
parser.add_argument("config", help="Path to config YAML file")
parser.add_argument("--num-workers", type=int, default=None, help="Number of workers for parallel collection (default: auto-detect)")
parser.add_argument("--device", type=str, default=None, help="Device to use (cpu or cuda)")
# ... other args ...

args = parser.parse_args()
cfg = load_config(args.config)
if args.num_workers is not None:
    cfg["num_workers"] = args.num_workers
if args.device is not None:
    cfg["device"] = args.device

trainer = SelfPlayTrainer(config=cfg)
trainer.train()
```

- [ ] **Step 3: Test CLI flag**

```bash
python scripts/train_self_play.py configs/ppo_baseline.yaml --num-workers 4 --iterations 1
```

Expected: Runs successfully with 4 workers

- [ ] **Step 4: Commit (if modified)**

```bash
git add scripts/train_self_play.py
git commit -m "feat(script): add --num-workers CLI flag to train_self_play"
```

---

## Verification Checklist

Before considering this complete:

- [ ] All tests pass: `pytest tests/test_parallel_rollout.py tests/test_rollout_equivalence.py -v`
- [ ] Sequential fallback works (set `num_workers: 1`)
- [ ] Parallel runs faster than sequential (benchmark: `python -m cProfile scripts/train_self_play.py ...`)
- [ ] Determinism verified: same seed + worker count produces same batch twice
- [ ] Documentation is clear and includes examples
- [ ] Config file includes `num_workers: null` default
- [ ] No breaking changes to existing code (old configs without `num_workers` still work)

---

## Post-Implementation Notes

### Benchmarking Command

To measure speedup:

```bash
# Sequential (1 worker)
time python scripts/train_self_play.py configs/ppo_baseline.yaml --num-workers 1 --iterations 1

# Parallel (auto-detect)
time python scripts/train_self_play.py configs/ppo_baseline.yaml --iterations 1
```

### Future Enhancements

1. **Multi-Machine Training**: Add distributed.launch support for gradient aggregation across nodes
2. **Worker Pool Warmup**: Pre-create worker pool once instead of per-iteration
3. **Adaptive Batch Size**: Adjust `games_per_iteration` based on wall-clock iteration time
4. **Ray Integration**: Replace multiprocessing.Pool with Ray for better resource management
