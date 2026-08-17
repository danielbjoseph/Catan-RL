# Parallel CPU Training

## Overview

Catan RL training now supports distributing game collection across multiple CPU cores. This allows faster data collection without requiring GPU resources.

## Quick Start

### Auto-Detect CPUs (Recommended)

By default, `num_workers: null` auto-detects available CPUs and uses all of them.

### Manual CPU Count

Set `num_workers` in your config:

```yaml
# configs/my_training.yaml
num_workers: 4  # Use exactly 4 workers
games_per_iteration: 16
```

## How It Works

1. **Game Collection**: N games are distributed evenly across M workers. Each worker runs its own game instances independently.
2. **Rollout Aggregation**: After all workers finish, their rollouts are combined into a single batch.
3. **PPO Training**: The aggregated batch is used for a single PPO gradient update (unchanged from sequential mode).

## Performance Expectations

- **1 CPU, 16 games/iter**: ~10-15 sec/iter (baseline)
- **4 CPUs, 16 games/iter**: ~3-5 sec/iter (depends on game length & CPU speed)
- **Speedup**: Generally linear with number of CPUs

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
- System memory is too low to spawn workers
- Multiprocessing is disabled on your platform

### Memory usage increased with parallel training

Each worker maintains its own game environments. Memory usage scales with `num_workers`. If you run out of memory:
1. Reduce `num_workers`
2. Reduce `games_per_iteration`
3. Reduce hidden network size (in `hidden_sizes` config)

## Development Notes

- Worker processes use `multiprocessing.Pool`
- Each worker gets an independent RNG seeded deterministically
- Rollout aggregation concatenates tensors along batch dimension
- PPO training remains single-process
