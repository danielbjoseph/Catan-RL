# Catan RL — Self-Play PPO for Settlers of Catan

A full reinforcement-learning training system for 4-player Catan: a
rules-correct game engine, a PettingZoo-style multi-agent environment with
legal-action masking, scripted baseline bots, and a custom PPO self-play
trainer where all four seats share one improving policy.

Built to the spec in [`Outline.md`](Outline.md).

## Layout

```
catan_rl/
├── env/          game engine + environment
│   ├── board.py            immutable board geometry (19 hexes / 54 vertices / 72 edges)
│   ├── game_state.py       mutable game state, serializable
│   ├── player_state.py     per-player state, build costs
│   ├── rules.py            state transition engine (apply_action)
│   ├── validators.py       legal action generation (source of truth for legality)
│   ├── scoring.py          VP, longest road (DFS), largest army
│   ├── actions.py          fixed 256-slot action catalog
│   ├── action_mask.py      legal_action_mask(state) -> bool[256]
│   ├── observation.py      observation vectors (self_play: 1520 dims, perfect: 1565)
│   ├── rules_profile.py    rules profiles (standard / simplified_v1 / custom YAML)
│   ├── pettingzoo_env.py   AEC multi-agent env (CatanAECEnv)
│   └── gym_wrapper.py      single-agent Gym-style wrapper
├── bots/         scripted baselines: random, greedy, heuristic
└── rl/           training stack
    ├── models.py           masked actor-critic MLP (always 256 logits)
    ├── rollout.py          4-seat rollout collection + per-seat GAE
    ├── ppo.py              custom PPO (no stable-baselines3)
    ├── self_play.py        training orchestrator + TensorBoard logging
    ├── evaluate.py         eval vs bots / past checkpoints (greedy play)
    └── checkpointing.py    state-dict checkpoints + JSON metadata

configs/          rules + PPO YAML configs
scripts/          CLI entry points
tests/            pytest suite (~190 tests)
docs/RUN_ON_GCLOUD.md   how to train on Google Cloud
```

## Setup

Python 3.11+ (developed on 3.14). Always use a venv:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
# run the test suite
python -m pytest tests/ -q

# watch a scripted-bot match
python scripts/render_match.py --bots greedy,random,heuristic,random --seed 3

# check environment throughput (spec target: >= 500 games/hour; measured ~67,000/hr
# with random agents on a laptop CPU)
python scripts/benchmark_throughput.py --games 100

# train the shared self-play policy (simplified_v1 rules)
python scripts/train_self_play.py --config configs/ppo_baseline.yaml

# monitor
tensorboard --logdir runs/

# evaluate checkpoints vs bots and prior snapshots
python scripts/evaluate_checkpoints.py --run runs/ppo_baseline --games 50 --vs random,greedy,prev

# watch the trained policy play
python scripts/render_match.py --ckpt runs/ppo_baseline/checkpoints/ckpt_000500.pt --seat 0 --bots greedy,greedy,greedy,greedy
```

## Design notes

- **Action space** — a fixed 256-slot catalog (`actions.py`); every
  `(action_type, parameter)` pair has a permanent index. The policy always
  outputs 256 logits; the environment masks illegal slots. Sub-phases
  (knight → move robber, road building → two road placements) reuse the same
  slots with a different mask.
- **Shared-policy self-play** — all four seats run the same network; each
  seat's transitions are collected in its own list and GAE is computed
  per seat (seats' turns interleave, so a naive sequence-wide GAE would be
  wrong). Winner gets +1, losers −1 (configurable).
- **Rules profiles** — `simplified_v1` disables dev cards for the first
  trainable curriculum stage; the engine supports full standard rules
  (dev cards, largest army, robber, ports, maritime trade). Player-to-player
  trading is intentionally deferred (spec Phase 5D).
- **Reproducibility** — board layout, dice, dev deck, and torch are all
  seeded; checkpoints carry config + metrics + architecture metadata.

## Training on Google Cloud

See [docs/RUN_ON_GCLOUD.md](docs/RUN_ON_GCLOUD.md).
