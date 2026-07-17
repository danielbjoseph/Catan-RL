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
│   ├── observation.py      observation vectors (self_play/perfect/realistic/global)
│   ├── belief.py           BeliefTracker: per-opponent hand/dev-deck estimates (realistic mode)
│   ├── trace.py            game-trace recording format (for the replay dashboard)
│   ├── rules_profile.py    rules profiles (standard / simplified_v1 / custom YAML)
│   ├── pettingzoo_env.py   AEC multi-agent env (CatanAECEnv)
│   └── gym_wrapper.py      single-agent Gym-style wrapper
├── bots/         scripted baselines: random, greedy, heuristic
├── rl/           training stack
│   ├── models.py           masked actor-critic MLP (always 256 logits)
│   ├── rollout.py          4-seat rollout collection + per-seat GAE
│   ├── ppo.py              custom PPO (no stable-baselines3)
│   ├── self_play.py        training orchestrator + TensorBoard logging
│   ├── evaluate.py         eval vs bots / past checkpoints / cross-mode policy-vs-policy
│   └── checkpointing.py    state-dict checkpoints + JSON metadata
└── dashboard/    Flask backend + static frontend for browsing recorded traces

configs/          rules + PPO YAML configs
scripts/          CLI entry points
tests/            pytest suite (280 tests)
docs/RULES_AUDIT.md     rules-correctness audit vs. the official Catan rulebook
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

## Observation modes

The observer always sees the board, its own hand exactly, and rotated
public per-player info (VP, resource/dev-card counts, road/settlement/city
counts, longest-road/largest-army holders). Modes differ in what they add
about opponents' hidden information:

| Mode | Dim | What the observer sees |
|---|---|---|
| `self_play` | 1520 | Own hand exact; opponents' hands are not exposed (public info only). |
| `perfect` | 1565 | `self_play` + every opponent's hand and dev cards exposed exactly (oracle view). |
| `realistic` | 1549 | `self_play` + a per-opponent *believed* hand (tracked from public actions, then blended toward uniform and Gaussian-noised) and its uncertainty, plus a believed dev-deck composition/count and the exact bank. |
| `global` | 1576 | `perfect` + the true remaining dev-deck composition/count and the exact bank. |

Dims are computed from `catan_rl/env/observation.py` (`OBS_DIM`,
`OBS_DIM_PERFECT`, `OBS_DIM_REALISTIC`, `OBS_DIM_GLOBAL`); use
`obs_dim_for_mode(mode)` rather than hardcoding a number.

`realistic` mode is driven by a `BeliefTracker` (`catan_rl/env/belief.py`)
that maintains a per-opponent expected-hand estimate from publicly visible
actions (builds, trades, robber steals, bank trades) and a dev-deck
estimate. Two knobs control how noisy the belief is:

- `belief_blend` — blends the tracked expected-hand vector toward a
  uniform prior (0 = pure tracker estimate, 1 = fully uniform).
- `belief_noise` — stddev scale of Gaussian noise layered on top before
  renormalizing back to the true hand size.

These are set in the training config YAML (see `obs_mode`, `belief_blend`,
`belief_noise` in `configs/ppo_baseline.yaml`) and are picked up
automatically by `scripts/train_self_play.py`. `scripts/evaluate_checkpoints.py`
reads `obs_mode` back out of each checkpoint's metadata (falling back to
`self_play` for older checkpoints), so mixed-mode evaluation — e.g. a
`realistic`-trained policy vs. a `global`-trained one via `--vs-ckpt` — just
works off what's stored on disk.

## Recording game traces

Any game can be recorded ply-by-ply (board state, hands, bank, dev deck,
dice, and the action taken) to a JSON trace for later replay:

- `scripts/train_self_play.py --trace N` — record every Nth game of each
  training iteration to `runs/<run_name>/traces/iter<k>_game<g>.json`
  (off by default).
- `scripts/evaluate_checkpoints.py --trace N [--trace-dir DIR]` — record
  every Nth evaluation game (`--trace 1` records all of them); traces land
  under `<run>/traces` (or `<ckpt>/../traces` when using `--ckpt` alone)
  unless `--trace-dir` overrides it.
- `scripts/render_match.py --trace out.json` — record a single rendered
  match to an explicit path.

## Replay dashboard

Browse recorded traces and scrub through them ply by ply in a browser:

```bash
python scripts/dashboard.py --port 8050 --runs-dir runs
# open http://127.0.0.1:8050
```

Pick a run, then a trace, to step through the match: an SVG board, each
player's hand/bank/dev-deck counts, the dice roll, and the action log all
update per ply as you scrub.

## Does it learn?

Verification run on a laptop CPU (config: `configs/ppo_baseline.yaml`,
`hidden_sizes=[512,512]`, 16 games/iteration): after **20 iterations
(~17 minutes)** the shared policy's win rate vs the random bot rose from
0.12 to **0.88**, and it beat its own 10-iterations-earlier checkpoint 75%
of head-to-head games. Beating the greedy/heuristic bots takes a longer run
(see the gcloud guide for cheap multi-hour training).

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
- **Rules coverage** — see [docs/RULES_AUDIT.md](docs/RULES_AUDIT.md) for a
  rule-by-rule audit of the engine against the official rulebook, with the
  test or commit backing each row.
- **Reproducibility** — board layout, dice, dev deck, and torch are all
  seeded; checkpoints carry config + metrics + architecture metadata.

## Training on Google Cloud

See [docs/RUN_ON_GCLOUD.md](docs/RUN_ON_GCLOUD.md).
