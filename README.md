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
│   ├── actions.py          fixed 512-slot action catalog (incl. P2P trade actions)
│   ├── action_mask.py      legal_action_mask(state) -> bool[512]
│   ├── observation.py      observation vectors (self_play/perfect/realistic/global)
│   ├── belief.py           BeliefTracker: per-opponent hand/dev-deck estimates (realistic mode)
│   ├── trace.py            game-trace recording format (for the replay dashboard)
│   ├── rules_profile.py    rules profiles (standard / simplified_v1 / trading variants / custom YAML)
│   ├── pettingzoo_env.py   AEC multi-agent env (CatanAECEnv)
│   └── gym_wrapper.py      single-agent Gym-style wrapper
├── bots/         scripted baselines: random, greedy, heuristic, trade personalities
├── rl/           training stack
│   ├── models.py           masked actor-critic MLP (always CATALOG_SIZE logits)
│   ├── rollout.py          4-seat rollout collection + per-seat GAE
│   ├── ppo.py              custom PPO (no stable-baselines3)
│   ├── self_play.py        training orchestrator + TensorBoard logging
│   ├── evaluate.py         eval vs bots / past checkpoints / cross-mode policy-vs-policy
│   └── checkpointing.py    state-dict checkpoints + JSON metadata
└── dashboard/    Flask backend + static frontend for browsing recorded traces

configs/          rules + PPO YAML configs
scripts/          CLI entry points
tests/            pytest suite (347 tests)
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
| `self_play` | 1548 | Own hand exact; opponents' hands are not exposed (public info only). |
| `perfect` | 1593 | `self_play` + every opponent's hand and dev cards exposed exactly (oracle view). |
| `realistic` | 1577 | `self_play` + a per-opponent *believed* hand (tracked from public actions, then blended toward uniform and Gaussian-noised) and its uncertainty, plus a believed dev-deck composition/count and the exact bank. |
| `global` | 1604 | `perfect` + the true remaining dev-deck composition/count and the exact bank. |

Every mode's vector ends with a fixed 28-float pending-trade block (active
flag, rotated proposer one-hot, give/get resource one-hots, give ratio, and
per-responder accept/decline/pending one-hots) — see "Player-to-player
trading" below. Because this block is appended after all mode-specific
content, every pre-trading observation vector is a strict prefix of its
current counterpart.

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

## Player-to-player trading

Bounded, templated P2P trading between the four seats (`catan_rl/env/actions.py`,
`catan_rl/env/rules.py`, `catan_rl/env/game_state.py`):

- **Action catalog v2 (512 slots)** — the original 256-slot catalog
  (`CATALOG_SIZE_V1`) is frozen at indices 0-255. Slots 256-295 are
  `PROPOSE_TRADE(give, get, give_n)` — the 20 ordered resource pairs
  (`give != get`) crossed with `give_n ∈ {1, 2}` (offer 1 or 2 of `give` for
  1 of `get`). Slot 296 is `ACCEPT_TRADE`, 297 is `DECLINE_TRADE`, and
  298-511 are reserved padding. The policy always outputs `CATALOG_SIZE`
  (512) logits, same masking scheme as v1.
- **`TRADE_RESPONSE` sub-phase state machine** — from `Phase.MAIN`, the
  current player may `PROPOSE_TRADE`. Responders who don't hold the
  requested resource are auto-declined; if anyone else is left pending, the
  game enters `Phase.TRADE_RESPONSE` and walks the remaining responders in
  seat order (starting at `(proposer + 1) % 4`), each choosing
  `ACCEPT_TRADE`/`DECLINE_TRADE`. Once every responder has answered, the
  **first accepter in seat order** executes the trade (proposer loses
  `give_n × give` / gains `1 × get`; that accepter loses `1 × get` / gains
  `give_n × give`) — the bank is never touched. Play then returns to
  `Phase.MAIN` with the proposer as current player.
- **Rules-profile gating** — `RulesProfile` gained `trades_enabled: bool`
  and `max_trades_per_turn: int` (default 3, reset every turn). Two builtin
  profiles turn trading on: `standard_trading` (full rules + trades) and
  `simplified_trading_v1` (dev cards off, trades on). All pre-existing
  profiles (`standard`, `simplified_v1`) default `trades_enabled=False` and
  are bit-identical to before — no trade slots ever become legal, and
  `Phase.TRADE_RESPONSE` is never reached.
- **Old-checkpoint compatibility** — because the pending-trade observation
  block is appended at the *absolute end* of every observation mode, every
  pre-trading observation vector is a strict prefix of its new counterpart.
  Policies trained before this feature (256 actions, shorter obs) keep
  working transparently against the new 512-action/longer-obs env: the
  loader routes action selection through `act_prefix_sliced`
  (`catan_rl/rl/models.py`), which slices the obs/mask down to the old
  policy's own dims. If the sliced legal-action mask is empty (e.g. an old
  256-head policy is asked to answer `Phase.TRADE_RESPONSE`), it returns
  `DECLINE_TRADE` — no retraining required, trade responses just
  auto-decline.

## Trade personalities

Five scripted trade-behavior presets (`catan_rl/bots/personalities.py`,
`PERSONALITIES` dict) wrap `heuristic_bot`: every non-trade decision
(placement, robber, discards, dev cards, builds, maritime trade, end turn)
is delegated verbatim to `heuristic_bot.pick_action`, so differences across
presets measure trade behavior alone. Each proposes/responds based on a
scarcity-plus-build-need value model over its own resources (see the module
docstring for the exact `trade_margin` formula).

| Preset | Behavior |
|---|---|
| `never_trader` | Never proposes and never accepts (`accept_margin=inf`) — a trade-free baseline opponent. |
| `opportunist` | Proposes often (`propose_rate=0.9`) and accepts even mildly unfavorable trades (`accept_margin=-0.5`), up to 3/turn. |
| `stall_the_leader` | Trades normally except it refuses any offer from a proposer within 2 VP of winning (or the sole current leader), to slow the frontrunner. |
| `fair_dealer` | Only accepts near-even trades — margin within a tight ±0.15 band — and proposes/accepts up to 2/turn. |
| `desperado` | Normally requires a decent margin (`0.3`) but that threshold loosens the further it falls behind the VP leader (`desperation_scale=0.25` per VP of deficit), so it gets more willing to accept lopsided trades when losing. |

`resolve_bot(name)` (`catan_rl/bots/__init__.py`) resolves any of
`"random"`, `"greedy"`, `"heuristic"`, or a personality preset name to a
`pick_action(state, rng)` callable — used by both training/eval code and
`scripts/evaluate_checkpoints.py --vs`.

## Opponent-pool training

`collect_rollouts` (`catan_rl/rl/rollout.py`) can train one seat against a
weighted mix of opponents instead of pure 4-seat self-play. Per game `g`,
the trained-policy seats rotate as `{(g + k) % 4 for k in
range(n_policy_seats)}`; every other seat draws independently (by weight,
seeded off the game seed) from the `opponents.pool` list, each entry one of:

- `{"type": "personality", "name": "<preset>", "weight": w}` — acts via
  `make_personality_bot`.
- `{"type": "checkpoint", "path": "<ckpt.pt>", "weight": w}` — acts
  deterministically from a saved checkpoint (old 256-action/short-obs
  checkpoints work here too, via `act_prefix_sliced`).
- `{"type": "self", "weight": w}` — acts with the live policy stochastically;
  its transitions are discarded (only policy-seat transitions train the
  model).

Only transitions from policy seats are collected. `configs/ppo_trading_pool.yaml`
is a working example (`rules_profile: standard_trading`, `n_policy_seats: 1`,
a pool of `opportunist`, `stall_the_leader`, and `self`):

```bash
python scripts/train_self_play.py --config configs/ppo_trading_pool.yaml
```

Per-opponent win rates are logged to TensorBoard as `game/win_rate_vs_<label>`
(and overall `game/policy_win_rate`), where `<label>` is `personality:<name>`,
`checkpoint:<stem>`, or `self`.

## Curriculum warm-start

`scripts/train_self_play.py --init-from PATH` warm-starts training from an
existing checkpoint, widening it to the current config's observation/action
dimensions if the checkpoint is smaller (e.g. bringing a pre-trading
256-action/1520-obs checkpoint into a 512-action/1548-obs trading run). The
widening (`widen_policy` in `catan_rl/rl/checkpointing.py`) is
prefix-preserving: existing weights are copied exactly (old inputs/outputs
behave identically), new observation input columns start at zero, and new
policy-head output rows start at zero weight with bias `-4.0` (near-zero
initial probability) so the new action space doesn't dominate untrained. The
optimizer always starts fresh. `--init-from` requires matching
`hidden_sizes` between the old checkpoint and the new config.

```bash
python scripts/train_self_play.py --config configs/ppo_trading_pool.yaml \
    --init-from runs/ppo_baseline/checkpoints/ckpt_000500.pt
```

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

- **Action space** — a fixed 512-slot catalog (`actions.py`); every
  `(action_type, parameter)` pair has a permanent index, and the original
  256-slot v1 catalog is a frozen prefix of it. The policy always outputs
  `CATALOG_SIZE` logits; the environment masks illegal slots. Sub-phases
  (knight → move robber, road building → two road placements, propose →
  `TRADE_RESPONSE`) reuse the same slots with a different mask.
- **Shared-policy self-play** — all four seats run the same network; each
  seat's transitions are collected in its own list and GAE is computed
  per seat (seats' turns interleave, so a naive sequence-wide GAE would be
  wrong). Winner gets +1, losers −1 (configurable). Opponent-pool training
  (see above) can instead train only designated policy seats against a
  mix of personality bots, checkpoints, and self.
- **Rules profiles** — `simplified_v1` disables dev cards for the first
  trainable curriculum stage; the engine supports full standard rules
  (dev cards, largest army, robber, ports, maritime trade). `standard_trading`
  and `simplified_trading_v1` additionally enable player-to-player trading
  (see "Player-to-player trading" above).
- **Rules coverage** — see [docs/RULES_AUDIT.md](docs/RULES_AUDIT.md) for a
  rule-by-rule audit of the engine against the official rulebook, with the
  test or commit backing each row.
- **Reproducibility** — board layout, dice, dev deck, and torch are all
  seeded; checkpoints carry config + metrics + architecture metadata.

## Training on Google Cloud

See [docs/RUN_ON_GCLOUD.md](docs/RUN_ON_GCLOUD.md).
