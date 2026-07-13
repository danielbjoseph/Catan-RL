# Package B: P2P Trading + Personality Opponents — Design

**Status:** designed 2026-07-12; build after Package A
(`2026-07-12-obs-modes-replay-audit-design.md`)

## Goals

1. Bounded structured player-to-player trading (spec Outline.md Phase 5D) —
   templated offers, accept/decline, no free-form negotiation.
2. Preset trade personalities modeling real-world player archetypes.
3. Opponent-pool training: train the shared policy against chosen
   personalities (specialists) or mixtures (generalist), with per-personality
   evaluation.

Core RL principle: the *policy's* trade behavior is never hand-coded — it is
conditioned on hand, VP standings, and board state, so desperation- and
status-dependent trading emerges from training. Hand-coded tendencies exist
only in the scripted personality opponents.

---

## 1. Trading system

### 1.1 Action catalog v2 (512 slots)

Indices 0–255 identical to v1 (old checkpoints stay loadable; arch metadata
already records `n_actions`). New slots from 256:

| Action | Slots | Notes |
|---|---|---|
| `PROPOSE_TRADE(give, get, ratio)` | 40 | 20 ordered pairs × ratio ∈ {1:1, 2:1} |
| `ACCEPT_TRADE` | 1 | |
| `DECLINE_TRADE` | 1 | |
| reserved padding | 214 | future: choose-partner, counter-offers |

`CATALOG_VERSION = 2`; `actions.py` exposes both sizes; policy head width
comes from the catalog version in the run config.

### 1.2 Trade sub-phase state machine

New `GameState` fields: `pending_trade = {proposer, give, get, ratio,
responses: {pid: bool}}`, `trades_proposed_this_turn: int`.

```
MAIN --PROPOSE_TRADE--> TRADE_RESPONSE
  TRADE_RESPONSE: each other player, in seat order, becomes current actor
                  with mask = {ACCEPT_TRADE, DECLINE_TRADE}
                  (auto-declined without acting if they lack the "get" resources)
  all responded:
    >=1 accept  -> execute with the FIRST accepter in seat order (v1
                   simplification; a choose-partner sub-phase is reserved
                   for later), return to MAIN
    all decline -> return to MAIN
```

Guards: proposer must hold the offered resources; offers capped at
`max_trades_per_turn` (default 3, in RulesProfile); no trades before rolling;
trading with the bank remains the separate maritime action.

`RulesProfile` gains `trades_enabled: bool` (false for `simplified_v1` and
`standard` to preserve existing behavior; new profiles `standard_trading` /
`simplified_trading_v1`).

### 1.3 Observation additions (all modes)

Pending-trade context block: proposer one-hot (4), give/get one-hots (5+5),
ratio (1), responses-so-far (4×3 states). BeliefTracker treats executed
trades as fully public transfers (exact reconciliation of both hands).

---

## 2. Trade personalities

### 2.1 Parameterization (`catan_rl/bots/personalities.py`)

```python
@dataclass(frozen=True)
class TradePersonality:
    name: str
    propose_rate: float        # P(attempt a beneficial proposal) per MAIN visit
    accept_margin: float       # required pip-value gain to accept (can be negative)
    leader_block_vp: int | None  # refuse trades with anyone within this many VP of winning
    desperation_scale: float   # accept_margin loosens by this per VP the bot trails the leader
    max_proposals_per_turn: int
```

Value model: a trade's worth = (pip-scarcity-weighted value of what you get)
− (value of what you give), scarcity from the bot's own production spread —
reuses `_common.py` pip machinery.

### 2.2 Presets

| Preset | Behavior |
|---|---|
| `never_trader` | declines everything, never proposes |
| `opportunist` | proposes constantly; accepts any trade with positive build progress, ignores who benefits |
| `stall_the_leader` | normal trading, but refuses all trades with the VP leader / anyone within 2 VP of winning; prefers partners furthest behind |
| `fair_dealer` | accepts only near-even value (|margin| ≤ small ε); proposes evenly |
| `desperado` | margin loosens sharply when trailing or when one resource short of a build |

Each preset wraps the heuristic bot: non-trade decisions are identical, so
differences in outcomes measure trade behavior alone.

### 2.3 Tests

Per preset: property tests on decision rules (never_trader never
accepts/proposes; stall_the_leader declines exactly the leader-adjacent
offers; desperado's threshold moves with VP deficit); full games with 4
mixed personalities terminate; trade executions conserve resources.

---

## 3. Opponent-pool training + evaluation

### 3.1 Rollout changes

`collect_rollouts(..., opponents=None)` — `None` keeps pure 4-seat self-play.
Otherwise a per-game sampler assigns seats: the learning policy takes
`n_policy_seats` (default 1) rotating seats; remaining seats draw from the
pool spec:

```yaml
opponents:
  pool:
    - {type: personality, name: opportunist, weight: 2}
    - {type: personality, name: stall_the_leader, weight: 1}
    - {type: checkpoint, path: runs/x/checkpoints/ckpt_000100.pt, weight: 1}
    - {type: self, weight: 2}          # current policy (self-play seat)
```

Transitions are collected **only from policy-controlled seats** (seat_ids
already in the Batch make this a filter). Stats gain per-opponent-type win
rates.

### 3.2 Training modes

- **Specialist**: pool = 3× one personality → "learn to beat the
  opportunist"; one run per archetype (config-only difference).
- **Generalist**: mixed pool incl. self-play seats and checkpoint history —
  the robust policy.

### 3.3 Evaluation

`evaluate_vs_bots` generalizes to any personality; TB scalars
`eval/win_rate_vs_<preset>` for every preset each eval interval, so training
dashboards show exactly which trade styles the policy exploits or struggles
against. `evaluate_checkpoints.py --vs never_trader,opportunist,...`.

### 3.4 Curriculum note

Recommended path: start from a Package-A-trained no-trade policy, enable
`trades_enabled`, fine-tune with the wider 512 head (first 256 logits
warm-started from the old head, new slots initialized near zero).

---

## Open items to settle at build time

- Exact pip-value function for the personality value model (calibrate so
  fair_dealer ≈ break-even on 1:1 same-scarcity trades).
- Whether TRADE_RESPONSE seats should also see the proposer's *believed*
  hand (realistic mode) — leaning yes, it's already in the observation.
- 2:1 template direction (give 2 identical for 1) — confirm no 1:2 needed
  (symmetric via the other side proposing).
