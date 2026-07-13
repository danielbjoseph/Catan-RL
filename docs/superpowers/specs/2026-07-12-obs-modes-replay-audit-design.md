# Package A: Observation Modes, Replay Dashboard, Rules Audit — Design

**Status:** approved design (brainstormed 2026-07-12)
**Branch:** `obs-modes-and-replay`
**Depends on:** v0.1 system (PR #1, merged)
**Followed by:** Package B — P2P trading + personality opponents (`2026-07-12-trading-personalities-design.md`)

## Goals

1. Two new observation modes so two policies can be trained and compared:
   **`global`** (omniscient) and **`realistic`** (human-like knowledge with
   tracked-but-noisy beliefs).
2. Opt-in game trace recording everywhere games are played.
3. A local Flask replay dashboard: scrub through any recorded game and see the
   board, hands, dev deck, bank, dice history at any ply.
4. A systematic rules audit against the official base-game rulebook, with
   fixes + regression tests for every inaccuracy found.

Non-goals (deferred to Package B): player-to-player trading, personality
opponents, opponent-pool training.

---

## 1. Belief tracking + observation modes

### 1.1 BeliefTracker (`catan_rl/env/belief.py`)

One tracker per game, maintained from the "table view" (public events only —
what any attentive player at the table could know). Pure engine-side module:
no torch, unit-testable with hand-constructed event sequences.

State per player `i`:
- `expected[i]`: float vector (5,) — believed resource counts by type
- `uncertainty[i]`: float — total probability mass moved by hidden events
  (steals received/lost, hidden discards), normalized by hand size

Event handling (hooked via a single `tracker.on_action(state_before, action,
state_after)` call inside `CatanAECEnv.step` and the eval/rollout game loops —
`rules.py` itself stays tracker-free):

| Event | Public info | Tracker update |
|---|---|---|
| Dice production | fully public | add exact gains to `expected` |
| Build road/settlement/city, buy dev | cost public | subtract exact cost |
| Maritime trade | fully public | apply exactly |
| Year of plenty | gains public | add exactly |
| Monopoly | fully public reveal | set victim type counts to 0, add to player; also *reconciles* that type exactly |
| Discard (on 7) | count public, composition hidden | subtract proportionally from `expected`, raise `uncertainty` |
| Robber steal | card hidden; only count public | move `expected/total`-weighted mass victim→thief, raise both `uncertainty` |
| Dev card play | card type revealed | update dev-deck estimate (below) |

Invariant maintained: `sum(expected[i]) == players[i].total_resources`
(hand sizes are always public), clipped at ≥ 0.

Dev deck estimate: remaining count is public. Believed composition =
initial (14/2/2/2/5) − publicly played cards − (for the observer) their own
held/new cards; normalized to remaining count. `global` mode instead gets the
exact multiset of the remaining deck.

Simplification (documented): steal participants (thief/victim) actually know
the stolen card; v1 uses table-view beliefs for all observers. Acceptable
because the observer's *own* hand is always exact in the observation — only
the thief's extra knowledge about a victim is lost.

### 1.2 Noise layer ("casual player, not card counter")

Applied only at observation time in `realistic` mode; the tracker itself stays
exact. Two knobs, config-driven (in the training YAML, not RulesProfile):

- `belief_blend` β (default 0.25): observed vector =
  `(1-β)·expected[i] + β·uniform_prior(hand_size)`
- `belief_noise` σ (default 0.5): add seeded Gaussian noise (σ scaled by
  `sqrt(hand_size)`), then re-normalize to hand size, clip ≥ 0.
  Noise is deterministic per (game_seed, turn, observer) for reproducibility.

β=0, σ=0 gives a perfect-memory card counter; defaults give a fuzzy but
grounded picture.

### 1.3 Observation modes

`make_observation(state, observer, mode, belief=None, noise_cfg=None)`:

| Mode | Base | Additions | Dim |
|---|---|---|---|
| `self_play` | existing 1520 | — (unchanged, back-compat) | 1520 |
| `perfect` | existing 1565 | — (unchanged, back-compat) | 1565 |
| `realistic` | self_play 1520 | 3 opponents × (5 belief + 1 uncertainty) = 18; dev-deck estimate 5 + count 1 = 6; bank 5 | **1549** |
| `global` | perfect 1565 | exact remaining dev-deck composition 5 + count 1; bank 5 | **1576** |

All new features normalized like existing segments (resources /19, dev /14).

### 1.4 Plumbing

- `CatanAECEnv(obs_mode=...)` accepts the two new modes; owns the
  `BeliefTracker` lifecycle (reset per game, `on_action` per step) only when
  `obs_mode == "realistic"`.
- `checkpointing.save_checkpoint` metadata gains `obs_mode`;
  `load_policy` returns it; `evaluate.policy_action` takes the mode from the
  checkpoint so policies with different modes/dims can sit at the same table.
- New eval entry point `evaluate_policy_vs_policy(policy_a, mode_a, policy_b,
  mode_b, n_games, ...)` (2 seats each, rotating) + script flag
  `evaluate_checkpoints.py --vs-ckpt <path>` — this is the
  "how much is omniscience worth?" experiment.
- `train_self_play.py` config: `obs_mode: global|realistic|self_play|perfect`,
  plus `belief_blend`, `belief_noise`.

### 1.5 Tests

- Tracker: production/build events reconcile exactly; steal moves proportional
  mass and raises uncertainty; monopoly reconciles a type; sum invariant holds
  over a full random game (property test vs. true hands: expected never
  exceeds bounds, exact when no hidden events occurred).
- Obs: dims per mode; realistic belief features match tracker output ± noise;
  global mode dev-deck features match the actual deck multiset.
- Determinism: same seed → identical realistic observations.
- Tiny end-to-end training run in each new mode.

---

## 2. Trace recording

### 2.1 Format (`catan_rl/env/trace.py`)

One JSON file per game: `{"header": ..., "plies": [...]}`.

- Header: schema version, seed(s), profile, seat labels (bot/policy names),
  board (hex resources/tokens/desert/ports/robber start) **and drawing
  geometry** (hex centers, vertex xy positions, edge endpoints) so the
  frontend never re-derives topology.
- Per ply: `{ply, turn, player, phase, action_index, action_str, dice,
  state}` where `state = GameState.to_dict()` **after** the action
  (full snapshot every ply — simple dashboard logic; ~2–5 MB per game is
  fine for opt-in traces).

`TraceRecorder` API: `start(state, meta)`, `record(action, state)`,
`save(path)`. Used by rollout collection, eval games, and render_match.

### 2.2 Recording flags

- `train_self_play.py --trace N` → every Nth self-play game per iteration →
  `runs/<name>/traces/iter<k>_game<g>.json`
- `evaluate_checkpoints.py --trace` → all eval games →
  `runs/<name>/traces/eval_<ckpt>_<opp>_<i>.json`
- `render_match.py --trace out.json`
- Default off everywhere; zero overhead when off.

### 2.3 Tests

Round-trip: record a seeded game → final snapshot equals live final
`state.to_dict()`; every ply's action was legal in the prior snapshot;
JSON serializable/loadable.

---

## 3. Replay dashboard (Flask)

### 3.1 Structure

```
catan_rl/dashboard/
├── app.py          Flask app factory
└── static/
    ├── index.html  run browser + replay view (single page)
    ├── app.js      SVG board renderer + scrubber + panels
    └── style.css
scripts/dashboard.py   entry point: python scripts/dashboard.py [--port 8050] [--runs-dir runs]
```

`flask` added to requirements.txt.

### 3.2 Endpoints

- `GET /` → index.html
- `GET /api/runs` → list of run dirs under `--runs-dir` that contain traces
- `GET /api/traces/<run>` → trace filenames + header summaries
- `GET /api/trace/<run>/<file>` → full trace JSON
  (paths validated: resolved path must stay under the runs dir)

### 3.3 Replay view

- SVG board from header geometry: hexes colored by resource with number
  tokens, robber marker, ports; roads/settlements/cities drawn per ply in
  seat colors.
- Scrubber: ply slider + ←/→ keys (ply) and ↑/↓ (whole turn), play button.
- Panels, all live at the selected ply: per-player resources by type, dev
  cards (hand / bought-this-turn / played), VP breakdown (public + hidden +
  road/army awards), piece counts; bank; dev deck remaining; dice-roll
  history strip (2–12 histogram + sequence); scrolling action log with the
  current ply highlighted.

### 3.4 Tests

Endpoint tests with Flask's test client (runs list, trace fetch, path
traversal rejected). Frontend is exercised manually via a recorded sample
trace (checked against known game facts from the trace test).

---

## 4. Rules audit

### 4.1 Method

Walk the official base-game rulebook (5–6th edition, 3–4 players) section by
section against `rules.py` / `validators.py` / `scoring.py` /
`game_state.py`. Produce `docs/RULES_AUDIT.md`: one row per rule —
**correct** (with the test that proves it), **fixed** (with commit + test), or
**intentionally simplified** (with rationale).

### 4.2 Known deviations to fix (found during design)

1. **Bank shortage on production**: official — if the bank cannot fully supply
   *all* players for a resource type on a roll, *no player* receives that
   type that roll (unless only one player is affected, who takes what's
   left). Engine currently pays first-come-first-served in vertex order.
2. **VP dev cards**: official — never "played"; they count automatically
   toward victory. Fix: remove `PLAY_VICTORY_POINT` from legal actions
   (catalog slot 253 becomes permanently masked/reserved), include held VP
   cards in `total_vp` for the winner check only (still hidden from public
   VP and opponents' observations).
3. **Dev card before roll**: official — you may play one dev card at any time
   during your turn, *including before rolling* (knight-before-roll matters).
   Fix: in `ROLL` phase, mask additionally offers playable dev cards
   (not bought this turn); playing a knight pre-roll runs robber/steal then
   returns to `ROLL`.

Anything else found during the systematic pass gets the same treatment.
Each fix lands with a regression test; existing 174 tests must stay green
(tests that encoded wrong behavior get corrected with a note in the audit).

### 4.3 Intentional simplifications (documented, not "fixed")

- No player-to-player trading (Package B).
- Discards resolve one card per action (interface simplification; net effect
  identical).
- No 5–6 player extension, no Cities & Knights.

---

## Delivery

Single PR from `obs-modes-and-replay`. Suggested build order: rules audit
fixes first (engine correctness before building on it), then belief/obs
modes, then traces, then dashboard. Tests-first throughout; tiny training
runs in both new modes as the final acceptance gate, plus one recorded trace
replayed in the dashboard end to end.
