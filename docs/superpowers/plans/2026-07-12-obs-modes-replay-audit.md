# Package A: Obs Modes, Replay Dashboard, Rules Audit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement spec `docs/superpowers/specs/2026-07-12-obs-modes-replay-audit-design.md`: rules-accuracy fixes + audit doc, BeliefTracker with `realistic`/`global` observation modes, opt-in game traces, and a Flask replay dashboard.

**Architecture:** Engine fixes land first (audit), then a table-view `BeliefTracker` in the env layer feeds two new observation modes; traces are full-state-per-ply JSON recorded by a `TraceRecorder` hooked into every game loop; the dashboard is a small Flask app serving a static SVG replay UI over trace JSON.

**Tech Stack:** Existing stack + Flask. Tests with pytest; browser verification of the dashboard via Playwright MCP.

## Global Constraints

- All installs into `.venv` (`.venv/Scripts/python.exe -m pip ...`).
- Back-compat: `self_play` (1520) and `perfect` (1565) modes unchanged; existing checkpoints must keep loading.
- New dims: `OBS_DIM_REALISTIC = 1549`, `OBS_DIM_GLOBAL = 1576`.
- Noise defaults: `belief_blend = 0.25`, `belief_noise = 0.5`; noise deterministic per (game_seed, turn, observer).
- Traces: full `state.to_dict()` snapshot **per ply**; recording strictly opt-in.
- Rules fixes must not silently change `simplified_v1` training semantics beyond correctness (dev-card fixes only affect `standard`).
- Same-session author/executor: plan pins exact interfaces, key algorithms, and test intent; boilerplate bodies are written at execution time.

---

### Task 1: Rules fix — bank shortage on production

**Files:** Modify `catan_rl/env/rules.py` (`_produce_resources`); Test `tests/test_rules_audit.py` (new file, grows through Tasks 1–4)

**Rule:** If the bank cannot fully supply **all** players for a resource type on a roll, **no player** receives that type (unless exactly one player is owed it — they take what's left).

- [ ] Failing tests: (a) two players owed wood, bank has 1 → both get 0 and bank unchanged; (b) one player owed 3, bank has 2 → gets 2, bank 0; (c) supply sufficient → unchanged behavior. Build states by hand (place settlements adjacent to a known-token hex, set `state.bank`).
- [ ] Implement two-pass production:

```python
def _produce_resources(state, number):
    geo, config = state.config.geometry, state.config
    occupied = state.all_occupied_vertices()
    owed = [[0] * 5 for _ in range(state.n_players)]   # pid -> per-resource counts
    for hex_id in range(geo.n_hexes):
        if config.hex_tokens[hex_id] != number or hex_id == state.robber_hex:
            continue
        hex_type = config.hex_resources[hex_id]
        if hex_type == HexType.DESERT:
            continue
        r = int(hex_type.to_resource())
        for v in geo.hex_to_vertices[hex_id]:
            pid = occupied.get(v)
            if pid is None:
                continue
            owed[pid][r] += 2 if v in state.players[pid].city_vertices else 1
    for r in range(5):
        demanders = [pid for pid in range(state.n_players) if owed[pid][r] > 0]
        total = sum(owed[pid][r] for pid in demanders)
        supply = state.bank[r]
        if not demanders:
            continue
        if total <= supply:
            for pid in demanders:
                state.players[pid].resources[r] += owed[pid][r]
            state.bank[r] -= total
        elif len(demanders) == 1:
            pid = demanders[0]
            state.players[pid].resources[r] += supply
            state.bank[r] = 0
        # else: shortage with multiple demanders -> nobody paid
```

- [ ] Full suite green (existing production tests may need updating if they relied on first-come-first-served — correct them, note in audit) → commit `fix(rules): official bank-shortage rule for production`

### Task 2: Rules fix — VP dev cards auto-count

**Files:** Modify `catan_rl/env/player_state.py` (`hidden_vp`), `catan_rl/env/validators.py` (drop `PLAY_VICTORY_POINT` from `_main_actions`); Test `tests/test_rules_audit.py`

**Rule:** VP cards are never played; they count automatically toward victory (still hidden from public VP).

- [ ] Failing tests: player with 9 public VP buys/holds a VP card → winner detected on next win-check; slot 253 never legal in any phase of a random standard game (200 plies); `public_vp` excludes VP cards.
- [ ] Implement: `hidden_vp = dev_cards[VP] + dev_cards_new[VP] + played_dev_cards[VP]`; remove the `PLAY_VICTORY_POINT` block from `_main_actions` (keep the `apply_action` handler for old traces; slot stays in catalog, permanently masked).
- [ ] Update any existing tests that played VP cards → suite green → commit `fix(rules): VP dev cards count automatically, never played`

### Task 3: Rules fix — dev card playable before rolling

**Files:** Modify `catan_rl/env/game_state.py` (add `rolled_this_turn: bool = False` + clone/serde), `catan_rl/env/rules.py`, `catan_rl/env/validators.py`; Test `tests/test_rules_audit.py`

**Rule:** One dev card may be played at any time during your turn, including before the roll (knight-before-roll unblocks your own hexes).

- [ ] Failing tests: in `ROLL` phase a player holding an eligible knight sees `PLAY_KNIGHT` legal; playing it pre-roll runs ROBBER→STEAL and returns to `ROLL` (must still roll); post-roll play returns to `MAIN`; cards bought this turn still unplayable; only one dev card per turn across both windows.
- [ ] Implement: set `rolled_this_turn = True` in `_roll_dice`, `False` in `_end_turn`. `_main_return_phase(state) = Phase.MAIN if state.rolled_this_turn else Phase.ROLL` used by `_move_robber` (no-target branch), `_steal`, road-building termination paths. Validators: `Phase.ROLL` → `[ROLL_DICE] + dev-card plays` (same gating as MAIN: profile enabled, not new, one per turn).
- [ ] Suite green → commit `fix(rules): allow one dev card before the roll`

### Task 4: Rules audit — systematic pass + two more fixes + doc

**Files:** Create `docs/RULES_AUDIT.md`; Modify `catan_rl/env/validators.py` (`_steal_actions`), `catan_rl/env/scoring.py` (`update_longest_road`); Test `tests/test_rules_audit.py`

Known items the pass must include (plus anything else found):

- **Steal targets must hold cards:** `_steal_actions` filters to players with `total_resources > 0`; if none, no steal (mirror the no-target branch in `_move_robber` — recheck there too).
- **Longest road revocation:** official — recompute on every road/settlement change; holder keeps only while still ≥ 5 and not strictly beaten; if holder drops below 5: award to the unique ≥ 5 maximum, else nobody. Replace `update_longest_road` body:

```python
def update_longest_road(state):
    lengths = [compute_longest_road(pid, state) for pid in range(state.n_players)]
    holder = state.longest_road_holder
    if holder is not None:
        for pid, ln in enumerate(lengths):
            if pid != holder and ln > lengths[holder] and ln >= LONGEST_ROAD_MIN:
                state.longest_road_holder = pid
                return
        if lengths[holder] < LONGEST_ROAD_MIN:
            eligible = [ln for ln in lengths if ln >= LONGEST_ROAD_MIN]
            if eligible:
                best = max(eligible)
                cands = [pid for pid, ln in enumerate(lengths) if ln == best]
                state.longest_road_holder = cands[0] if len(cands) == 1 else None
            else:
                state.longest_road_holder = None
    else:
        best = max(lengths)
        if best >= LONGEST_ROAD_MIN:
            cands = [pid for pid, ln in enumerate(lengths) if ln == best]
            if len(cands) == 1:
                state.longest_road_holder = cands[0]
```

  (Note: settlement builds already call `update_longest_road`; verify a settlement that splits an opponent's road triggers revocation — test with a hand-built board.)
- [ ] Failing tests for both fixes (steal-empty-target masked; road split below 5 revokes card; tie after split → nobody).
- [ ] Write `docs/RULES_AUDIT.md`: table of every base-game rule (setup, production, robber, building, dev cards, awards, win) × status (**correct** w/ test reference | **fixed** w/ commit | **simplified** w/ rationale — incl. one-card-per-action discards, no P2P trade until Package B, win-check scans all players harmlessly).
- [ ] Suite green → commit `fix(rules): steal-target and longest-road revocation rules + audit doc`

### Task 5: BeliefTracker

**Files:** Create `catan_rl/env/belief.py`; Test `tests/test_belief.py`

**Interfaces (consumed by Tasks 6–7):**

```python
class BeliefTracker:
    def __init__(self, state: GameState): ...        # captures initial hands (all 0)
    def reset(self, state: GameState) -> None
    def on_action(self, state_before: GameState, action: Action, state_after: GameState) -> None
    def expected(self, pid: int) -> np.ndarray        # (5,) float32, sums to hand size
    def uncertainty(self, pid: int) -> float          # hidden-event mass / max(hand,1)
    def dev_deck_estimate(self, observer: int, state: GameState) -> tuple[np.ndarray, int]
        # ((5,) believed remaining composition scaled to count, count)
```

Update dispatch (spec §1.1 table): public-delta events (production, builds, buys, maritime, YoP, monopoly) apply the exact per-player resource delta `after − before`; `DISCARD_RESOURCE` subtracts proportionally from `expected[pid]` and adds the discarded amount to `uncertainty`; `CHOOSE_STEAL_TARGET` moves `expected[victim]/total(victim)` mass to the thief and raises both uncertainties; monopoly zeroes the type for victims (exact reconciliation). Always renormalize `expected[pid]` to the true public hand size, clip ≥ 0.

- [ ] Failing tests: production/build reconcile exactly; steal moves proportional mass + uncertainty; monopoly reconciles; discard proportional; **property test**: full seeded random standard game, tracker fed every ply → `sum(expected[i]) == hand size` always, and `expected == true hand` exactly whenever `uncertainty == 0`.
- [ ] Implement → green → commit `feat(env): table-view belief tracker`

### Task 6: Realistic + global observation modes

**Files:** Modify `catan_rl/env/observation.py`; Test `tests/test_observation_modes.py`

**Interfaces:**

```python
OBS_DIM_REALISTIC = 1549   # 1520 + 3*(5+1) + (5+1) + 5
OBS_DIM_GLOBAL    = 1576   # 1565 + (5+1) + 5
def obs_dim_for_mode(mode: str) -> int
def make_observation(state, observer, mode="self_play",
                     belief: BeliefTracker | None = None,
                     noise_cfg: dict | None = None) -> np.ndarray
    # noise_cfg = {"belief_blend": float, "belief_noise": float, "seed": int}
def apply_belief_noise(vec, hand_size, blend, sigma, key: tuple) -> np.ndarray
    # key = (seed, turn, observer); np.random.default_rng(hash) — deterministic
```

Feature blocks (normalization: resources /19, dev /14, deck count /25, bank /19):
- realistic (needs `belief`): per opponent in rotated order: noised `expected/19` (5) + `uncertainty` (1); dev-deck believed composition /14 (5) + count /25 (1); bank /19 (5).
- global: exact remaining `dev_deck` composition /14 (5) + count /25 (1); bank /19 (5). No tracker needed.

- [ ] Failing tests: dims per mode; `self_play`/`perfect` byte-identical to before (regression: fixed seed, compare to stored reference vector); realistic features equal tracker output when blend=σ=0; noise deterministic per key and changes with observer; global deck features match actual `state.dev_deck` multiset.
- [ ] Implement → green → commit `feat(env): realistic and global observation modes`

### Task 7: Env/trainer/eval plumbing for modes

**Files:** Modify `catan_rl/env/pettingzoo_env.py`, `catan_rl/env/gym_wrapper.py`, `catan_rl/rl/checkpointing.py`, `catan_rl/rl/evaluate.py`, `catan_rl/rl/self_play.py`, `configs/ppo_baseline.yaml` (comment the new keys); Test `tests/test_observation_modes.py`, `tests/test_self_play.py`

**Interfaces:**
- `CatanAECEnv(obs_mode=..., belief_blend=0.25, belief_noise=0.5)`: accepts 4 modes; owns tracker when `realistic` (create on `reset`, `on_action` each `step`), passes `belief`/`noise_cfg` to `make_observation`.
- `save_checkpoint(..., obs_mode: str = "self_play")` → top-level metadata key; `load_policy` returns metadata (mode read by callers).
- `evaluate.policy_action(policy, state, obs_mode=..., belief=None, noise_cfg=None)`; `_play_eval_game` maintains one shared table-view tracker when any seat's actor needs it; `evaluate_vs_bots/vs_checkpoint` gain `obs_mode` (vs_checkpoint uses each policy's own stored mode).
- `SelfPlayTrainer`: `obs_mode`, `belief_blend`, `belief_noise` config keys; obs_dim via `obs_dim_for_mode`; passes mode to eval + checkpoints. `collect_rollouts` passes env kwargs through.

- [ ] Failing tests: AEC env in each new mode runs 200 random plies with correct obs dims and no dev-slot leakage differences; checkpoint round-trip preserves `obs_mode`; tiny end-to-end training run (2 iters, 2 games, hidden 32×32) in `realistic` and in `global` completes with TB scalars present (parametrize the existing tiny-run test).
- [ ] Implement → green → commit `feat(rl): train/eval plumbing for realistic and global modes`

### Task 8: Cross-mode policy-vs-policy evaluation

**Files:** Modify `catan_rl/rl/evaluate.py`, `scripts/evaluate_checkpoints.py`; Test `tests/test_evaluate.py`

**Interfaces:**

```python
def evaluate_policy_vs_policy(policy_a, mode_a, policy_b, mode_b, n_games=20, *,
                              rules_profile=..., seed=0, max_turns=500,
                              noise_cfg=None, device="cpu") -> dict
    # 2 seats each, rotating; returns {"win_rate_a", "win_rate_b", "draws", "games"}
```

Script: `evaluate_checkpoints.py --ckpt A.pt --vs-ckpt B.pt` reads each checkpoint's stored `obs_mode` — this is the omniscience-value experiment.

- [ ] Failing test: two random-init policies with different modes (`global` vs `realistic`) complete 2 fast-profile games; win rates sum ≤ 1.
- [ ] Implement → green → commit `feat(rl): cross-mode policy-vs-policy evaluation`

### Task 9: Board drawing geometry + trace format

**Files:** Modify `catan_rl/env/board.py` (BoardGeometry gains `vertex_positions: Tuple[Tuple[float, float], ...]` and `hex_centers: Tuple[Tuple[float, float], ...]`, captured in `build()` from the already-computed position maps); Create `catan_rl/env/trace.py`; Test `tests/test_trace.py`

**Interfaces:**

```python
class TraceRecorder:
    def __init__(self): ...
    def start(self, state: GameState, meta: dict) -> None   # meta: seeds, seat labels, notes
    def record(self, action: Action, state: GameState) -> None  # state AFTER action
    def to_dict(self) -> dict
    def save(self, path: str | Path) -> Path
```

Trace JSON: `{"version": 1, "header": {meta, profile, board: {hex_resources, hex_tokens, desert_hex, ports: [{vertices, resource}], robber_start}, geometry: {vertex_positions, hex_centers, edge_to_vertices, hex_to_vertices}}, "plies": [{ply, turn, player, phase, action_index, action_str, dice, state}]}`.

- [ ] Failing tests: geometry arrays have 54/19 entries and match adjacency (each hex's 6 vertices average to its center within tolerance); record a full seeded greedy game → last ply's `state` equals live `state.to_dict()`; every recorded `action_index` was legal in the preceding snapshot (rebuild via `GameState.from_dict`); file saves/loads as valid JSON.
- [ ] Implement → green → commit `feat(env): game trace recording + drawing geometry`

### Task 10: Trace flags in all game loops

**Files:** Modify `catan_rl/rl/rollout.py` (`collect_rollouts(..., trace_dir=None, trace_every=None)`), `catan_rl/rl/self_play.py` + `scripts/train_self_play.py` (`--trace N` → `runs/<name>/traces/iter<k>_game<g>.json`), `catan_rl/rl/evaluate.py` + `scripts/evaluate_checkpoints.py` (`--trace`), `scripts/render_match.py` (`--trace out.json`); Test `tests/test_trace.py`

- [ ] Failing tests: `collect_rollouts(n_games=2, trace_dir=tmp, trace_every=1)` writes 2 loadable traces whose final snapshots are terminal/truncated states; `trace_every=None` writes nothing.
- [ ] Implement (recorder created per traced game; zero code path when off) → smoke-run `render_match.py --trace` → green → commit `feat: opt-in trace recording in train/eval/render`

### Task 11: Dashboard backend (Flask)

**Files:** Create `catan_rl/dashboard/__init__.py`, `catan_rl/dashboard/app.py`; Create `scripts/dashboard.py`; Modify `requirements.txt` (+`flask>=3.0`); Test `tests/test_dashboard.py`

**Interfaces:**

```python
def create_app(runs_dir: str | Path) -> Flask
# GET /                      -> static/index.html
# GET /api/runs              -> [{"run": name, "n_traces": int}]
# GET /api/traces/<run>      -> [{"file": name, "turns": int, "winner": int|None, "seats": [...]}]
# GET /api/trace/<run>/<file> -> full trace JSON
# 404 for missing; 400 for paths escaping runs_dir (resolve + is_relative_to check)
```

`scripts/dashboard.py --port 8050 --runs-dir runs` → `app.run`.

- [ ] Failing tests (Flask test client, tmp runs dir with a real recorded trace): runs list, trace list summaries, trace fetch equals file, `../` traversal → 400.
- [ ] Install flask into `.venv` → implement → green → commit `feat(dashboard): flask backend for trace browsing`

### Task 12: Dashboard frontend

**Files:** Create `catan_rl/dashboard/static/index.html`, `static/app.js`, `static/style.css`

Single page, two views. Run browser: run list → trace list → open. Replay view:
- SVG board from header geometry (viewBox fit to vertex extents): hex polygons colored by resource (wood/brick/sheep/wheat/ore/desert palette), number tokens (6/8 highlighted), robber marker, port markers at port vertices; roads as thick edge lines, settlements as squares, cities as larger pentagons, in 4 fixed seat colors.
- Controls: ply slider, ◀/▶ step (arrow keys), ⏮/⏭ turn jump (up/down), play/pause at ~5 plies/s, ply/turn/phase readout.
- Panels bound to the selected ply's `state`: per-player card table (5 resources + dev held/new/played + VP breakdown incl. hidden VP and award badges), bank row, dev-deck remaining count, dice history (sequence strip + 2–12 histogram up to current ply), scrolling action log (click a row → jump to ply).
- [ ] Implement (no build step, vanilla JS; theme via CSS variables).
- [ ] **Browser verification** (Playwright MCP): record a fresh greedy-vs-mixed trace, start `scripts/dashboard.py` in background, navigate, assert board SVG renders 19 hexes, scrub to final ply, cross-check one player's resource panel against the trace JSON, screenshot for the user.
- [ ] Commit `feat(dashboard): SVG replay UI`

### Task 13: Finalize

- [ ] Full suite `.venv/Scripts/python.exe -m pytest tests/ -q` green.
- [ ] README: new sections — observation modes table, trace flags, dashboard quick-start (`python scripts/dashboard.py` → localhost:8050), link to RULES_AUDIT.md.
- [ ] Tiny real check: 10-iteration `realistic`-mode training run completes; record one traced game and replay it in the dashboard.
- [ ] Commit `docs: package A docs` → push branch → PR with summary + dashboard screenshot.

## Self-Review Notes

- Spec coverage: §1.1→T5, §1.2–1.3→T6, §1.4→T7, eval experiment→T8, §2→T9–10, §3→T11–12, §4→T1–4, delivery→T13. Build order matches spec (audit first).
- Type consistency: `BeliefTracker` produced in T5 consumed by T6 (`belief` arg) and T7 (env lifecycle, eval tracker); `TraceRecorder` produced T9 consumed T10–11; `obs_dim_for_mode` produced T6 consumed T7.
- No placeholders; large-module bodies deferred to execution per the same-session rationale in Global Constraints.
