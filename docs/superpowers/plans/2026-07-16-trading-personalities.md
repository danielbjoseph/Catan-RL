# P2P Trading + Personality Opponents (Package B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bounded player-to-player trading (templated offers, accept/decline), five scripted trade-personality opponents, and opponent-pool training so the policy can be trained/evaluated against each archetype.

**Architecture:** The action catalog doubles to 512 slots (v1 prefix untouched; 40 PROPOSE_TRADE templates + ACCEPT + DECLINE from index 256). A `TRADE_RESPONSE` sub-phase walks responders in seat order; the first accepter in that order executes. A 28-dim pending-trade context block is appended at the *absolute end* of every observation mode, so every old observation vector is a strict prefix of its new counterpart — this makes old-checkpoint evaluation (prefix slicing) and curriculum warm-starting (zero-padded widening) exact. Personalities wrap the heuristic bot: only trade decisions differ. Opponent pools slot into `collect_rollouts`; transitions are collected only from designated policy seats.

**Tech Stack:** Existing repo stack — Python 3.14 in `.venv` (`.venv/Scripts/python.exe`), torch CPU, pytest, Flask dashboard, vanilla-JS frontend. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-12-trading-personalities-design.md`

## Global Constraints

- Windows; run everything with `.venv/Scripts/python.exe`; PowerShell has no `&&` (use `;`).
- Full suite must be green before every commit: `.venv/Scripts/python.exe -m pytest tests/ -q` (~2.5 min; 288 tests at branch start — count grows each task).
- Commit messages: plain conventional-commit style, 2-4 sentence bodies, **never** mention Claude/AI, **no** co-author trailers.
- Never commit: `scraper.py`, `shared.js`, `ui-game.js` (stale untracked files), anything under `runs/`.
- Branch: `trading-personalities` (created off `obs-modes-and-replay`).
- Indices 0–255 of the catalog and the pre-existing prefix of every observation layout are **frozen** — tests must prove they didn't move.
- `trades_enabled=False` profiles (all pre-existing ones) must behave bit-identically to before: no new legal actions, no new phases reachable.
- Core RL principle (spec): the *policy's* trade behavior is never hand-coded; hand-coded tendencies live only in scripted personality opponents.

---

### Task 1: Action catalog v2 (512 slots, trade actions)

**Files:**
- Modify: `catan_rl/env/actions.py`
- Test: `tests/test_trading_catalog.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces (later tasks rely on these exact names):
  - `CATALOG_SIZE = 512`, `CATALOG_SIZE_V1 = 256`, `CATALOG_VERSION = 2`
  - `ActionType.PROPOSE_TRADE = 15`, `ActionType.ACCEPT_TRADE = 16`, `ActionType.DECLINE_TRADE = 17`
  - `Action` gains field `give_n: Optional[int] = None` (1 or 2; the "ratio")
  - `propose_trade_action(give: Resource, get: Resource, give_n: int) -> Action` (slots 256–295)
  - `ACCEPT_TRADE = CATALOG[296]`, `DECLINE_TRADE = CATALOG[297]`; slots 298–511 reserved padding
- Slot math: ordered pairs enumerated exactly like maritime (`for give in Resource: for get in Resource: if give != get`), pair index `p` in 0–19; PROPOSE index = `256 + p*2 + (give_n - 1)`.

- [ ] **Step 1: Write failing tests** in `tests/test_trading_catalog.py`:

```python
import pytest
from catan_rl.env.actions import (
    CATALOG, CATALOG_SIZE, CATALOG_SIZE_V1, CATALOG_VERSION,
    Action, ActionType, Resource,
    propose_trade_action, ACCEPT_TRADE, DECLINE_TRADE,
    maritime_trade_action, monopoly_action, PLAY_VICTORY_POINT,
)

def test_catalog_sizes():
    assert CATALOG_SIZE == 512
    assert CATALOG_SIZE_V1 == 256
    assert CATALOG_VERSION == 2
    assert len(CATALOG) == 512

def test_v1_prefix_frozen():
    # Spot-check every v1 segment boundary is untouched.
    assert CATALOG[0].action_type == ActionType.ROLL_DICE
    assert CATALOG[1].action_type == ActionType.END_TURN
    assert CATALOG[2].action_type == ActionType.BUILD_ROAD and CATALOG[2].edge_id == 0
    assert CATALOG[73].edge_id == 71
    assert CATALOG[74].vertex_id == 0 and CATALOG[74].action_type == ActionType.BUILD_SETTLEMENT
    assert CATALOG[181].action_type == ActionType.BUILD_CITY
    assert CATALOG[200].hex_id == 18
    assert CATALOG[204].player_id == 3
    assert maritime_trade_action(Resource.WOOD, Resource.BRICK).catalog_index == 205
    assert CATALOG[230].action_type == ActionType.BUY_DEV_CARD
    assert monopoly_action(Resource.ORE).catalog_index == 252
    assert PLAY_VICTORY_POINT.catalog_index == 253

def test_propose_trade_slots():
    a = propose_trade_action(Resource.WOOD, Resource.BRICK, give_n=1)
    assert a.catalog_index == 256
    assert a.action_type == ActionType.PROPOSE_TRADE
    assert a.resource == Resource.WOOD and a.resource2 == Resource.BRICK and a.give_n == 1
    b = propose_trade_action(Resource.WOOD, Resource.BRICK, give_n=2)
    assert b.catalog_index == 257
    last = propose_trade_action(Resource.ORE, Resource.WHEAT, give_n=2)
    assert last.catalog_index == 295
    with pytest.raises(AssertionError):
        propose_trade_action(Resource.WOOD, Resource.WOOD, give_n=1)
    with pytest.raises(AssertionError):
        propose_trade_action(Resource.WOOD, Resource.BRICK, give_n=3)

def test_accept_decline_slots():
    assert ACCEPT_TRADE.catalog_index == 296
    assert ACCEPT_TRADE.action_type == ActionType.ACCEPT_TRADE
    assert DECLINE_TRADE.catalog_index == 297
    assert DECLINE_TRADE.action_type == ActionType.DECLINE_TRADE

def test_action_str():
    a = propose_trade_action(Resource.SHEEP, Resource.ORE, give_n=2)
    assert str(a) == "PROPOSE_TRADE(give=2xSHEEP, get=ORE)"
    assert str(ACCEPT_TRADE) == "ACCEPT_TRADE"
    assert str(DECLINE_TRADE) == "DECLINE_TRADE"

def test_padding_never_real():
    for i in range(298, 512):
        assert CATALOG[i].action_type == ActionType.ROLL_DICE  # unreachable filler
```

- [ ] **Step 2: Run to verify failure** — `.venv/Scripts/python.exe -m pytest tests/test_trading_catalog.py -q` → ImportError (`CATALOG_SIZE_V1` missing).

- [ ] **Step 3: Implement** in `catan_rl/env/actions.py`:
  - `CATALOG_SIZE_V1 = 256`, `CATALOG_SIZE = 512`, `CATALOG_VERSION = 2`; update the module docstring's layout table (add rows for 256–295, 296, 297, 298–511).
  - `ActionType` gains `PROPOSE_TRADE = 15`, `ACCEPT_TRADE = 16`, `DECLINE_TRADE = 17`.
  - `Action` gains `give_n: Optional[int] = None`; `_build_catalog`'s `add()` must pass it through.
  - `__str__` additions:

```python
        if self.action_type == ActionType.PROPOSE_TRADE:
            return f"{t}(give={self.give_n}x{self.resource.name}, get={self.resource2.name})"
```

  - In `_build_catalog`, after the v1 section, pad to 256 first (existing `while` loop but bounded at `CATALOG_SIZE_V1`), then:

```python
    # 256-295: PROPOSE_TRADE (20 ordered pairs x give_n in {1,2})
    for give in Resource:
        for get in Resource:
            if give != get:
                for n in (1, 2):
                    add(Action(ActionType.PROPOSE_TRADE, resource=give, resource2=get, give_n=n))
    # 296: ACCEPT_TRADE, 297: DECLINE_TRADE
    add(Action(ActionType.ACCEPT_TRADE))
    add(Action(ActionType.DECLINE_TRADE))
    # 298-511: reserved padding
    while len(catalog) < CATALOG_SIZE:
        add(Action(ActionType.ROLL_DICE))
```

  - Accessors:

```python
def propose_trade_action(give: Resource, get: Resource, give_n: int) -> Action:
    assert give != get
    assert give_n in (1, 2)
    p = 0
    for g in Resource:
        for r in Resource:
            if g != r:
                if g == give and r == get:
                    return CATALOG[256 + p * 2 + (give_n - 1)]
                p += 1
    raise ValueError(f"Invalid trade pair: {give} -> {get}")

ACCEPT_TRADE  = CATALOG[296]
DECLINE_TRADE = CATALOG[297]
```

  (Note: `ACCEPT_TRADE`/`DECLINE_TRADE` module constants must be defined *after* `CATALOG` is built, next to the other accessors.)

- [ ] **Step 4: Ripple check** — grep tests for hardcoded `256` catalog-size assertions (`grep -rn "== 256\|(256" tests/`) and update any that assert catalog width (e.g. `tests/test_actions.py`, mask-shape asserts in `tests/test_env_smoke.py` use the `CATALOG_SIZE` constant already — those auto-adjust). `catan_rl/rl/rollout.py` `Batch.masks` docstring says `(N, 256)` → change to `(N, CATALOG_SIZE)`. `catan_rl/env/pettingzoo_env.py` module docstring mentions shape (256,) → update comment.

- [ ] **Step 5: Full suite green** — `.venv/Scripts/python.exe -m pytest tests/ -q`. Everything (masks, models, env) sizes off `CATALOG_SIZE`, so this proves the widening is transparent: trade slots are never legal yet, so behavior is unchanged.

- [ ] **Step 6: Commit** — `feat(env): 512-slot action catalog v2 with trade actions`

---

### Task 2: RulesProfile trading fields + new profiles

**Files:**
- Modify: `catan_rl/env/rules_profile.py`
- Test: extend `tests/test_rules_profile.py` (exists; read it first and match style)

**Interfaces:**
- Produces: `RulesProfile.trades_enabled: bool = False`, `RulesProfile.max_trades_per_turn: int = 3`; builtins `STANDARD_TRADING` (`name="standard_trading"`, dev cards on, trades on) and `SIMPLIFIED_TRADING_V1` (`name="simplified_trading_v1"`, dev cards off, trades on); both registered in `_BUILTIN`; `to_dict`/`from_dict`/`load` round-trip the new fields with backcompat defaults (`d.get("trades_enabled", False)`, `d.get("max_trades_per_turn", 3)`).

- [ ] **Step 1: Failing tests**:

```python
def test_trading_profiles_builtin():
    p = RulesProfile.get("standard_trading")
    assert p.trades_enabled and p.dev_cards_enabled and p.max_trades_per_turn == 3
    q = RulesProfile.get("simplified_trading_v1")
    assert q.trades_enabled and not q.dev_cards_enabled

def test_existing_profiles_no_trading():
    assert not RulesProfile.get("standard").trades_enabled
    assert not RulesProfile.get("simplified_v1").trades_enabled

def test_from_dict_backcompat_missing_trade_keys():
    p = RulesProfile.from_dict({"name": "standard", "dev_cards_enabled": True, "win_vp": 10})
    assert p.trades_enabled is False and p.max_trades_per_turn == 3

def test_round_trip_with_trading():
    p = RulesProfile.get("standard_trading")
    assert RulesProfile.from_dict(p.to_dict()) == p
```

- [ ] **Step 2: Run → fail.** **Step 3: Implement** (dataclass fields, two builtins, serialization incl. `load()` YAML keys). **Step 4: Suite green.** Note: old checkpoints/traces store profile dicts without the new keys — `from_dict` backcompat covers both (there is a trace-header `profile.to_dict()` in every recorded game; `tests/test_trace.py` will exercise it).

- [ ] **Step 5: Commit** — `feat(env): trades_enabled rules profiles`

---

### Task 3: Trade sub-phase state machine

**Files:**
- Modify: `catan_rl/env/game_state.py`, `catan_rl/env/rules.py`, `catan_rl/env/validators.py`
- Test: `tests/test_trading_rules.py` (create)

**Interfaces:**
- Consumes: Task 1 actions, Task 2 profile fields.
- Produces:
  - `Phase.TRADE_RESPONSE = 12` (appended after `GAME_OVER = 11`; do NOT renumber existing members).
  - `GameState.pending_trade: Optional[dict]` shaped `{"proposer": int, "give": int, "get": int, "give_n": int, "responses": {pid: Optional[bool]}}` (`give`/`get` stored as ints; `responses` covers the 3 non-proposer pids: `None`=pending, `False`=declined, `True`=accepted).
  - `GameState.trades_proposed_this_turn: int = 0` (reset in `_end_turn`).
  - `clone()`, `to_dict()`, `from_dict()` handle both (deep-copy `responses`; in `to_dict` stringify response keys like `discard_obligations`; `from_dict` uses `d.get("pending_trade")` / `d.get("trades_proposed_this_turn", 0)` for old traces).
  - Response/resolution seat order: ascending seats starting at `(proposer+1) % 4`, wrapping (helper `_responder_order(proposer, n_players) -> list[int]` in rules.py).

**Semantics (from spec §1.2):**
- `PROPOSE_TRADE` legal only in `Phase.MAIN` (MAIN is only reachable post-roll, so "no trades before rolling" is automatic), only when `profile.trades_enabled`, `trades_proposed_this_turn < profile.max_trades_per_turn`, and proposer holds ≥ `give_n` of `give`.
- On propose: increment `trades_proposed_this_turn`; responders lacking ≥1 of `get` are auto-declined (`responses[pid] = False`) without ever acting; if any responder is still pending → `phase = TRADE_RESPONSE`, `current_player` = first pending responder in order; if none → resolve immediately (all declined → back to MAIN, no transfer).
- In `TRADE_RESPONSE`: legal actions are exactly `[ACCEPT_TRADE, DECLINE_TRADE]`. Acting sets `responses[current]`; advance to next pending responder or resolve.
- Resolve: accepters in responder order; if ≥1, the FIRST executes — proposer loses `give_n × give` / gains `1 × get`; accepter loses `1 × get` / gains `give_n × give`. Bank untouched. Always: `pending_trade = None`, `current_player = proposer`, `phase = Phase.MAIN`.

- [ ] **Step 1: Failing tests** in `tests/test_trading_rules.py`. Build states directly (pattern: `BoardConfig.standard(seed=0)`, `GameState.new_game(config, profile="standard_trading")`, then force `state.phase = Phase.MAIN`, `state.current_player = 0`, `state.rolled_this_turn = True`, hand-set `resources`). Required tests (write all with real assertions):

```python
def _trading_state(seed=0):
    config = BoardConfig.standard(seed=seed)
    state = GameState.new_game(config, n_players=4, seed=seed, profile="standard_trading")
    state.phase = Phase.MAIN
    state.current_player = 0
    state.rolled_this_turn = True
    return state

def test_propose_requires_holdings():
    state = _trading_state()
    state.players[0].resources = [1, 0, 0, 0, 0]
    legal = {a.catalog_index for a in legal_actions(state)}
    assert propose_trade_action(Resource.WOOD, Resource.BRICK, 1).catalog_index in legal
    assert propose_trade_action(Resource.WOOD, Resource.BRICK, 2).catalog_index not in legal  # only 1 wood
    assert propose_trade_action(Resource.ORE, Resource.WOOD, 1).catalog_index not in legal    # no ore

def test_no_propose_when_trades_disabled():
    # same state but profile "standard": no PROPOSE_TRADE in legal actions
    ...

def test_full_response_walk_first_accepter_wins():
    state = _trading_state()
    state.players[0].resources = [2, 0, 0, 0, 0]
    for pid in (1, 2, 3):
        state.players[pid].resources = [0, 1, 0, 0, 0]  # all hold the wanted brick
    rng = random.Random(0)
    apply_action(state, propose_trade_action(Resource.WOOD, Resource.BRICK, 2), rng)
    assert state.phase == Phase.TRADE_RESPONSE and state.current_player == 1
    apply_action(state, DECLINE_TRADE, rng)
    assert state.current_player == 2
    apply_action(state, ACCEPT_TRADE, rng)
    assert state.current_player == 3            # 3 still gets to respond
    apply_action(state, ACCEPT_TRADE, rng)
    # first accepter in order (2) executes
    assert state.phase == Phase.MAIN and state.current_player == 0
    assert state.players[0].resources == [0, 1, 0, 0, 0]
    assert state.players[2].resources == [2, 0, 0, 0, 0]
    assert state.players[3].resources == [0, 1, 0, 0, 0]  # untouched

def test_auto_decline_skips_broke_responders():
    # only player 3 holds the get resource -> phase jumps straight to player 3
    ...

def test_all_auto_declined_returns_to_main_no_transfer():
    ...

def test_trade_cap_per_turn():
    # after max_trades_per_turn proposals (each fully declined), PROPOSE slots leave the mask
    ...

def test_end_turn_resets_cap_counter():
    ...

def test_resource_conservation_across_trades():
    # sum over players+bank of each resource is invariant through propose/respond/execute
    ...

def test_clone_and_serialization_round_trip_mid_trade():
    # clone() deep-copies pending_trade; to_dict/from_dict round-trips it exactly;
    # from_dict of a dict WITHOUT the new keys yields pending_trade None (backcompat)
    ...

def test_notrade_profile_bit_identical():
    # run 200 random-legal plies with seed on "standard" profile: legal_action_mask
    # never sets any index >= 256, and Phase.TRADE_RESPONSE never occurs
    ...
```

- [ ] **Step 2: Run → fail.** **Step 3: Implement**:
  - `game_state.py`: `Phase.TRADE_RESPONSE = 12`; two new dataclass fields; clone/to_dict/from_dict updates.
  - `rules.py`: dispatch `PROPOSE_TRADE` → `_propose_trade(state, action)`, `ACCEPT_TRADE`/`DECLINE_TRADE` → `_respond_trade(state, accept: bool)`; `_end_turn` gains `state.trades_proposed_this_turn = 0`; implement `_responder_order`, `_propose_trade`, `_respond_trade`, `_resolve_trade` exactly per semantics above.
  - `validators.py`: in `_main_actions`, after maritime trades:

```python
    # P2P trade proposals
    if state.profile.trades_enabled and state.trades_proposed_this_turn < state.profile.max_trades_per_turn:
        for give in Resource:
            for get in Resource:
                if give == get:
                    continue
                for n in (1, 2):
                    if player.resources[int(give)] >= n:
                        actions.append(propose_trade_action(give, get, n))
```

  and a new top-level branch in `legal_actions`:

```python
    if phase == Phase.TRADE_RESPONSE:
        return [ACCEPT_TRADE, DECLINE_TRADE]
```

- [ ] **Step 4: Suite green** (the no-trade bit-identical test is the regression gate for all 288 existing tests' semantics). **Step 5: Commit** — `feat(env): P2P trade sub-phase state machine`

---

### Task 4: Observation trade block, dims, golden fixture migration, belief test

**Files:**
- Modify: `catan_rl/env/observation.py`, `tests/test_observation_modes.py` (dim literals + fixture), `tests/fixtures/golden_observations.npz` (regenerate)
- Test: extend `tests/test_observation_modes.py`, `tests/test_belief.py`

**Interfaces:**
- Produces: `_SEG_TRADE = 28`; new dims `OBS_DIM = 1548`, `OBS_DIM_PERFECT = 1593`, `OBS_DIM_REALISTIC = 1577`, `OBS_DIM_GLOBAL = 1604`. Trade block layout (appended at the very END of every mode's vector, after all mode-specific extras):
  - `[0]` active flag (1.0 iff `state.pending_trade is not None`)
  - `[1:5]` proposer one-hot, rotated (`rel = (proposer - observer) % 4`)
  - `[5:10]` give one-hot; `[10:15]` get one-hot
  - `[15]` `give_n / 2.0`
  - `[16:28]` responses: for `rel_i` in 0..3, 3 states one-hot at `16 + rel_i*3`: `+0` pending/none (also used for the proposer's own slot and when no trade), `+1` declined, `+2` accepted
- **Prefix invariant (load-bearing for Tasks 7–8):** for every mode, the first `old_dim` entries of the new vector are bit-identical to the old vector. This holds because the block is appended last; prove it before regenerating the fixture.

- [ ] **Step 1: Prefix-invariant migration check FIRST.** Before touching observation.py, write a throwaway script (scratchpad, not committed) that loads `tests/fixtures/golden_observations.npz` and re-runs the fixture generation procedure (read `_generate_golden_fixture` / the fixture docs in `tests/test_observation_modes.py` for the exact seed/plies/observer), asserting current vectors equal the fixture. This is the baseline. After implementing, re-run it comparing `new_vec[:old_dim] == fixture_vec` exactly. If the prefix drifts, STOP — the layout change is wrong.

- [ ] **Step 2: Failing tests**:

```python
def test_new_dims():
    assert OBS_DIM == 1548 and OBS_DIM_PERFECT == 1593
    assert OBS_DIM_REALISTIC == 1577 and OBS_DIM_GLOBAL == 1604

def test_trade_block_zero_when_no_pending_trade():
    # fresh game, any mode: last 28 entries all zero except the 4 response
    # "pending/none" one-hots at offsets 16,19,22,25 which are 1.0
    ...

def test_trade_block_encodes_pending_trade():
    # build a mid-trade state (pattern from test_trading_rules), observer=2:
    # active==1; proposer rel one-hot correct; give/get one-hots; give_n/2;
    # responder that declined shows +1 hot in ITS rel slot
    ...

def test_trade_block_rotation():
    # same state, two observers: proposer rel slot differs correctly
    ...
```

  Decision (spec open item, settled): the "pending/none" response state is deliberately hot when idle so the block is never all-zero garbage-vs-meaningful ambiguous; document in the module docstring.

- [ ] **Step 3: Implement** in `observation.py`: `_SEG_TRADE = 28`, add to all four dim constants, and append the block in `make_observation` after all existing mode branches:

```python
    trade = np.zeros(_SEG_TRADE, dtype=np.float32)
    pt = state.pending_trade
    if pt is not None:
        trade[0] = 1.0
        trade[1 + (pt["proposer"] - observer) % _N_PLAYERS] = 1.0
        trade[5 + pt["give"]] = 1.0
        trade[10 + pt["get"]] = 1.0
        trade[15] = pt["give_n"] / 2.0
    for rel_i in range(_N_PLAYERS):
        pid = (observer + rel_i) % _N_PLAYERS
        resp = pt["responses"].get(pid) if pt is not None else None
        # note: from_dict'd states may have str keys; normalize in game_state, not here
        off = 16 + rel_i * 3
        trade[off + (0 if resp is None else (1 if resp is False else 2))] = 1.0
    return np.concatenate([obs, trade])
```

  (`from_dict` in Task 3 must int-ify response keys so this indexing is safe.)

- [ ] **Step 4: Migration** — run the Step-1 prefix check (must pass), then regenerate `tests/fixtures/golden_observations.npz` with the existing generation script and update the dim literals in `tests/test_observation_modes.py`. The commit body must state the fixture was regenerated due to the deliberate layout extension and that prefix equality with the old fixture was verified.

- [ ] **Step 5: Belief trade test** in `tests/test_belief.py`:

```python
def test_executed_trade_reconciles_exactly():
    # trading state; tracker anchored at a pre-trade snapshot where both
    # hands are publicly known (fresh reset); run propose->accept via
    # apply_action with tracker.on_action(before, action, after) per ply;
    # assert tracker.expected(proposer) and expected(accepter) EXACTLY match
    # true hands (trades are public deltas -> zero drift, zero added
    # hidden_mass).
    ...
```

  No `belief.py` changes expected — `ACCEPT_TRADE` flows through `_apply_public_delta`. If this test fails, the bug is in the state machine, not the tracker.

- [ ] **Step 6: Suite green** (env smoke on a trading profile too: add one `CatanAECEnv(rules_profile="standard_trading")` random-legal full-game test to `tests/test_env_smoke.py`). **Step 7: Commit** — `feat(env): pending-trade observation block for all modes`

---

### Task 5: Trade personalities

**Files:**
- Create: `catan_rl/bots/personalities.py`
- Modify: `catan_rl/bots/heuristic_bot.py`, `catan_rl/bots/greedy_bot.py` (default-decline), `catan_rl/bots/__init__.py` (export)
- Test: `tests/test_personalities.py` (create)

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class TradePersonality:
    name: str
    propose_rate: float               # P(attempt a proposal) per MAIN visit
    accept_margin: float              # required value gain to accept (inf = never)
    leader_block_vp: Optional[int]    # decline proposer within this many VP of winning (None = off)
    desperation_scale: float          # accept_margin loosens by this per VP behind the leader
    max_proposals_per_turn: int
    accept_band: Optional[float] = None  # fair_dealer: accept iff |margin| <= band (overrides accept_margin)

PERSONALITIES: Dict[str, TradePersonality]   # 5 presets keyed by name
def make_personality_bot(personality: TradePersonality) -> Callable  # (state, rng) -> Action, same BotFn shape as greedy_bot.pick_action
def resource_pips(state, pid) -> np.ndarray          # (5,) own production pips per resource
def trade_margin(state, pid, gain: Dict[int,int], lose: Dict[int,int]) -> float
```

- Value model (spec §2.1 + open item settled): `value(r) = 1.0 / (1.0 + pips[r])` (scarcity from own production spread — a resource you produce a lot of is cheap to you) **plus** a build-need bonus of `+0.5` if gaining one `r` makes a build (settlement > city > road, checked in that order) affordable that wasn't. `trade_margin = Σ gain[r]*value(r) − Σ lose[r]*value(r)` (+ bonus on the gain side). This makes `fair_dealer` naturally break-even on 1:1 same-scarcity trades: equal pips → equal value → margin 0.
- Presets (calibration numbers are starting points; the property tests below pin the *behavioral contracts*, not the numbers):

| name | propose_rate | accept_margin | leader_block_vp | desperation_scale | max/turn | band |
|---|---|---|---|---|---|---|
| `never_trader` | 0.0 | `float("inf")` | None | 0.0 | 0 | — |
| `opportunist` | 0.9 | −0.5 | None | 0.0 | 3 | — |
| `stall_the_leader` | 0.4 | 0.0 | 2 | 0.0 | 3 | — |
| `fair_dealer` | 0.4 | 0.0 | None | 0.0 | 2 | 0.15 |
| `desperado` | 0.6 | 0.3 | None | 0.25 | 3 | — |

- Behavior wrapper: `make_personality_bot` returns `pick_action(state, rng)`:
  - `Phase.TRADE_RESPONSE`: decide via `pending_trade` — decline if `leader_block_vp` set and `compute_vp(proposer) >= state.profile.win_vp - leader_block_vp` or proposer is the current sole VP leader; else compute responder margin (`gain = {give: give_n}`, `lose = {get: 1}`), effective threshold `accept_margin - desperation_scale * max(0, leader_vp - my_vp)`; accept iff margin ≥ threshold (or `|margin| <= accept_band` when band set). Return `ACCEPT_TRADE`/`DECLINE_TRADE`.
  - `Phase.MAIN`: if PROPOSE actions are legal, own proposals so far this turn < `max_proposals_per_turn`, and `rng.random() < propose_rate`: among legal PROPOSE actions compute proposer margin (`gain = {get: 1}`, `lose = {give: give_n}`), pick the argmax if its margin > 0, else fall through. Everything else → `heuristic_bot.pick_action(state, rng)` (spec: non-trade decisions identical across presets).
  - `stall_the_leader`'s "prefers partners furthest behind" is a documented no-op in v1 broadcast trading (no partner choice exists); note it in the preset's docstring.
- Base-bot patch: `heuristic_bot.pick_action` and `greedy_bot.pick_action` get, right after the DISCARD branch: `if ActionType.DECLINE_TRADE in by_type: return by_type[ActionType.DECLINE_TRADE][0]` — plain bots conservatively decline (random_bot stays random). Without this they'd fall to `rng.choice` on responses.

- [ ] **Step 1: Failing tests** (property style, per spec §2.3):

```python
def test_never_trader_never_proposes_or_accepts():
    # 5 seeded full games, 4x never_trader on standard_trading: assert no ply
    # ever applies PROPOSE_TRADE or ACCEPT_TRADE (walk with a recording loop)

def test_stall_the_leader_declines_leader_adjacent_only():
    # construct a response state twice: proposer at win_vp-1 (declines) vs
    # proposer far behind with an attractive margin (accepts)

def test_desperado_threshold_moves_with_deficit():
    # identical marginal trade; desperado behind by 4 VP accepts, tied declines

def test_fair_dealer_band():
    # near-even trade accepted; lopsided-in-their-FAVOR trade declined (band, not floor)

def test_opportunist_accepts_negative_margin():
    ...

def test_mixed_personality_games_terminate_and_conserve():
    # 4 different presets, 3 seeds, full games on standard_trading:
    # terminates or hits turn cap; per-resource sum(players)+bank invariant
    # checked every ply; at least one executed trade across the batch

def test_plain_bots_decline():
    # heuristic_bot and greedy_bot in TRADE_RESPONSE return DECLINE_TRADE
```

- [ ] **Step 2: Run → fail.** **Step 3: Implement** (personalities.py ~200 lines; the `resource_pips` helper reuses `_common.hex_pips` over the player's settlement/city vertices — cities count double, mirroring `player_production_on_hex` but bucketed per resource). Track own proposals-per-turn inside the closure by watching `state.trades_proposed_this_turn` (it's the proposer's own counter — only the current player proposes).

- [ ] **Step 4: Suite green.** **Step 5: Commit** — `feat(bots): trade personality presets`

---

### Task 6: Opponent-pool rollouts + trainer plumbing

**Files:**
- Modify: `catan_rl/rl/rollout.py`, `catan_rl/rl/self_play.py`, `scripts/train_self_play.py` (config keys only; no new CLI flags needed beyond what config carries)
- Create: `configs/ppo_trading_pool.yaml` (working example)
- Test: `tests/test_opponent_pool.py` (create)

**Interfaces:**
- Consumes: `PERSONALITIES`, `make_personality_bot` (Task 5); `load_policy` metadata `obs_mode`.
- Produces:

```python
def collect_rollouts(policy, n_games, *, ..., opponents: Optional[Dict] = None, n_policy_seats: int = 1, ...) -> Batch
```

  - `opponents=None` (default): existing pure 4-seat self-play, byte-for-byte unchanged (all current call sites unaffected).
  - `opponents={"pool": [entry, ...]}` with entries `{"type": "personality", "name": ..., "weight": w}`, `{"type": "checkpoint", "path": ..., "weight": w}`, `{"type": "self", "weight": w}`.
  - Per game `g`: policy seats = `{(g + k) % 4 for k in range(n_policy_seats)}` (rotating). Remaining seats each draw independently from the pool by weight, seeded from the game seed (`random.Random(game_seed ^ 0x5EED)` so seat draws are reproducible and independent of the action RNG).
  - Transitions are collected ONLY from policy seats. `self` pool seats act with the live policy (stochastic sample) but their transitions are discarded; `checkpoint` opponents act deterministically (eval convention) in their own stored `obs_mode`; `personality` opponents act via `pick_action(env._state, rng).catalog_index`.
  - **v1 restriction:** a `checkpoint` pool entry whose stored `obs_mode == "realistic"` is only allowed when it equals the env's `obs_mode` (shares `env._belief`); otherwise raise `ValueError` at pool-build time with a clear message. (Perfect/global/self_play modes need no tracker: build their obs via `make_observation(env._state, observer=seat, mode=their_mode)`.)
  - Checkpoint opponents may be old 256-action/short-obs policies: act through the prefix-slicing helper (Task 7 — see interface below; for THIS task, implement the helper here in `catan_rl/rl/models.py` since rollout needs it first):

```python
def act_prefix_sliced(policy: ActorCritic, obs: np.ndarray, mask: np.ndarray,
                      device: str = "cpu", deterministic: bool = True) -> int:
    """Act with a policy whose obs_dim/n_actions may be a PREFIX of the
    current layout (old checkpoints). Slices obs/mask to the policy's dims;
    if the sliced mask is empty (e.g. a 256-head policy asked to respond to
    a trade), returns DECLINE_TRADE's index."""
```

  - `Batch.stats` gains `"policy_win_rate"` (wins from policy seats / games) and `"opponent_win_rates": {label: rate}` where label is `personality:<name>` / `checkpoint:<stem>` / `self`, rate = policy wins in games where that label held ≥1 seat / such games.
- Trainer: `_RUN_DEFAULTS` gains `"opponents": None`, `"n_policy_seats": 1`; `collect_rollouts` call passes both; `_log_iteration` adds `game/policy_win_rate` and `game/win_rate_vs_<label>` scalars when pool mode is on.
- `configs/ppo_trading_pool.yaml`: copy `configs/ppo_baseline.yaml` structure with `rules_profile: standard_trading`, `obs_mode: self_play`, and the spec's example pool (opportunist ×2, stall_the_leader ×1, self ×2 — drop the checkpoint entry, it's environment-specific; document the shape in a comment).

- [ ] **Step 1: Failing tests** (use tiny `ActorCritic(hidden_sizes=(8,8))`, `FAST_PROFILE`-style low turn caps as in `tests/test_rollout.py`):

```python
def test_default_none_is_pure_self_play():
    # opponents=None: Batch has transitions from all 4 seats (seat_ids covers {0,1,2,3})

def test_pool_only_policy_seat_transitions():
    # pool of 3x never_trader, n_policy_seats=1, 4 games:
    # every Batch.seat_ids value equals the rotating policy seat of its episode_id

def test_pool_sampling_reproducible():
    # same seed twice -> identical opponent assignment (assert via stats labels/games)

def test_stats_have_opponent_win_rates():
    ...

def test_realistic_checkpoint_mode_mismatch_raises():
    # save a tiny ckpt with obs_mode="realistic", env obs_mode="self_play" -> ValueError

def test_act_prefix_sliced_old_head_declines_trades():
    # 256-action policy + a 512 mask where only indices 296/297 are legal -> returns 297
```

- [ ] **Step 2: Run → fail.** **Step 3: Implement.** Keep the `opponents=None` hot path free of any pool machinery (guard once at the top, mirroring the tracing pattern). **Step 4: Suite green.** **Step 5: Commit** — `feat(rl): opponent-pool training with personality and checkpoint seats`

---

### Task 7: Personality evaluation + old-checkpoint compatibility

**Files:**
- Modify: `catan_rl/rl/evaluate.py` (`policy_action` prefix slicing), `catan_rl/rl/self_play.py` (`_evaluate` personality scalars), `scripts/evaluate_checkpoints.py` (`--vs` accepts personality names), `catan_rl/bots/__init__.py` (bot registry)
- Test: extend `tests/test_evaluate.py`, `tests/test_personalities.py`

**Interfaces:**
- Consumes: `act_prefix_sliced` (Task 6, in models.py), `PERSONALITIES`/`make_personality_bot` (Task 5).
- Produces:
  - `catan_rl/bots/__init__.py`: `def resolve_bot(name: str) -> BotFn` — resolves `"random"`, `"greedy"`, `"heuristic"`, or any personality preset name; `ValueError` with the valid-name list otherwise.
  - `policy_action` in evaluate.py routes through `act_prefix_sliced`, so ALL eval paths (vs bots, vs checkpoint, policy-vs-policy) transparently support old Package-A checkpoints (1520-obs/256-action) inside the new 1548-obs/512-action env: obs and mask are exact prefixes (Task 4 invariant), and trade responses auto-decline.
  - Trainer `_evaluate`: new config key `"eval_personalities"` (default `None` → all 5 preset names when `profile.trades_enabled`, else `[]`); for each, `evaluate_vs_bots(policy, make_personality_bot(PERSONALITIES[name]), n, **kwargs)` → TB scalar `eval/win_rate_vs_<name>` (spec §3.3).
  - `scripts/evaluate_checkpoints.py --vs never_trader,opportunist,...` works via `resolve_bot`; update `--help`.

- [ ] **Step 1: Failing tests**:

```python
def test_resolve_bot_names():
    # random/greedy/heuristic/all 5 personalities resolve; "nope" raises with names listed

def test_old_checkpoint_plays_in_trading_env():
    # build ActorCritic(obs_dim=1520, n_actions=256) untrained, save_checkpoint
    # with obs_mode="self_play"; evaluate_vs_bots(policy, resolve_bot("opportunist"),
    # n_games=1, rules_profile="standard_trading", max_turns=60, ...) completes
    # without a shape error (this is the load-bearing backcompat test)

def test_eval_personalities_default_gating():
    # trainer cfg with standard profile -> personality eval list empty;
    # with standard_trading -> all five (unit-test the helper that computes it)
```

- [ ] **Step 2: Run → fail.** **Step 3: Implement.** **Step 4: Suite green.** **Step 5: Commit** — `feat(rl): personality evaluation and old-checkpoint compatibility`

---

### Task 8: Curriculum warm-start (widen old checkpoints)

**Files:**
- Modify: `catan_rl/rl/checkpointing.py`, `scripts/train_self_play.py` (`--init-from PATH`)
- Test: `tests/test_checkpointing.py` (extend or create)

**Interfaces:**
- Consumes: checkpoint arch metadata (`obs_dim`, `n_actions`, `hidden_sizes`).
- Produces:

```python
def widen_policy(old: ActorCritic, new_obs_dim: int, new_n_actions: int) -> ActorCritic:
    """Prefix-preserving widening: first trunk layer's new input columns are
    zero (new features ignored until learned); policy head's new rows get
    zero weights and bias -4.0 (near-zero initial probability, spec §3.4);
    all other weights copied. Requires new dims >= old dims and identical
    hidden_sizes."""
```

  - `train_self_play.py --init-from runs/x/checkpoints/ckpt_000100.pt`: loads the checkpoint, widens to the current config's `obs_dim_for_mode(obs_mode)` / `CATALOG_SIZE` if smaller, and installs it as the trainer's starting policy (optimizer starts fresh). Print a one-line summary of the widening performed.

- [ ] **Step 1: Failing tests**:

```python
def test_widen_preserves_old_function_exactly():
    old = ActorCritic(obs_dim=1520, n_actions=256, hidden_sizes=(32, 32))
    new = widen_policy(old, 1548, 512)
    x_old = torch.randn(3, 1520)
    x_new = torch.cat([x_old, torch.zeros(3, 28)], dim=1)
    lo, vo = old(x_old); ln, vn = new(x_new)
    assert torch.allclose(lo, ln[:, :256], atol=1e-6)
    assert torch.allclose(vo, vn, atol=1e-6)
    assert torch.all(ln[:, 256:] == -4.0)  # zero weights + bias -4

def test_widen_rejects_shrink_or_hidden_mismatch():
    ...

def test_init_from_flag_smoke():
    # save a 1520/256 ckpt; run SelfPlayTrainer init path with obs_mode self_play
    # + init_from -> trainer.policy has obs_dim 1548, n_actions 512
```

- [ ] **Step 2: Run → fail.** **Step 3: Implement.** **Step 4: Suite green.** **Step 5: Commit** — `feat(rl): warm-start widening for curriculum fine-tuning`

---

### Task 9: Dashboard pending-trade panel

**Files:**
- Modify: `catan_rl/dashboard/static/app.js`, `catan_rl/dashboard/static/style.css` (index.html only if a container div is needed)

**Interfaces:**
- Consumes: trace ply `state.pending_trade` (Task 3 serialization: `give`/`get` ints, `responses` str-keyed in JSON) and `state.trades_proposed_this_turn`.
- Produces: in the replay view, when the selected ply's state has a non-null `pending_trade`, render a banner between the board and the readout: `"<seat> offers <n>× <RES> for 1× <RES>"` with seat-colored name chips and per-responder status chips (pending / declined / accepted). Hidden when null. All strings through the existing `escapeHtml`; resource names from the existing `RESOURCE_NAMES` table.

- [ ] **Step 1: Generate a trading trace** — scratchpad script: 4 mixed personalities (opportunist, fair_dealer, desperado, stall_the_leader), `standard_trading`, seeded, `TraceRecorder`, save into a scratch runs dir. Verify the JSON contains ≥1 ply with non-null `pending_trade` (pick a seed until it does; opportunist ×1 guarantees proposals quickly).
- [ ] **Step 2: Implement** the banner render (bind into the existing per-ply `renderPly` flow so scrubbing updates it).
- [ ] **Step 3: Browser verification (required, Playwright MCP)** — serve via `scripts/dashboard.py --port 8053 --runs-dir <scratch>`; navigate to the trace; jump to a ply with a pending trade (find its index from the JSON first); assert via `browser_evaluate` that the banner is visible and its text matches the JSON's `pending_trade` (give_n/resources); assert it's hidden on ply 0; console clean; screenshot to `.superpowers/sdd/task-b9-screenshot.png`; kill the server.
- [ ] **Step 4: Full suite still green** (no backend change expected). **Step 5: Commit** — `feat(dashboard): pending-trade banner in replay view`

---

### Task 10: Finalize

- [ ] Full suite green.
- [ ] README: new "Player-to-player trading" section (state machine, profiles, 512-slot catalog, note that old checkpoints keep working via prefix slicing); "Trade personalities" table (5 presets, one behavioral line each); opponent-pool YAML example (point at `configs/ppo_trading_pool.yaml`); curriculum quick-guide (`--init-from` widening). Verify every flag/name against the code before documenting.
- [ ] Real end-to-end check: 10-iteration training run, `configs/ppo_trading_pool.yaml`-based (8 games/iter, `--trace 4`), completes; checkpoint metadata sane; TB/e stdout shows per-opponent win rates; one trace contains executed trades; replay it in the dashboard and see the trade banner.
- [ ] Commit `docs: package B docs` → push branch. (PR is created by the main agent after the final whole-branch review.)

## Self-Review Notes

- Spec coverage: §1.1→T1, §1.2→T2+T3, §1.3→T4, §2.1–2.3→T5, §3.1–3.2→T6, §3.3→T7, §3.4 curriculum→T8, delivery→T9–10. Open items settled: value-model calibration (T5: scarcity + build-need bonus, fair_dealer break-even property holds by construction); TRADE_RESPONSE seats see the proposer's believed hand (already true — realistic obs carries beliefs for all opponents every ply); 2:1 direction confirmed give-2-identical-for-1 only (1:2 is the other side's 2:1, per spec).
- Type consistency: `propose_trade_action(give, get, give_n)` (T1) consumed by T3 validators and T5 personalities; `pending_trade` dict shape (T3) consumed by T4 obs and T9 dashboard; `act_prefix_sliced` defined in T6 (models.py), consumed by T6 rollout and T7 evaluate; `PERSONALITIES`/`make_personality_bot`/`resolve_bot` (T5/T7) consumed by T6/T7/T9/T10; `widen_policy` (T8) standalone.
- The Task-4 prefix invariant is the single load-bearing migration assumption; it is verified empirically before the fixture regeneration and again by T7's old-checkpoint test and T8's widening test.
