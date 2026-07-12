# Catan RL Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Catan self-play RL system per `Outline.md`: rules profiles, scripted bots, PPO shared-policy self-play trainer with per-seat GAE, evaluation harness, throughput benchmark, and a gcloud training how-to.

**Architecture:** The existing engine (`catan_rl/env/`) is rules-complete with a 256-slot action catalog, AEC env, and observation generator (OBS_DIM=1520). We add a `RulesProfile` threaded through `GameState`, scripted bots sharing the `pick_action(state, rng) -> Action` signature, and a `catan_rl/rl/` package: ActorCritic MLP with masked categorical head → per-seat rollout collection from the AEC env → per-seat GAE → pooled PPO update → TensorBoard logging → checkpointed evaluation vs bots and prior checkpoints.

**Tech Stack:** Python 3.14 (project venv at `.venv/`), PyTorch (latest stable), TensorBoard, PyYAML, numpy, pytest.

## Global Constraints

- All installs go into `.venv` (`.venv/Scripts/python.exe -m pip install ...`), never global. (user rule)
- Custom PPO — no stable-baselines3. (spec §0)
- Policy head is always `nn.Linear(hidden, 256)`; illegal logits masked to -inf. (spec §3 Phase 2)
- GAE computed independently per seat; never interleave seats' transitions. (spec Phase 4)
- Rewards: winner +1, losers -1, configurable. (spec Phase 4)
- TensorBoard scalars exactly as named in spec Phase 4 (`train/policy_loss`, ..., `eval/win_rate_vs_random`, ...). Logs to `runs/<experiment_name>/`.
- Throughput target: ≥ 500 complete simplified games/hour on CPU, measured with random agents.
- Everything seedable: env seed, board seed, dice, dev deck, torch seed, checkpoint metadata.
- Serialization: dataclasses + JSON for game state; `torch.save` for weights only (state_dict, not pickled modules).
- Existing interfaces to build on (do not break):
  - `CatanAECEnv(obs_mode, reward_win, reward_loss, max_turns)` — `reset(seed)`, `step(int)`, `observe(agent) -> {"observation", "action_mask"}`, `last()`, `agent_selection`, `rewards`, `terminations`, `truncations`.
  - `legal_actions(state) -> List[Action]`, `legal_action_mask(state) -> np.ndarray(256, bool)`
  - `apply_action(state, action, rng)`, `GameState.new_game(config, n_players, seed)`, `BoardConfig.standard(seed)`
  - `compute_vp(pid, state)`, bots: `pick_action(state, rng) -> Action`.

---

### Task 1: Dependencies (requirements.txt + torch)

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore` (runs/, checkpoints/, __pycache__/, .venv/, .pytest_cache/, .playwright-mcp/)

**Steps:**
- [ ] Write `requirements.txt`: numpy, torch, tensorboard, pyyaml, pytest, pytest-cov
- [ ] `.venv/Scripts/python.exe -m pip install -r requirements.txt` (if torch has no cp314 wheel, pin the newest version that does or document fallback)
- [ ] Verify: `.venv/Scripts/python.exe -c "import torch; print(torch.__version__)"`
- [ ] Commit: `chore: add requirements and gitignore`

### Task 2: Rules profile system (simplified_v1)

**Files:**
- Create: `catan_rl/env/rules_profile.py`, `configs/rules_standard.yaml`, `configs/rules_simplified_v1.yaml`
- Modify: `catan_rl/env/game_state.py` (add `profile` field; `new_game(..., profile=None)`; empty dev deck when disabled), `catan_rl/env/scoring.py` (`check_winner` uses `state.profile.win_vp`), `catan_rl/env/validators.py` (no dev-card buy/play when `not state.profile.dev_cards_enabled`), `catan_rl/env/pettingzoo_env.py` + `gym_wrapper.py` (accept `rules_profile: str | RulesProfile`)
- Test: `tests/test_rules_profile.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) RulesProfile(name: str, dev_cards_enabled: bool = True, win_vp: int = 10)`; `RulesProfile.load(name_or_path) -> RulesProfile` (resolves `configs/rules_<name>.yaml`); `STANDARD`, `SIMPLIFIED_V1` module constants. `GameState.profile` always set (defaults STANDARD, so all existing tests pass unchanged).

**Key tests:** simplified game has empty dev deck; mask never enables slots 230-253 in simplified mode; win_vp=8 profile ends game at 8 VP; standard profile unchanged (existing suite green); profile survives `to_dict`/`from_dict` round-trip (store profile name + fields).

**Steps (TDD):**
- [ ] Write failing tests → run → implement → run full suite → commit `feat: rules profile system with simplified_v1`

### Task 3: Greedy and heuristic bots

**Files:**
- Create: `catan_rl/bots/greedy_bot.py`, `catan_rl/bots/heuristic_bot.py`
- Test: `tests/test_bots.py`

**Interfaces:**
- Produces: `greedy_bot.pick_action(state, rng=None) -> Action`, `heuristic_bot.pick_action(state, rng=None) -> Action` (same signature as `random_bot.pick_action`).
- Shared helper in `heuristic_bot`: `vertex_production_score(config, vertex_id) -> float` using dice-probability weights (PIP counts: 6/8→5 ... 2/12→1).

**Behavior:**
- Greedy: fixed priority among legal actions — BUILD_CITY > BUILD_SETTLEMENT > BUILD_ROAD > BUY_DEV_CARD > play VP card > END_TURN; setup/robber/steal/discard: pick highest-production option (robber: opposing hex with max production not adjacent to self; discard: most-held resource); maritime trade only when it enables an immediate build next priority tier.
- Heuristic: like greedy but chooses settlements/roads by `vertex_production_score` + resource-diversity bonus, roads chosen toward best reachable unowned vertex (2-ply lookahead over road frontier).

**Key tests:** each bot finishes 5 seeded simplified games without crash inside 5000 plies; in 30 simplified games of greedy vs 3 randoms, greedy win rate > 0.4; heuristic beats random similarly; bots only return legal actions (assert membership in `legal_actions(state)` for 200 sampled plies).

- [ ] Failing tests → implement greedy → implement heuristic → suite green → commit `feat: greedy and heuristic scripted bots`

### Task 4: Throughput benchmark

**Files:**
- Create: `scripts/benchmark_throughput.py`
- Test: `tests/test_benchmark.py` (import + tiny run)

**Interfaces:** `python scripts/benchmark_throughput.py --games 50 --profile simplified_v1` prints games/hour, mean turns, plies/sec; exits non-zero if < 500 games/hour.

- [ ] Implement, run with 100 games, record the number in README later; commit `feat: throughput benchmark script`

### Task 5: ActorCritic model with action masking

**Files:**
- Create: `catan_rl/rl/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
```python
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = 256, hidden_sizes=(512, 512)): ...
    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]  # logits (B,256), value (B,)
    def act(self, obs: Tensor, mask: Tensor, deterministic=False) -> tuple[Tensor, Tensor, Tensor]
        # action (B,), logprob (B,), value (B,)
    def evaluate_actions(self, obs, mask, actions) -> tuple[Tensor, Tensor, Tensor]
        # logprob (B,), entropy (B,), value (B,)

def masked_logits(logits: Tensor, mask: Tensor) -> Tensor  # mask==False → -1e9 (not literal -inf, keeps softmax finite)
```
- orthogonal init, tanh activations; value head separate linear on shared trunk.

**Key tests:** shapes; sampling 512 times with a 3-legal-action mask never yields illegal action; `evaluate_actions` logprob matches `act` logprob; entropy ≤ log(n_legal); deterministic act = argmax over legal.

- [ ] TDD cycle → commit `feat: masked actor-critic MLP`

### Task 6: Rollout collection + per-seat GAE

**Files:**
- Create: `catan_rl/rl/rollout.py`
- Test: `tests/test_rollout.py`

**Interfaces:**
- Produces:
```python
@dataclass
class Batch:  # flat tensors over all seats/games
    obs, actions, logprobs, values, advantages, returns, masks  # torch tensors
    stats: dict  # mean_episode_length, win_counts[4], mean_vp, games_completed, truncated_games

def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                gamma: float, lam: float, last_value: float = 0.0) -> tuple[np.ndarray, np.ndarray]
    # returns (advantages, returns); standard GAE over ONE seat's contiguous sub-trajectory

def collect_rollouts(policy: ActorCritic, n_games: int, *, rules_profile="simplified_v1",
                     gamma=0.999, lam=0.95, max_turns=500, seed=None, device="cpu",
                     obs_mode="self_play") -> Batch
```
- Collection loop per game: fresh `CatanAECEnv`, 4 per-seat trajectory lists; each decision point stores (obs, mask, action, logprob, value); reward 0 except final transition per seat = env terminal reward (win +1 / loss -1; truncation → 0, done=True). GAE per seat, then flatten all seats/games. Advantages normalized at batch level in PPO, not here.

**Key tests:** `compute_gae` matches hand-computed values on a 3-step example (gamma=0.5, lam=0.5); every game contributes exactly 4 seat trajectories whose final done=True; winner's final reward +1 and others -1 (seeded game with win_vp=8 profile to finish fast); batch tensor lengths all equal; all stored actions were legal under stored masks.

- [ ] TDD cycle → commit `feat: per-seat rollout collection and GAE`

### Task 7: PPO trainer

**Files:**
- Create: `catan_rl/rl/ppo.py`, `configs/ppo_baseline.yaml`
- Test: `tests/test_ppo.py`

**Interfaces:**
- Produces:
```python
@dataclass
class PPOConfig:
    lr=3e-4; clip_coef=0.2; epochs=4; minibatch_size=256; value_coef=0.5;
    entropy_coef=0.01; max_grad_norm=0.5; gamma=0.999; gae_lambda=0.95;
    hidden_sizes=(512,512); target_kl=None
    @classmethod def from_yaml(path) -> "PPOConfig"

class PPOTrainer:
    def __init__(self, policy: ActorCritic, cfg: PPOConfig, device="cpu")
    def update(self, batch: Batch) -> dict
        # keys: policy_loss, value_loss, entropy, approx_kl, clip_fraction, learning_rate
```
- Clipped surrogate; advantage normalization per minibatch; value loss MSE (optionally clipped); early stop epoch loop if approx_kl > 1.5*target_kl when set.
- `configs/ppo_baseline.yaml` also carries run settings: experiment_name, seed, iterations, games_per_iteration=16, eval_interval, checkpoint_interval, rules_profile, max_turns, reward_win/loss, obs_mode.

**Key tests:** update runs on a synthetic batch and returns all six keys finite; after 20 updates on a fixed synthetic batch where one action has positive advantage, its probability increases; gradients clipped (norm ≤ max_grad_norm + eps).

- [ ] TDD cycle → commit `feat: custom PPO trainer`

### Task 8: Checkpointing + evaluation harness

**Files:**
- Create: `catan_rl/rl/checkpointing.py`, `catan_rl/rl/evaluate.py`
- Test: `tests/test_checkpointing.py`, `tests/test_evaluate.py`

**Interfaces:**
```python
# checkpointing.py
def save_checkpoint(dir, policy, optimizer, iteration, config: dict, metrics: dict) -> Path
    # writes <dir>/ckpt_<iteration:06d>.pt (state_dicts) + sidecar .json metadata
def load_checkpoint(path, policy, optimizer=None) -> dict   # returns metadata
def latest_checkpoint(dir) -> Optional[Path]
def list_checkpoints(dir) -> list[Path]

# evaluate.py
def policy_action(policy, obs_dict, device="cpu", deterministic=True) -> int
def evaluate_vs_bots(policy, bot_pick_action, n_games=20, *, rules_profile, seed, max_turns) -> dict
    # policy on rotating seat vs 3 bot seats → {"win_rate": float, "mean_vp": float, "mean_turns": float}
def evaluate_vs_checkpoint(policy, ckpt_path, n_games=20, ...) -> dict  # current (2 seats) vs old (2 seats)
```
**Key tests:** save→load round-trips weights exactly (allclose on a forward pass); metadata json contains iteration/config/metrics; `evaluate_vs_bots(random policy, random_bot)` returns win_rate in [0,1] over 4 tiny games; seat rotation covers all 4 seats.

- [ ] TDD cycle → commit `feat: checkpointing and evaluation harness`

### Task 9: Self-play orchestrator + training script

**Files:**
- Create: `catan_rl/rl/self_play.py`, `scripts/train_self_play.py`
- Test: `tests/test_self_play.py`

**Interfaces:**
```python
class SelfPlayTrainer:
    def __init__(self, cfg_path_or_dict, run_dir=None, device=None)  # seeds torch/np/random
    def train(self, iterations=None) -> None
    # per iteration: collect_rollouts → trainer.update → SummaryWriter scalars:
    #   train/{policy_loss,value_loss,entropy,approx_kl,clip_fraction,learning_rate}
    #   game/{mean_episode_length,win_rate_seat0..3,mean_vp_at_end,games_completed}
    #   eval/{win_rate_vs_random,win_rate_vs_greedy,win_rate_vs_prev_checkpoint} at eval_interval
    # checkpoint at checkpoint_interval into runs/<experiment_name>/checkpoints/
```
- `scripts/train_self_play.py --config configs/ppo_baseline.yaml [--iterations N] [--device cpu|cuda] [--resume]`

**Key tests:** end-to-end tiny run (hidden 64,64; 2 games/iter; 2 iterations; win_vp=8 simplified) completes < 120 s, writes a checkpoint file and TB event file containing `train/policy_loss` (read back with `tensorboard.backend.event_processing.event_accumulator`).

- [ ] TDD cycle → commit `feat: self-play trainer and train script`

### Task 10: Evaluation + render scripts

**Files:**
- Create: `scripts/evaluate_checkpoints.py`, `scripts/render_match.py`

**Interfaces:**
- `evaluate_checkpoints.py --run runs/<name> [--games 50] [--vs random,greedy,heuristic,prev]` → table of win rates per checkpoint.
- `render_match.py [--ckpt path] [--bots random,greedy,...] --seed N` → text-renders a full game turn by turn using `env.render()`.

- [ ] Implement, smoke-run both, commit `feat: checkpoint evaluation and match render scripts`

### Task 11: Docs — README + gcloud guide

**Files:**
- Create: `README.md`, `docs/RUN_ON_GCLOUD.md`

**README:** project layout, install, run tests, play random game, benchmark numbers, train, monitor with TensorBoard, evaluate.
**RUN_ON_GCLOUD.md (user-requested deliverable):** creating a Compute Engine VM (CPU e2-standard-8 default; optional GPU T4/L4 with driver install), gcloud CLI auth, transferring the repo (git or `gcloud compute scp`), venv + requirements (CPU torch wheel index note), running training under `tmux`/`nohup`, TensorBoard via SSH tunnel (`gcloud compute ssh ... -- -L 6006:localhost:6006`), pulling checkpoints back, Spot VM + cost guidance, teardown.

- [ ] Write both docs; verify commands against actual script flags; commit `docs: README and gcloud training guide`

### Task 12: Final verification + baseline training smoke

- [ ] Full suite: `.venv/Scripts/python.exe -m pytest tests/ -q` — all green
- [ ] `scripts/benchmark_throughput.py --games 100` — record games/hour (must be ≥ 500)
- [ ] Short real training run (~20 iterations, simplified_v1, win_vp=8) — confirm losses move, TB logs populate, checkpoints save; run `evaluate_checkpoints.py` on the result
- [ ] Update README with measured numbers; final commit `chore: final verification pass`

## Self-Review Notes

- Spec coverage: Phases 1–2 pre-exist; Tasks 2–4 cover Phase 3 (profiles, bots, throughput); Tasks 5–9 cover Phase 4 (shared-policy PPO, per-seat GAE, TB metrics, checkpoints, eval); Task 10–11 cover evaluation scripts + docs; Phase 5+ (dev-card hiding beyond current, recurrent policy, trading) is explicitly staged later by the spec and out of scope for v0.1.
- Deviation from skill template: implementation code for large modules is specified by exact interface + behavior rather than fully inlined, because the plan author and executor are the same session (autonomous run); critical algorithms (GAE, masking, PPO losses, per-seat bookkeeping) are pinned above.
- Type consistency: bots share `pick_action(state, rng)`; `Batch` is produced by Task 6 and consumed by Tasks 7/9; `RulesProfile` threaded via `state.profile` everywhere.
