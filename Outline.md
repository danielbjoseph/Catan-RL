# Catan Self-Play RL System Specification

## Purpose

This document is a build specification for an AI coding agent. The goal is to create a full reinforcement-learning training system for *Settlers of Catan*-style gameplay in which four agents play the game at once, but all four are instances of the **same improving policy**. The system should be built in phases so that the game environment, rule engine, observation model, training loop, and evaluation harness are developed in an orderly way.

This is not meant to be a toy demo. It should be structured so that we can start with a simplified and trainable version, then progressively add hidden information, trading, development cards, and stronger belief/opponent modeling.

---

# 0. Tech stack

The following concrete choices must be used throughout the project. Do not substitute alternatives without updating this section first.

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| Deep learning | PyTorch (latest stable) |
| RL algorithm | Custom PPO — do **not** use stable-baselines3; we need full control over multi-seat trajectory batching |
| Multi-agent env API | PettingZoo AEC (Agent Environment Cycle) |
| Gym compatibility | `gymnasium` (not legacy `gym`) |
| Logging / monitoring | TensorBoard via `torch.utils.tensorboard.SummaryWriter` |
| Config management | YAML files loaded via PyYAML |
| Testing | `pytest` with `pytest-cov` for coverage reporting |
| Serialization | Python `dataclasses` + JSON; no unsafe serialization for game state |
| Package management | `pip` with a `requirements.txt`; all installs into the project venv |

---

# 1. High-level objective

We want a system that can:

1. Simulate a full four-player Catan game.
2. Expose the game in a Gym-compatible environment interface.
3. Support masked legal actions.
4. Support multi-agent self-play.
5. Train a single shared policy from four simultaneous seats.
6. Improve over time via self-play rather than hand-coded strategy.
7. Eventually support imperfect information and partially observable observations.
8. Remain modular enough that rules can be simplified early, then expanded.

The intended end state is:

* one environment
* one shared policy network
* four players per game
* self-play training
* periodic evaluation against past checkpoints and scripted baselines

---

# 2. Design philosophy

The coding agent should follow these principles.

## 2.1 Build for staged complexity

Do **not** try to implement “full perfect Catan with optimal trading and belief modeling” in the first pass. That is too much complexity at once and will make training unstable and debugging painful.

Instead, build in layers:

* Phase 1: deterministic, rules-correct environment core
* Phase 2: Gym-compatible interface + legal action masks
* Phase 3: simplified RL training version
* Phase 4: self-play with shared policy
* Phase 5: richer hidden information and full rules
* Phase 6: scaling, evaluation, and improvement

## 2.2 Favor trainability over purity

A theoretically pure raw-state end-to-end solution is less important than a system that can actually learn. Use structured observations, action masks, and possibly a few summary features if needed. This is acceptable.

## 2.3 Separate game logic from learning code

The environment and rule engine must be completely separable from the RL algorithm. We should be able to:

* unit test the rules without neural nets
* run random or scripted agents
* swap RL algorithms later

## 2.4 Support reproducibility

Everything should be seedable and logged:

* environment seed
* board seed
* dice rolls
* dev deck shuffle
* training seed
* checkpoint metadata

---

# 3. Recommended phased roadmap

---

## Phase 1 — Build the Catan game engine

### Goal

Create a fully functional Catan engine that can simulate games correctly without any RL.

### Deliverables

* core board representation
* full turn logic
* player state tracking
* resource production
* robber logic
* build legality checks
* scoring
* winner detection
* dev cards
* longest road
* largest army
* deterministic test mode

### Requirements

The engine should support standard Catan rules or a close, well-documented variant.

At minimum, include:

* hex tiles with resource types
* number tokens
* robber position
* vertices for settlements/cities
* edges for roads
* ports
* bank resources
* development deck
* per-player resources
* per-player dev cards
* public victory points
* hidden victory points from dev cards
* turn order
* dice rolling
* discard-on-7
* steal-on-robber
* road/settlement/city/dev-card purchase logic

### Internal architecture

Recommended modules:

* `board.py`
* `rules.py`
* `player_state.py`
* `game_state.py`
* `actions.py`
* `validators.py`
* `scoring.py`
* `setup.py`

### Data model recommendations

Use explicit graph-based representations:

* vertices indexed by integer
* edges indexed by integer
* hexes indexed by integer
* lookup maps:

  * hex -> adjacent vertices
  * hex -> adjacent edges
  * vertex -> adjacent hexes
  * vertex -> neighboring vertices
  * edge -> endpoint vertices

This should be fully inspectable and serializable.

### Action system

Represent actions explicitly, not as free-form dicts floating around the codebase.

Use a structured action object or dataclass such as:

* `ROLL_DICE`
* `END_TURN`
* `BUILD_ROAD(edge_id)`
* `BUILD_SETTLEMENT(vertex_id)`
* `BUILD_CITY(vertex_id)`
* `BUY_DEV_CARD`
* `PLAY_KNIGHT(target_hex, target_player?)`
* `PLAY_ROAD_BUILDING(edge_a, edge_b)`
* `PLAY_YEAR_OF_PLENTY(resource_a, resource_b)`
* `PLAY_MONOPOLY(resource_type)`
* `PROPOSE_TRADE(...)`
* `ACCEPT_TRADE`
* `DECLINE_TRADE`
* `MARITIME_TRADE(...)`
* `DISCARD_RESOURCES(...)`
* `CHOOSE_STEAL_TARGET(player_id)`

### Must-have tests

The coding agent should create extensive tests for:

* legal settlement distance rule
* legal road connectivity
* city upgrades only on owned settlement
* resource production after dice roll
* robber blocks production
* discard logic when 7 rolled
* knight movement + steal
* longest road correctness in branching cases
* largest army assignment
* dev card draw/use restrictions
* win condition
* setup phase rules

### Acceptance criteria

Phase 1 is complete when:

* two random agents can play a complete game without crashing
* the rules appear correct in logged traces
* unit tests cover critical legality/scoring logic
* game state can be serialized/deserialized

---

## Phase 2 — Wrap the game in a Gym-compatible multi-agent environment

### Goal

Expose the engine as an environment suitable for RL.

### Important note

Standard old Gym is single-agent-oriented. This project is multi-agent. The implementation should still be “Gym-compatible in spirit,” but it should likely follow one of these two patterns:

1. **Turn-based single-active-agent API**, where the env exposes only the current player’s turn and rotates across players.
2. **PettingZoo-style AEC or parallel API**, which is better suited for multi-agent games.

### Recommendation

Implement the environment in a **PettingZoo-style turn-based API**, while keeping a thin Gym-compatible wrapper for current-player training. This is the cleanest path.

### Deliverables

* environment reset
* observation generation
* legal action mask
* turn progression
* reward emission
* terminal detection
* render/log utility

### Observation design

The environment should support multiple observation modes:

#### Mode A: perfect-information debug observation

Used for validation and early experiments.

Includes:

* full board state
* all players’ exact resources
* all dev cards
* bank counts
* current phase
* legal action mask

#### Mode B: self-play training observation

Used for real RL.

Includes:

* full public board state
* exact self-hand
* exact self-dev cards
* public counts for opponents
* public played dev cards
* current turn/phase
* legal action mask
* optionally simple belief features later

#### Mode C: feature-rich training observation

A superset of Mode B with optional engineered summaries:

* expected production per player
* resource scarcity
* road lengths
* settlement spots available
* discard risk

### Action space design

The action space is a **fixed-size flat catalog of 256 slots**. Every possible (action_type, parameter) combination is assigned a permanent index in this catalog. The policy network always outputs 256 logits. Illegal actions are masked to `-inf` before softmax. Simplified-mode games just have more slots masked.

#### Catalog breakdown (standard board: 54 vertices, 72 edges, 19 hexes, 5 resource types, 4 players)

| Action | Slots | Notes |
|---|---|---|
| `ROLL_DICE` | 1 | |
| `END_TURN` | 1 | |
| `BUILD_ROAD(edge_id)` | 72 | one slot per edge |
| `BUILD_SETTLEMENT(vertex_id)` | 54 | one slot per vertex |
| `BUILD_CITY(vertex_id)` | 54 | one slot per vertex |
| `MOVE_ROBBER(hex_id)` | 19 | shared between 7-roll and knight card |
| `CHOOSE_STEAL_TARGET(player_id)` | 4 | one slot per seat; masked to players on target hex |
| `MARITIME_TRADE(give, get)` | 20 | 5 × 4 (no same-resource trade); rate inferred from ports |
| `DISCARD_RESOURCE(resource)` | 5 | called once per resource unit discarded |
| `BUY_DEV_CARD` | 1 | |
| `PLAY_KNIGHT` | 1 | triggers MOVE_ROBBER sub-phase |
| `PLAY_ROAD_BUILDING` | 1 | triggers two BUILD_ROAD sub-phases |
| `PLAY_YEAR_OF_PLENTY(res_a, res_b)` | 15 | unordered pairs with repetition: C(5+1,2) = 15 |
| `PLAY_MONOPOLY(resource)` | 5 | |
| `PLAY_VICTORY_POINT` | 1 | auto-triggered at win; masked otherwise |
| *(padding to power of 2)* | 3 | reserved |
| **Total** | **256** | |

The catalog index for each action is a constant defined in `actions.py` at project init. The policy head is always `nn.Linear(hidden_dim, 256)`.

#### Sub-phases and re-used slots

Some actions (PLAY_KNIGHT, PLAY_ROAD_BUILDING) put the game into a sub-phase where a different set of slots becomes legal. The environment tracks the current sub-phase in its state and applies the correct mask automatically. The network sees the same 256-slot output regardless of sub-phase; only the mask differs.

This is much easier than trying to encode the entire combinatorial action space in a universal fixed discrete space from day one.

### Reward design

Primary reward:

* +1 for game win
* 0 or -1 for loss depending on algorithm choice

Optional intermediate shaping for early training only:

* small reward for gaining VP
* small reward for legal build progress
* small reward for longest road / largest army acquisition

But the system must support turning shaping off later.

### Acceptance criteria

Phase 2 is complete when:

* environment can run many games in sequence
* random and scripted agents can act via the env API
* legal action masks prevent impossible moves
* observations are consistent with game state
* terminal outcomes and rewards are correct

---

## Phase 3 — Create a simplified training version of Catan

### Goal

Make the first learnable version before full complexity.

### Why this matters

Full Catan with hidden information, dev cards, and open negotiation is probably too hard as the initial learning target. We need a curriculum.

### Recommended simplified ruleset v1

The coding agent should create a configurable ruleset flag called something like `rules_profile="simplified_v1"`.

Suggested simplified v1 features:

* standard board geometry
* standard dice/resource generation
* no player-to-player trade
* allow maritime/bank trades only
* optionally disable dev cards initially
* keep robber
* keep roads/settlements/cities
* keep longest road
* optionally disable largest army if dev cards are off
* keep win condition at 10 VP, or reduce to 8 VP for faster training experiments

### Rationale

This preserves:

* stochastic production
* spatial expansion
* blocking
* resource management
* race dynamics

while removing:

* negotiation complexity
* most hidden-information issues
* some large action branching

### Deliverables

* rules profile system
* simplified environment mode
* scripted baseline bots for simplified mode

### Scripted baseline bots

Build at least:

* random legal bot
* greedy build bot
* heuristic expansion bot
* ore-wheat-city bot if feasible

These are crucial for:

* debugging
* sanity-checking learning progress
* future evaluation benchmarks

### Acceptance criteria

Phase 3 is complete when:

* simplified games run end-to-end
* baseline bots produce sensible strategies
* training can begin on a reduced-complexity game

---

## Phase 4 — Train a single shared self-play policy

### Goal

Run four copies of the same policy in each game and train that one policy from all player experiences.

### Core training concept

Each seat in the 4-player game is controlled by the same policy network. This means:

* Player 0, 1, 2, 3 all use the same weights
* They differ only by observation
* The policy improves from pooled experience across all seats

This is the correct setup for “four agents play each other, but ultimately improve one policy.”

### Why this is desirable

It creates:

* symmetric self-play
* large sample generation
* reduced maintenance complexity
* natural curriculum as the policy improves against itself

### Recommended RL algorithm

For the early simplified and partially observed setting, the best practical first choice is:

## **PPO with action masking**, possibly with recurrent policy support later

Why PPO first:

* stable and widely used
* easy to implement/debug
* works well with self-play
* handles on-policy updates cleanly
* action masking is common and practical

Why not DQN:

* action space is too structured and variable
* multi-agent self-play is less natural here
* partial observability hurts plain value-based methods

Why not pure AlphaZero/MCTS first:

* hidden information and long combinatorial action space make it significantly harder
* implementation overhead is much larger
* not needed for first useful results

Why not population-based MARL from day one:

* too much complexity too early

### Architecture recommendation

#### Initial architecture

* MLP over structured flat/vectorized observation

#### Better architecture after first success

* graph encoder for board
* player feature encoder
* concatenated self-state + public-state embedding
* policy head
* value head

#### For imperfect information later

* recurrent layer or transformer memory
* optional belief auxiliary head

### Training loop requirements

The coding agent should implement:

* batched self-play rollouts
* four-seat experience collection
* shared-policy PPO update
* checkpointing
* evaluation against prior checkpoints
* Elo-like rating or head-to-head win rate tracking

### Experience collection detail

Important: when a game is running, only one player acts at a time. The training system must still correctly attribute transitions to the acting player/seat.

Each acting turn should store:

* observation
* legal action mask
* selected action
* logprob
* reward
* done
* value estimate
* seat id
* policy version
* episode id

At episode end, assign final outcome to all seat trajectories appropriately.

### GAE and advantage estimation across interleaved turns (critical implementation note)

In a 4-player game, each seat's trajectory is **not contiguous** — other players act between each of a given seat's turns. This creates a subtle bug if GAE is applied naively across the full game sequence.

The correct approach:

1. **Collect per-seat sub-trajectories separately.** Each seat accumulates its own list of `(obs, action, logprob, value, reward, done)` tuples, appended only when that seat acts.
2. **GAE is computed independently per seat** after the episode ends, treating each seat's sub-trajectory as its own standalone sequence. There are no "gap" timesteps between a seat's entries.
3. **Terminal bootstrapping:** at the end of the episode (game over), `done=True` for the acting seat's final transition. All other seats' trajectories also receive `done=True` on their last stored transition, with their final reward set to the game outcome for that seat.
4. **Do not interleave seats' transitions** into one shared time series before running GAE.

This means the rollout buffer holds four independent trajectory lists per episode, each of variable length (≈ total turns / 4). All four are flattened together for the PPO minibatch update after GAE is computed per-seat.

### Reward recommendation for shared-policy multiplayer

Use a per-seat episodic reward:

* winner gets +1
* all others get -1/3 or 0 depending on desired normalization

I recommend:

* winner = +1
* losers = -1

for strong signal, but make it configurable.

Potential alternative:

* rank-based rewards
* VP-differential shaping

These can be experimented with later.

### Self-play stabilization features

The coding agent should include:

* checkpoint pool for opponents
* evaluation matches against older snapshots
* optional opponent sampling from recent checkpoint history

Even though the main system is one shared policy, evaluation against old checkpoints is important to detect regressions.

### Game throughput target

The environment must be fast enough for RL to be practical. Target: **≥ 500 complete simplified-rule games per hour on CPU** (no GPU required for environment stepping). Measure this with a random-agent benchmark before starting PPO training. If throughput is below target, profile and optimize the engine before adding RL overhead.

### TensorBoard logging requirements

The trainer must log the following scalars to TensorBoard at each update step:

**Training metrics:**
* `train/policy_loss`
* `train/value_loss`
* `train/entropy`
* `train/approx_kl`
* `train/clip_fraction`
* `train/learning_rate`

**Game metrics (logged per rollout batch):**
* `game/mean_episode_length`
* `game/win_rate_seat0` through `seat3`
* `game/mean_vp_at_end`
* `game/games_completed`

**Evaluation metrics (logged per eval interval):**
* `eval/win_rate_vs_random`
* `eval/win_rate_vs_greedy`
* `eval/win_rate_vs_prev_checkpoint`

Use `torch.utils.tensorboard.SummaryWriter`. Write logs to `runs/<experiment_name>/`. Launch with `tensorboard --logdir runs/`.

### Acceptance criteria

Phase 4 is complete when:

* four-seat self-play runs
* one shared policy is updated from all seats
* all required TensorBoard metrics are logging
* environment achieves ≥ 500 games/hour throughput target
* policy beats random and simple scripted bots consistently in simplified mode

---

## Phase 5 — Add the hard parts incrementally

### Goal

Expand from simplified Catan toward richer, more realistic play.

This should happen one major complexity source at a time.

---

### Phase 5A — Development cards

Add:

* buy dev card
* all dev card effects
* hidden dev card ownership
* no same-turn use for newly purchased card
* largest army

Observation changes:

* self exact dev cards
* opponent dev card counts
* public played dev cards

---

### Phase 5B — Partial observability and belief-aware training

Move from simplified public-ish state toward real hidden information.

Add:

* exact self resources
* opponent resource counts only, not compositions
* self dev cards exact
* opponent dev card counts only
* uncertainty from robber steals and trades

Recommended upgrade:

* recurrent PPO or transformer-based policy memory

Optional auxiliary heads:

* predict opponent resource distributions
* predict opponent dev card type probabilities
* predict next legal build likelihood

This is useful because the environment is now clearly a POMDP.

---

### Phase 5C — Maritime strategy refinement and ports

Strengthen valuation of:

* port access
* maritime trades
* scarcity handling

May require:

* feature additions
* more training time
* better scripted evaluation bots

---

### Phase 5D — Player-to-player trading

This is the hardest step and should not be added before the rest is working.

#### Recommendation

Do not begin with free-form bargaining language or open-ended trading.

Instead, implement **bounded structured trade actions**.

Example action families:

* offer 1-for-1 resource trade
* offer 2-for-1
* accept/decline specific trade
* cap number of trade offers per turn

This keeps the action space manageable.

#### Further recommendation

Trading should likely be a separate subproject with its own curriculum:

1. no trade
2. bank-only trade
3. limited bilateral trade templates
4. richer trade space if needed

### Acceptance criteria

Phase 5 is complete when:

* richer rules are added without destabilizing the codebase
* training still runs
* policy performance improves with added complexity
* hidden information is respected in training observations

---

## Phase 6 — Evaluation, scaling, and stronger methods

### Goal

Turn the prototype into a serious experimental platform.

### Evaluation suite

The coding agent should implement:

* win rate vs random bot
* win rate vs greedy bot
* win rate vs earlier checkpoints
* seat-position performance
* average game length
* VP progression curves
* build frequencies
* resource hoarding metrics
* road/settlement/city composition stats
* longest road / largest army rates

### Important diagnostics

Track:

* invalid action attempts before masking
* entropy over policy
* average legal-action count
* value prediction error
* policy collapse indicators
* reward variance
* first-settlement placement quality
* resignation/stalling pathologies if any

### Scaling improvements

Potential later upgrades:

* parallel environment workers
* GPU batching
* distributed rollout collection
* larger board encoders
* GNN architecture
* recurrent memory
* population-based opponent sets
* league training
* NFSP-style trade modeling if trading becomes central

### Stronger algorithm directions later

After PPO baseline success, possible later explorations:

#### Recurrent PPO

Best next step for hidden information.

#### IMPALA / APPO

Useful if scaling to many parallel rollouts.

#### MuZero-style or search-assisted planning

Potentially very strong, but much heavier lift and less appropriate initially.

#### Policy + belief auxiliary learning

Very attractive for Catan because hidden state inference matters.

---

# 4. Detailed environment specification

## 4.1 Core environment interface

The environment should expose methods similar to:

* `reset(seed=None, rules_profile="...")`
* `step(action)`
* `observe(player_id)`
* `legal_actions(player_id)`
* `render(mode="text" | "json")`

If PettingZoo-like:

* `agent_selection`
* `rewards`
* `terminations`
* `truncations`
* `infos`
* `observe(agent)`

## 4.2 Observation contents

### Public board channels

* hex resource types
* hex dice values
* robber position
* ports
* road ownership
* settlement ownership
* city ownership

### Per-player public features

For each player:

* public VP
* hand size
* dev card count
* roads left
* settlements left
* cities left
* longest road holder flag
* largest army holder flag

### Self-private features

For the observing player only:

* exact resource counts by type
* exact dev card counts by type

### Turn context

* current player index
* current phase
* dice rolled this turn or none
* trade subphase flag
* discard required flag
* must move robber flag
* must choose steal target flag

### Optional engineered summaries

Configurable on/off:

* expected production by resource
* expected production by player
* discard risk
* reachable legal settlement count
* current longest-road length estimate

## 4.3 Action mask

Legal action masking is mandatory.

The environment must be the source of truth for legality. The policy should never be responsible for inferring legality from scratch.

The mask should:

* block impossible actions
* block illegal parameter combinations
* update every turn
* be included in observation or info dict

---

# 5. Recommended neural-network architecture

## Initial baseline network

For the first working trainer:

* input: flattened structured observation vector
* body: 2–4 fully connected layers with nonlinearities
* outputs:

  * masked policy logits
  * scalar value

This is enough to validate pipeline correctness.

## Recommended stronger network

Once baseline training works:

### Board encoder

Encode graph/board structure using:

* GNN over vertices/edges/hexes, or
* structured learned embeddings with adjacency-aware aggregation

### Player encoder

Encode each player as a feature vector.

### Self/public split

Treat self-private info distinctly from public opponent info.

### Heads

* policy head
* value head
* optional belief head

### Belief head

Optional auxiliary prediction targets:

* estimated opponent resource distributions
* opponent dev card probabilities
* next-build-type likelihood

This can improve hidden-information reasoning.

## Recurrent upgrade

When hidden information becomes serious, use:

* LSTM or GRU after encoded observation
* or small transformer with turn history window

This helps infer hidden state from observed history.

---

# 6. Self-play training design

## Core approach

Each training game contains four seats:

* seat 0
* seat 1
* seat 2
* seat 3

All use the same policy weights.

At each decision point:

* current seat receives observation
* shared policy outputs masked action distribution
* sampled action is applied
* transition is stored for that seat

After many games:

* pool all seat trajectories
* run PPO update
* checkpoint new policy

## Important detail

Although all four seats share a policy, trajectory bookkeeping must still preserve which seat saw which observation and got which reward. The policy is shared; the experiences are not identical.

## Exploration

Use stochastic action sampling during training. Greedy play only for evaluation.

## Opponent diversity

Even in a shared-policy setting, include optional matches where one or more seats use older checkpoints. This reduces cyclic overfitting.

---

# 7. Suggested repository structure

```text
catan_rl/
│
├── env/
│   ├── board.py
│   ├── game_state.py
│   ├── player_state.py
│   ├── rules.py
│   ├── actions.py
│   ├── validators.py
│   ├── scoring.py
│   ├── setup.py
│   ├── observation.py
│   ├── action_mask.py
│   ├── pettingzoo_env.py
│   └── gym_wrapper.py
│
├── bots/
│   ├── random_bot.py
│   ├── greedy_bot.py
│   ├── heuristic_bot.py
│   └── rule_based_city_bot.py
│
├── rl/
│   ├── models.py
│   ├── ppo.py
│   ├── rollout.py
│   ├── replay_utils.py
│   ├── self_play.py
│   ├── evaluate.py
│   └── checkpointing.py
│
├── configs/
│   ├── rules_simplified_v1.yaml
│   ├── rules_standard.yaml
│   ├── ppo_baseline.yaml
│   └── ppo_recurrent.yaml
│
├── tests/
│   ├── test_rules.py
│   ├── test_setup.py
│   ├── test_scoring.py
│   ├── test_observation.py
│   ├── test_action_mask.py
│   └── test_env_smoke.py
│
├── scripts/
│   ├── play_random_game.py
│   ├── train_self_play.py
│   ├── evaluate_checkpoints.py
│   └── render_match.py
│
└── README.md
```

---

# 8. Development order the coding agent should follow

The coding agent should proceed in this order.

## Step 1

Implement board topology and immutable board metadata.

## Step 2

Implement mutable game state and player state.

## Step 3

Implement setup phase logic.

## Step 4

Implement turn progression and dice/resource production.

## Step 5

Implement roads, settlements, cities, robber, and scoring.

## Step 6

Implement dev cards and special awards.

## Step 7

Add legal action generation and validation.

## Step 8

Add observation generator.

## Step 9

Add legal action masks.

## Step 10

Wrap as PettingZoo-style env and thin Gym wrapper.

## Step 11

Create scripted bots and smoke-test many games.

## Step 12

Create simplified rules profile.

## Step 13

Build PPO self-play trainer with shared policy.

## Step 14

Run baseline training against simplified rules.

## Step 15

Evaluate against bots and past checkpoints.

## Step 16

Add partial observability and recurrent policy.

## Step 17

Add dev cards if deferred earlier.

## Step 18

Add structured player trading.

---

# 9. Known hard problems and how to handle them

## 9.1 Longest road correctness

This is tricky because branching road networks are not trivial. The coding agent should isolate longest-road computation into its own tested module and include difficult graph test cases.

## 9.2 Action-space explosion

Do not encode all possible actions globally at first. Generate legal actions dynamically and mask.

## 9.3 Hidden information

Do not fully solve belief modeling immediately. Start with self-private/public-opponent observation and add recurrent memory later.

## 9.4 Trading

Do not start with unrestricted trading. Use bounded templates later.

## 9.5 Reward sparsity

Start with simplified mode and possibly light shaping, but make shaping easy to disable.

## 9.6 Self-play instability

Evaluate against older checkpoints and scripted bots, not just current mirror matches.

---

# 10. Success milestones

## Milestone A

A fully rules-correct Catan engine can run random games to completion.

## Milestone B

A Gym/PettingZoo-style environment exposes observations and legal action masks correctly.

## Milestone C

Simplified-rule PPO self-play trains without crashing.

## Milestone D

Shared policy beats random and greedy baselines.

## Milestone E

Recurrent or richer policy handles hidden information better than feedforward baseline.

## Milestone F

Standard-rule version with dev cards becomes trainable.

## Milestone G

Structured trading is added without blowing up training.

---

# 11. Specific instruction to the coding agent on algorithm choice

Use **PPO with legal action masking** as the first training algorithm.

Then, once hidden information matters more, move to **recurrent PPO**.

Do not start with:

* DQN
* AlphaZero
* MuZero
* free-form language-based trade negotiation
* population MARL frameworks unless baseline PPO already works

The reason is not that those methods are bad. It is that they are too costly and fragile for the first build.

---

# 12. Final intended end-state

The mature version of the project should support:

* four-player Catan self-play
* one shared policy improving through self-play
* hidden information respected in observations
* action masking
* checkpoint-based evaluation
* configurable rulesets
* eventual stronger architecture for board graphs and memory

In other words, the project should evolve from:

**rules engine -> RL environment -> simplified self-play PPO -> recurrent hidden-info self-play -> fuller Catan**

rather than trying to leap directly to the hardest version.

---

# 13. Concrete first implementation target

The coding agent’s first serious target should be this:

## Version 0.1

* simplified Catan
* no player trading
* no dev cards
* robber included
* roads/settlements/cities included
* longest road included
* PettingZoo-style turn-based env
* legal action mask
* PPO shared-policy self-play
* four seats per game
* evaluation against random and greedy bots

If Version 0.1 works, the project is on the right track.

---

# 14. Non-goals for the first implementation

Do not prioritize these initially:

* fancy graphics/UI
* web deployment
* human play interface
* language-model negotiation
* perfect opponent belief tracking
* massive distributed training infrastructure
* search-based planning hybrid

Those can come later.

---

# 15. Closing implementation guidance

The coding agent should optimize for:

* correctness first
* modularity second
* trainability third
* sophistication fourth

A correct, testable, simplified environment with stable PPO self-play is far more valuable than an overambitious, half-working “full Catan MARL system.”

The right build path is incremental, measurable, and benchmarked at every step.

Build it in a GitHub repo. Commit frequently. Comment frequently. Add branches for significant features. 