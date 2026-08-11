"""
Self-play rollout collection with per-seat trajectories and GAE.

Critical detail (spec Phase 4): in a 4-player game each seat's trajectory is
NOT contiguous — other seats act between a given seat's turns. Transitions are
therefore accumulated in four separate per-seat lists and GAE is computed
independently per seat AFTER the episode ends, treating each seat's
sub-trajectory as its own standalone sequence. The four lists are only
flattened together for the PPO minibatch update.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ..bots.personalities import PERSONALITIES, make_personality_bot
from ..env.action_mask import legal_action_mask
from ..env.actions import CATALOG, CATALOG_SIZE
from ..env.observation import make_observation
from ..env.pettingzoo_env import CatanAECEnv
from ..env.rules_profile import RulesProfile
from ..env.scoring import compute_vp
from ..env.trace import TraceRecorder
from .checkpointing import load_policy
from .models import ActorCritic, act_prefix_sliced

# Seat/opponent draws are seeded from `game_seed ^ _POOL_SEED_XOR`, kept
# independent of the action RNG (`env._rng`, seeded from `game_seed` itself
# a level down inside CatanAECEnv.reset).
_POOL_SEED_XOR = 0x5EED


@dataclass
class Batch:
    """Flattened training batch pooled over all seats and games."""

    obs: torch.Tensor         # (N, obs_dim) float32
    masks: torch.Tensor       # (N, CATALOG_SIZE) bool
    actions: torch.Tensor     # (N,) long
    logprobs: torch.Tensor    # (N,) float32
    values: torch.Tensor      # (N,) float32
    advantages: torch.Tensor  # (N,) float32
    returns: torch.Tensor     # (N,) float32
    seat_ids: torch.Tensor    # (N,) long
    episode_ids: torch.Tensor # (N,) long
    stats: Dict

    def __len__(self) -> int:
        return self.obs.shape[0]


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard GAE over ONE contiguous (per-seat) trajectory.

    dones[t] == 1 means no bootstrapping past step t.
    Returns (advantages, returns) with returns = advantages + values.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(n)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages, advantages + values


class _SeatTrajectory:
    """Accumulates one seat's transitions within a single episode."""

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.actions: List[int] = []
        self.logprobs: List[float] = []
        self.values: List[float] = []

    def __len__(self) -> int:
        return len(self.actions)


@dataclass
class _PoolEntry:
    """A resolved (pre-loaded) opponent-pool entry.

    type == "personality": `bot_fn` is a `pick_action(state, rng) -> Action`
    closure from `make_personality_bot`.
    type == "checkpoint": `policy` is a loaded `ActorCritic`, `obs_mode` is
    its own stored observation mode (may differ from the env's).
    type == "self": acts with the live policy being trained, stochastically;
    its transitions are never recorded.
    """

    type: str
    weight: float
    label: str
    bot_fn: Optional[Callable] = None
    policy: Optional[ActorCritic] = None
    obs_mode: Optional[str] = None


def _build_pool(pool_spec: List[Dict], env_obs_mode: str, device: str) -> List[_PoolEntry]:
    """Resolve a raw `opponents["pool"]` spec into `_PoolEntry` objects once,
    up front, so per-game seat draws are just a weighted choice over an
    already-loaded list (no checkpoint I/O or personality lookups in the
    hot per-game loop)."""
    if not pool_spec:
        raise ValueError("opponents['pool'] must be non-empty")

    entries: List[_PoolEntry] = []
    for spec in pool_spec:
        kind = spec["type"]
        weight = float(spec.get("weight", 1.0))

        if kind == "personality":
            name = spec["name"]
            if name not in PERSONALITIES:
                raise ValueError(f"unknown personality {name!r}")
            entries.append(_PoolEntry(
                type="personality", weight=weight, label=f"personality:{name}",
                bot_fn=make_personality_bot(PERSONALITIES[name]),
            ))
        elif kind == "checkpoint":
            path = spec["path"]
            ckpt_policy, meta = load_policy(path, device=device)
            ckpt_obs_mode = meta.get("obs_mode", "self_play")
            if ckpt_obs_mode == "realistic" and env_obs_mode != "realistic":
                raise ValueError(
                    f"checkpoint {path!r} was trained with obs_mode='realistic', "
                    f"which requires sharing the env's belief tracker; the env "
                    f"obs_mode here is {env_obs_mode!r}, not 'realistic'."
                )
            ckpt_policy.eval()
            entries.append(_PoolEntry(
                type="checkpoint", weight=weight,
                label=f"checkpoint:{Path(path).stem}",
                policy=ckpt_policy, obs_mode=ckpt_obs_mode,
            ))
        elif kind == "self":
            entries.append(_PoolEntry(type="self", weight=weight, label="self"))
        else:
            raise ValueError(f"unknown opponent pool entry type {kind!r}")

    return entries


def _observe_as(env: CatanAECEnv, pid: int, mode: str) -> np.ndarray:
    """Build an observation for `pid` in an arbitrary `mode`, independent of
    the env's own `obs_mode` (used for checkpoint opponents in the pool)."""
    if mode == "realistic":
        noise_cfg = {
            "belief_blend": env.belief_blend,
            "belief_noise": env.belief_noise,
            "seed": env._seed,
        }
        return make_observation(
            env._state, observer=pid, mode=mode, belief=env._belief, noise_cfg=noise_cfg,
        )
    return make_observation(env._state, observer=pid, mode=mode)


def _act_opponent(
    env: CatanAECEnv,
    entry: _PoolEntry,
    policy: ActorCritic,
    rng: random.Random,
    device: str,
) -> int:
    """Return a catalog action index for a non-policy (pool) seat."""
    if entry.type == "personality":
        return entry.bot_fn(env._state, rng).catalog_index
    if entry.type == "checkpoint":
        pid = env._state.current_player
        obs = _observe_as(env, pid, entry.obs_mode)
        mask = legal_action_mask(env._state)
        return act_prefix_sliced(entry.policy, obs, mask, device=device, deterministic=True)
    # "self": live policy, stochastic sample, transitions discarded by the caller.
    obs_dict = env.observe(env.agent_selection)
    obs_t = torch.as_tensor(obs_dict["observation"], dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(obs_dict["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)
    action_t, _, _ = policy.act(obs_t, mask_t, deterministic=False)
    return int(action_t.item())


def collect_rollouts(
    policy: ActorCritic,
    n_games: int,
    *,
    rules_profile: Union[str, RulesProfile, None] = "simplified_v1",
    gamma: float = 0.999,
    lam: float = 0.95,
    max_turns: int = 500,
    seed: Optional[int] = None,
    device: str = "cpu",
    obs_mode: str = "self_play",
    reward_win: float = 1.0,
    reward_loss: float = -1.0,
    belief_blend: float = 0.25,
    belief_noise: float = 0.5,
    trace_dir: Optional[Union[str, Path]] = None,
    trace_every: Optional[int] = None,
    trace_prefix: str = "",
    opponents: Optional[Dict] = None,
    n_policy_seats: int = 1,
) -> Batch:
    """Play n_games of 4-seat Catan, optionally against an opponent pool.

    trace_dir / trace_every: opt-in game recording. When both are set, every
    game whose index `g` satisfies `g % trace_every == 0` is recorded with a
    TraceRecorder and saved to `trace_dir / f"{trace_prefix}game{g:04d}.json"`.
    When either is None (the default), no recorder is created and there is no
    extra state cloning/dict overhead in the hot loop.

    opponents: None (default) is pure 4-seat self-play with one shared
    policy -- byte-for-byte identical to the pre-pool behavior, with zero
    pool machinery on the hot path. `{"pool": [entry, ...]}` switches on
    opponent-pool mode: entries are
    `{"type": "personality", "name": ..., "weight": w}`,
    `{"type": "checkpoint", "path": ..., "weight": w}`, or
    `{"type": "self", "weight": w}`. Per game `g`, policy seats are
    `{(g + k) % 4 for k in range(n_policy_seats)}` (rotating); the remaining
    seats each draw independently from the pool by weight, seeded from the
    game seed so draws are reproducible and independent of the action RNG.
    Transitions are collected ONLY from policy seats.
    """
    tracing_enabled = trace_dir is not None and trace_every is not None
    if tracing_enabled:
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)

    pool_mode = opponents is not None
    pool_entries = _build_pool(opponents["pool"], obs_mode, device) if pool_mode else None

    env = CatanAECEnv(
        obs_mode=obs_mode,
        reward_win=reward_win,
        reward_loss=reward_loss,
        max_turns=max_turns,
        rules_profile=rules_profile,
        belief_blend=belief_blend,
        belief_noise=belief_noise,
    )

    all_obs, all_masks, all_actions = [], [], []
    all_logprobs, all_values, all_adv, all_ret = [], [], [], []
    all_seats, all_episodes = [], []

    episode_lengths: List[int] = []
    win_counts = [0, 0, 0, 0]
    vp_sums: List[float] = []
    truncated_games = 0

    # Pool-mode bookkeeping: policy win rate and per-opponent-label win rate,
    # left untouched (never even allocated) when opponents is None.
    policy_wins = 0
    label_games: Dict[str, int] = {}
    label_wins: Dict[str, int] = {}

    policy.eval()
    for game_idx in range(n_games):
        game_seed = None if seed is None else seed + game_idx
        env.reset(seed=game_seed)
        seats = {agent: _SeatTrajectory() for agent in env.agents}

        recorder = None
        if tracing_enabled and game_idx % trace_every == 0:
            recorder = TraceRecorder()
            recorder.start(
                env._state,
                {"seed": game_seed, "game_index": game_idx, "obs_mode": obs_mode},
            )

        policy_seats: set = set()
        seat_entry: Dict[int, _PoolEntry] = {}
        opp_rng: Optional[random.Random] = None
        if pool_mode:
            policy_seats = {(game_idx + k) % 4 for k in range(n_policy_seats)}
            pool_seed = (game_seed if game_seed is not None else game_idx) ^ _POOL_SEED_XOR
            opp_rng = random.Random(pool_seed)
            non_policy_seats = [s for s in range(4) if s not in policy_seats]
            weights = [e.weight for e in pool_entries]
            draws = opp_rng.choices(pool_entries, weights=weights, k=len(non_policy_seats))
            seat_entry = dict(zip(non_policy_seats, draws))

        while not (all(env.terminations.values()) or all(env.truncations.values())):
            agent = env.agent_selection

            if pool_mode:
                pid = int(agent.split("_")[1])
                if pid not in policy_seats:
                    action_idx = _act_opponent(env, seat_entry[pid], policy, opp_rng, device)
                    env.step(action_idx)
                    if recorder is not None:
                        recorder.record(CATALOG[action_idx], env._state)
                    continue

            obs_dict = env.observe(agent)
            obs = obs_dict["observation"]
            mask = obs_dict["action_mask"]

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            action_t, logprob_t, value_t = policy.act(obs_t, mask_t)

            traj = seats[agent]
            traj.obs.append(obs.astype(np.float32))
            traj.masks.append(mask)
            traj.actions.append(int(action_t.item()))
            traj.logprobs.append(float(logprob_t.item()))
            traj.values.append(float(value_t.item()))

            action_idx = int(action_t.item())
            env.step(action_idx)
            if recorder is not None:
                recorder.record(CATALOG[action_idx], env._state)

        if recorder is not None:
            recorder.save(trace_dir / f"{trace_prefix}game{game_idx:04d}.json")

        truncated = all(env.truncations.values()) and not all(env.terminations.values())
        if truncated:
            truncated_games += 1

        state = env._state
        episode_lengths.append(state.turn_number)
        if state.winner is not None:
            win_counts[state.winner] += 1
        vp_sums.append(float(np.mean([compute_vp(pid, state) for pid in range(4)])))

        if pool_mode:
            policy_won = state.winner is not None and state.winner in policy_seats
            if policy_won:
                policy_wins += 1
            for label in {entry.label for entry in seat_entry.values()}:
                label_games[label] = label_games.get(label, 0) + 1
                if policy_won:
                    label_wins[label] = label_wins.get(label, 0) + 1

        # Per-seat GAE: final transition carries the terminal reward, done=True.
        for seat_idx, agent in enumerate(env.agents):
            traj = seats[agent]
            n = len(traj)
            if n == 0:
                continue
            rewards = np.zeros(n, dtype=np.float32)
            rewards[-1] = env.rewards[agent]  # +win/-loss at termination; 0 on truncation
            dones = np.zeros(n, dtype=np.float32)
            dones[-1] = 1.0
            values = np.asarray(traj.values, dtype=np.float32)
            adv, ret = compute_gae(rewards, values, dones, gamma, lam)

            all_obs.extend(traj.obs)
            all_masks.extend(traj.masks)
            all_actions.extend(traj.actions)
            all_logprobs.extend(traj.logprobs)
            all_values.extend(traj.values)
            all_adv.extend(adv.tolist())
            all_ret.extend(ret.tolist())
            all_seats.extend([seat_idx] * n)
            all_episodes.extend([game_idx] * n)

    stats = {
        "mean_episode_length": float(np.mean(episode_lengths)),
        "win_counts": win_counts,
        "mean_vp_at_end": float(np.mean(vp_sums)),
        "games_completed": n_games,
        "truncated_games": truncated_games,
    }
    if pool_mode:
        stats["policy_win_rate"] = policy_wins / n_games
        stats["opponent_win_rates"] = {
            label: label_wins.get(label, 0) / games for label, games in label_games.items()
        }

    return Batch(
        obs=torch.as_tensor(np.stack(all_obs), dtype=torch.float32, device=device),
        masks=torch.as_tensor(np.stack(all_masks), dtype=torch.bool, device=device),
        actions=torch.as_tensor(all_actions, dtype=torch.long, device=device),
        logprobs=torch.as_tensor(all_logprobs, dtype=torch.float32, device=device),
        values=torch.as_tensor(all_values, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(all_adv, dtype=torch.float32, device=device),
        returns=torch.as_tensor(all_ret, dtype=torch.float32, device=device),
        seat_ids=torch.as_tensor(all_seats, dtype=torch.long, device=device),
        episode_ids=torch.as_tensor(all_episodes, dtype=torch.long, device=device),
        stats=stats,
    )


def _aggregate_stats(stats_list: List[Dict]) -> Dict:
    """Merge stats dicts from multiple workers by summing numeric values.

    All keys from all dicts are preserved. For numeric values, they are summed;
    for non-numeric values, the first value is taken.
    """
    if not stats_list:
        return {}

    aggregated: Dict = {}

    for stats in stats_list:
        for key, value in stats.items():
            if key not in aggregated:
                aggregated[key] = value
            else:
                # Try to add numerically
                try:
                    aggregated[key] = aggregated[key] + value
                except (TypeError, ValueError):
                    # Non-numeric: keep the first value
                    pass

    return aggregated


def _aggregate_batches(batches: List[Batch]) -> Batch:
    """Concatenate batches along dim=0 (batch dimension).

    Concatenates all tensor fields and merges stats dicts using _aggregate_stats().
    """
    if not batches:
        raise ValueError("batches list cannot be empty")

    # Concatenate tensors along dim=0
    aggregated_batch = Batch(
        obs=torch.cat([b.obs for b in batches], dim=0),
        masks=torch.cat([b.masks for b in batches], dim=0),
        actions=torch.cat([b.actions for b in batches], dim=0),
        logprobs=torch.cat([b.logprobs for b in batches], dim=0),
        values=torch.cat([b.values for b in batches], dim=0),
        advantages=torch.cat([b.advantages for b in batches], dim=0),
        returns=torch.cat([b.returns for b in batches], dim=0),
        seat_ids=torch.cat([b.seat_ids for b in batches], dim=0),
        episode_ids=torch.cat([b.episode_ids for b in batches], dim=0),
        stats=_aggregate_stats([b.stats for b in batches]),
    )

    return aggregated_batch


def collect_rollouts_parallel(
    policy: ActorCritic,
    n_games: int,
    num_workers: Optional[int] = None,
    rules_profile: Optional[RulesProfile] = None,
    gamma: float = 0.99,
    lam: float = 0.95,
    max_turns: int = 500,
    seed: int = 42,
    opponent_pool: Optional[List[Dict]] = None,
    trace_every: Optional[int] = None,
    cfg: Optional[Dict] = None,
) -> Batch:
    """Collect game rollouts in parallel using multiprocessing.

    If num_workers is None, auto-detect CPU count. If num_workers <= 1, fall back
    to sequential collection using collect_rollouts(). Otherwise, creates a pool
    and distributes games evenly across workers.

    Args:
        policy: The ActorCritic policy to use.
        n_games: Number of games to collect.
        num_workers: Number of worker processes. None = auto-detect CPU count.
        rules_profile: Game rules profile.
        gamma: Discount factor for GAE.
        lam: Lambda parameter for GAE.
        max_turns: Maximum turns per game.
        seed: Random seed for reproducibility.
        opponent_pool: Optional opponent pool spec.
        trace_every: Optional trace recording interval.
        cfg: Optional additional configuration.

    Returns:
        Aggregated Batch from all workers.
    """
    # Auto-detect CPU count if not specified
    if num_workers is None:
        num_workers = cpu_count()

    # Fall back to sequential if num_workers <= 1
    if num_workers <= 1:
        return collect_rollouts(
            policy=policy,
            n_games=n_games,
            rules_profile=rules_profile,
            gamma=gamma,
            lam=lam,
            max_turns=max_turns,
            seed=seed,
            opponents=opponent_pool,
        )

    # Distribute games across workers (round-robin for remainder)
    games_per_worker = [n_games // num_workers] * num_workers
    for i in range(n_games % num_workers):
        games_per_worker[i] += 1

    # Prepare arguments for each worker
    worker_args = []
    for worker_id in range(num_workers):
        worker_args.append(
            (
                worker_id,
                games_per_worker[worker_id],
                policy,
                rules_profile,
                gamma,
                lam,
                max_turns,
                seed,
                opponent_pool,
                cfg,
            )
        )

    # Create pool and collect results
    try:
        from ..rl.parallel_rollout import _worker_collect_games

        with Pool(processes=num_workers) as pool:
            batches = pool.starmap(_worker_collect_games, worker_args)

        # Aggregate results
        return _aggregate_batches(batches)

    except Exception as e:
        # Fall back to sequential collection on pool creation failure
        warnings.warn(
            f"Parallel pool creation failed ({e}), falling back to sequential collection.",
            RuntimeWarning,
        )
        return collect_rollouts(
            policy=policy,
            n_games=n_games,
            rules_profile=rules_profile,
            gamma=gamma,
            lam=lam,
            max_turns=max_turns,
            seed=seed,
            opponents=opponent_pool,
        )
