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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ..env.actions import CATALOG, CATALOG_SIZE
from ..env.pettingzoo_env import CatanAECEnv
from ..env.rules_profile import RulesProfile
from ..env.scoring import compute_vp
from ..env.trace import TraceRecorder
from .models import ActorCritic


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
) -> Batch:
    """Play n_games of 4-seat self-play with a single shared policy.

    trace_dir / trace_every: opt-in game recording. When both are set, every
    game whose index `g` satisfies `g % trace_every == 0` is recorded with a
    TraceRecorder and saved to `trace_dir / f"{trace_prefix}game{g:04d}.json"`.
    When either is None (the default), no recorder is created and there is no
    extra state cloning/dict overhead in the hot loop.
    """
    tracing_enabled = trace_dir is not None and trace_every is not None
    if tracing_enabled:
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)

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

        while not (all(env.terminations.values()) or all(env.truncations.values())):
            agent = env.agent_selection
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
