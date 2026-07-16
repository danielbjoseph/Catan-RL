"""
Evaluation harness: trained policy vs scripted bots and past checkpoints.

Evaluation always uses greedy (deterministic) action selection for the
policy (spec §6: stochastic sampling is for training only).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch

from ..env.action_mask import legal_action_mask
from ..env.actions import CATALOG
from ..env.belief import BeliefTracker
from ..env.board import BoardConfig
from ..env.game_state import GameState
from ..env.observation import make_observation
from ..env.rules import apply_action
from ..env.rules_profile import RulesProfile
from ..env.scoring import compute_vp
from ..env.trace import TraceRecorder
from .checkpointing import load_policy
from .models import ActorCritic

BotFn = Callable[..., object]  # pick_action(state, rng) -> Action


def policy_action(
    policy: ActorCritic,
    state: GameState,
    device: str = "cpu",
    deterministic: bool = True,
    obs_mode: str = "self_play",
    belief: Optional[BeliefTracker] = None,
    noise_cfg: Optional[Dict] = None,
) -> int:
    """Greedy catalog index for the current player of `state`."""
    obs = make_observation(
        state, observer=state.current_player, mode=obs_mode,
        belief=belief, noise_cfg=noise_cfg,
    )
    mask = legal_action_mask(state)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
    action, _, _ = policy.act(obs_t, mask_t, deterministic=deterministic)
    return int(action.item())


def _play_eval_game(
    seat_actors,  # list of 4 callables: state, rng -> catalog index int
    seed: int,
    profile: RulesProfile,
    max_turns: int,
    tracker: Optional[BeliefTracker] = None,
    recorder: Optional[TraceRecorder] = None,
    trace_meta: Optional[Dict] = None,
) -> GameState:
    rng = random.Random(seed)
    config = BoardConfig.standard(seed=seed)
    state = GameState.new_game(config, n_players=4, seed=seed, profile=profile)
    if recorder is not None:
        recorder.start(state, trace_meta or {"seed": seed})
    max_plies = max_turns * 20  # generous ply budget per game
    plies = 0
    while not state.is_terminal and state.turn_number < max_turns and plies < max_plies:
        idx = seat_actors[state.current_player](state, rng)
        action = CATALOG[idx]
        if tracker is not None:
            before = state.clone()
            apply_action(state, action, rng)
            tracker.on_action(before, action, state)
        else:
            apply_action(state, action, rng)
        if recorder is not None:
            recorder.record(action, state)
        plies += 1
    return state


def _maybe_recorder(
    trace_dir: Optional[Union[str, Path]],
    trace_every: Optional[int],
    game_idx: int,
) -> Optional[TraceRecorder]:
    """Zero-overhead when tracing is off: no recorder, no dict built."""
    if trace_dir is None or trace_every is None or game_idx % trace_every != 0:
        return None
    return TraceRecorder()


def _save_trace(
    recorder: Optional[TraceRecorder],
    trace_dir: Optional[Union[str, Path]],
    trace_prefix: str,
    game_idx: int,
) -> None:
    if recorder is None:
        return
    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    recorder.save(Path(trace_dir) / f"{trace_prefix}game{game_idx:04d}.json")


def _bot_actor(bot_pick_action: BotFn):
    def actor(state: GameState, rng: random.Random) -> int:
        return bot_pick_action(state, rng).catalog_index
    return actor


def _policy_actor(
    policy: ActorCritic,
    device: str,
    deterministic: bool = True,
    obs_mode: str = "self_play",
    noise_cfg: Optional[Dict] = None,
    tracker_ref: Optional[BeliefTracker] = None,
):
    def actor(state: GameState, rng: random.Random) -> int:
        return policy_action(
            policy, state, device=device, deterministic=deterministic,
            obs_mode=obs_mode, belief=tracker_ref, noise_cfg=noise_cfg,
        )
    return actor


def _make_tracker(game_seed: int, profile: RulesProfile) -> BeliefTracker:
    """A fresh BeliefTracker anchored to a game's initial (all-zero-hand)
    state. Constructed independently of `_play_eval_game`'s own state so it
    can be handed to the policy actors before the game state exists; a fresh
    game's opening hands are always zero regardless of board layout, so this
    matches the tracker `_play_eval_game` would have anchored internally."""
    config = BoardConfig.standard(seed=game_seed)
    init_state = GameState.new_game(config, n_players=4, seed=game_seed, profile=profile)
    return BeliefTracker(init_state)


def evaluate_vs_bots(
    policy: ActorCritic,
    bot_pick_action: BotFn,
    n_games: int = 20,
    *,
    rules_profile: Union[str, RulesProfile, None] = "simplified_v1",
    seed: int = 0,
    max_turns: int = 500,
    device: str = "cpu",
    obs_mode: str = "self_play",
    noise_cfg: Optional[Dict] = None,
    trace_dir: Optional[Union[str, Path]] = None,
    trace_every: Optional[int] = None,
    trace_prefix: str = "",
) -> Dict:
    """Policy on a rotating seat vs three copies of a scripted bot."""
    profile = RulesProfile.get(rules_profile)
    policy.eval()
    wins = 0
    vps, turns, seats_played = [], [], []

    for i in range(n_games):
        seat = i % 4
        game_seed = seed + i
        tracker = _make_tracker(game_seed, profile) if obs_mode == "realistic" else None
        actors = [_bot_actor(bot_pick_action)] * 4
        actors[seat] = _policy_actor(
            policy, device, obs_mode=obs_mode, noise_cfg=noise_cfg, tracker_ref=tracker,
        )
        recorder = _maybe_recorder(trace_dir, trace_every, i)
        state = _play_eval_game(
            actors, seed=game_seed, profile=profile, max_turns=max_turns, tracker=tracker,
            recorder=recorder,
            trace_meta={"seed": game_seed, "game_index": i, "obs_mode": obs_mode, "seat": seat},
        )
        _save_trace(recorder, trace_dir, trace_prefix, i)
        seats_played.append(seat)
        if state.winner == seat:
            wins += 1
        vps.append(compute_vp(seat, state))
        turns.append(state.turn_number)

    return {
        "win_rate": wins / n_games,
        "mean_vp": float(np.mean(vps)),
        "mean_turns": float(np.mean(turns)),
        "games": n_games,
        "seats_played": seats_played,
    }


def evaluate_vs_checkpoint(
    policy: ActorCritic,
    checkpoint_path: Union[str, Path],
    n_games: int = 20,
    *,
    rules_profile: Union[str, RulesProfile, None] = "simplified_v1",
    seed: int = 0,
    max_turns: int = 500,
    device: str = "cpu",
    obs_mode: str = "self_play",
    noise_cfg: Optional[Dict] = None,
    trace_dir: Optional[Union[str, Path]] = None,
    trace_every: Optional[int] = None,
    trace_prefix: str = "",
) -> Dict:
    """Current policy (2 seats, `obs_mode`) vs an older checkpoint (2 seats,
    its own stored obs_mode from checkpoint metadata), seats rotating."""
    profile = RulesProfile.get(rules_profile)
    old_policy, old_meta = load_policy(checkpoint_path, device=device)
    old_obs_mode = old_meta.get("obs_mode", "self_play")
    policy.eval()

    wins = 0
    for i in range(n_games):
        current_seats = {s for s in range(4) if (s + i) % 2 == 0}
        game_seed = seed + i
        needs_tracker = obs_mode == "realistic" or old_obs_mode == "realistic"
        tracker = _make_tracker(game_seed, profile) if needs_tracker else None

        actors = []
        for s in range(4):
            if s in current_seats:
                actors.append(_policy_actor(
                    policy, device, obs_mode=obs_mode, noise_cfg=noise_cfg, tracker_ref=tracker,
                ))
            else:
                actors.append(_policy_actor(
                    old_policy, device, obs_mode=old_obs_mode, noise_cfg=noise_cfg, tracker_ref=tracker,
                ))
        recorder = _maybe_recorder(trace_dir, trace_every, i)
        state = _play_eval_game(
            actors, seed=game_seed, profile=profile, max_turns=max_turns, tracker=tracker,
            recorder=recorder,
            trace_meta={
                "seed": game_seed, "game_index": i,
                "obs_mode": obs_mode, "old_obs_mode": old_obs_mode,
                "current_seats": sorted(current_seats),
            },
        )
        _save_trace(recorder, trace_dir, trace_prefix, i)
        if state.winner is not None and state.winner in current_seats:
            wins += 1

    return {"win_rate": wins / n_games, "games": n_games}


def evaluate_policy_vs_policy(
    policy_a: ActorCritic,
    mode_a: str,
    policy_b: ActorCritic,
    mode_b: str,
    n_games: int = 20,
    *,
    rules_profile: Union[str, RulesProfile, None] = "simplified_v1",
    seed: int = 0,
    max_turns: int = 500,
    noise_cfg_a: Optional[Dict] = None,
    noise_cfg_b: Optional[Dict] = None,
    device: str = "cpu",
    trace_dir: Optional[Union[str, Path]] = None,
    trace_every: Optional[int] = None,
    trace_prefix: str = "",
) -> Dict:
    """Two in-memory policies (each own obs mode), 2 seats apiece, seats
    rotating game to game: game i gives policy_a the seats where
    (s + i) % 2 == 0."""
    profile = RulesProfile.get(rules_profile)
    policy_a.eval()
    policy_b.eval()

    wins_a = 0
    wins_b = 0
    draws = 0
    for i in range(n_games):
        a_seats = {s for s in range(4) if (s + i) % 2 == 0}
        game_seed = seed + i
        needs_tracker = mode_a == "realistic" or mode_b == "realistic"
        tracker = _make_tracker(game_seed, profile) if needs_tracker else None

        actors = []
        for s in range(4):
            if s in a_seats:
                actors.append(_policy_actor(
                    policy_a, device, obs_mode=mode_a, noise_cfg=noise_cfg_a, tracker_ref=tracker,
                ))
            else:
                actors.append(_policy_actor(
                    policy_b, device, obs_mode=mode_b, noise_cfg=noise_cfg_b, tracker_ref=tracker,
                ))
        recorder = _maybe_recorder(trace_dir, trace_every, i)
        state = _play_eval_game(
            actors, seed=game_seed, profile=profile, max_turns=max_turns, tracker=tracker,
            recorder=recorder,
            trace_meta={
                "seed": game_seed, "game_index": i,
                "mode_a": mode_a, "mode_b": mode_b, "a_seats": sorted(a_seats),
            },
        )
        _save_trace(recorder, trace_dir, trace_prefix, i)
        if state.winner is None:
            draws += 1
        elif state.winner in a_seats:
            wins_a += 1
        else:
            wins_b += 1

    return {
        "win_rate_a": wins_a / n_games,
        "win_rate_b": wins_b / n_games,
        "draws": draws,
        "games": n_games,
    }
