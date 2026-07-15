"""
Self-play training orchestrator.

Each iteration:
  1. collect_rollouts: N games of 4-seat self-play, one shared policy
  2. PPO update over the pooled per-seat batch
  3. TensorBoard logging (train/*, game/*)
  4. periodic evaluation vs random/greedy bots and the previous checkpoint (eval/*)
  5. periodic checkpointing to <run_dir>/checkpoints/

Config comes from a YAML file (see configs/ppo_baseline.yaml) or a dict.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..bots import greedy_bot, random_bot
from ..env.observation import obs_dim_for_mode
from ..env.rules_profile import RulesProfile
from .checkpointing import latest_checkpoint, load_checkpoint, save_checkpoint
from .evaluate import evaluate_vs_bots, evaluate_vs_checkpoint
from .models import ActorCritic
from .ppo import PPOConfig, PPOTrainer
from .rollout import collect_rollouts

_RUN_DEFAULTS = {
    "experiment_name": "ppo_baseline",
    "seed": 42,
    "iterations": 500,
    "games_per_iteration": 16,
    "eval_interval": 25,
    "eval_games": 12,
    "checkpoint_interval": 25,
    "rules_profile": "simplified_v1",
    "max_turns": 500,
    "reward_win": 1.0,
    "reward_loss": -1.0,
    "obs_mode": "self_play",
    "belief_blend": 0.25,
    "belief_noise": 0.5,
    "device": "cpu",
}


def _load_config(config: Union[str, Path, Dict]) -> Dict:
    if isinstance(config, (str, Path)):
        import yaml

        with open(config, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = dict(config)
    cfg = dict(_RUN_DEFAULTS)
    cfg.update(data)
    return cfg


class SelfPlayTrainer:
    def __init__(
        self,
        config: Union[str, Path, Dict],
        run_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        resume: bool = False,
    ):
        self.cfg = _load_config(config)
        self.device = device or self.cfg["device"]
        self.profile = RulesProfile.get(self.cfg["rules_profile"])

        seed = int(self.cfg["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.run_dir = Path(run_dir) if run_dir else Path("runs") / self.cfg["experiment_name"]
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.writer = SummaryWriter(log_dir=str(self.run_dir))

        # PPOConfig shares the config namespace; unknown keys are ignored.
        ppo_fields = {f for f in PPOConfig.__dataclass_fields__}
        self.ppo_cfg = PPOConfig(**{k: v for k, v in self.cfg.items() if k in ppo_fields})
        if isinstance(self.ppo_cfg.hidden_sizes, list):
            self.ppo_cfg = PPOConfig(**{
                **{k: getattr(self.ppo_cfg, k) for k in ppo_fields},
                "hidden_sizes": tuple(self.ppo_cfg.hidden_sizes),
            })

        obs_dim = obs_dim_for_mode(self.cfg["obs_mode"])
        self.policy = ActorCritic(obs_dim=obs_dim, hidden_sizes=self.ppo_cfg.hidden_sizes)
        self.trainer = PPOTrainer(self.policy, self.ppo_cfg, device=self.device)
        self.iteration = 0

        if resume:
            ckpt = latest_checkpoint(self.ckpt_dir)
            if ckpt is not None:
                meta = load_checkpoint(ckpt, self.policy, self.trainer.optimizer)
                self.iteration = int(meta["iteration"])
                print(f"[resume] loaded {ckpt} at iteration {self.iteration}")

    # ------------------------------------------------------------------

    def train(self, iterations: Optional[int] = None) -> None:
        total = iterations if iterations is not None else int(self.cfg["iterations"])
        games_per_iter = int(self.cfg["games_per_iteration"])
        end = self.iteration + total

        while self.iteration < end:
            it = self.iteration
            t0 = time.perf_counter()

            batch = collect_rollouts(
                self.policy,
                n_games=games_per_iter,
                rules_profile=self.profile,
                gamma=self.ppo_cfg.gamma,
                lam=self.ppo_cfg.gae_lambda,
                max_turns=int(self.cfg["max_turns"]),
                seed=int(self.cfg["seed"]) + it * games_per_iter,
                device=self.device,
                obs_mode=self.cfg["obs_mode"],
                reward_win=float(self.cfg["reward_win"]),
                reward_loss=float(self.cfg["reward_loss"]),
                belief_blend=float(self.cfg["belief_blend"]),
                belief_noise=float(self.cfg["belief_noise"]),
            )
            stats = self.trainer.update(batch)
            elapsed = time.perf_counter() - t0

            self._log_iteration(it, batch, stats, elapsed)
            self.iteration = it + 1

            if self.iteration % int(self.cfg["eval_interval"]) == 0:
                self._evaluate(it)
            if (
                self.iteration % int(self.cfg["checkpoint_interval"]) == 0
                or self.iteration == end
            ):
                self._checkpoint(stats)

    # ------------------------------------------------------------------

    def _log_iteration(self, it: int, batch, stats: Dict, elapsed: float) -> None:
        w = self.writer
        for k, v in stats.items():
            w.add_scalar(f"train/{k}", v, it)

        s = batch.stats
        games = max(s["games_completed"], 1)
        w.add_scalar("game/mean_episode_length", s["mean_episode_length"], it)
        for seat in range(4):
            w.add_scalar(f"game/win_rate_seat{seat}", s["win_counts"][seat] / games, it)
        w.add_scalar("game/mean_vp_at_end", s["mean_vp_at_end"], it)
        w.add_scalar("game/games_completed", s["games_completed"], it)
        w.add_scalar("game/truncated_games", s["truncated_games"], it)
        w.add_scalar("perf/iteration_seconds", elapsed, it)
        w.add_scalar("perf/transitions_per_iteration", len(batch), it)

        print(
            f"[iter {it:5d}] steps={len(batch):6d} "
            f"ep_len={s['mean_episode_length']:6.1f} vp={s['mean_vp_at_end']:4.2f} "
            f"pi_loss={stats['policy_loss']:+.4f} v_loss={stats['value_loss']:.4f} "
            f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} "
            f"({elapsed:.1f}s)"
        )

    def _evaluate(self, it: int) -> None:
        n = int(self.cfg["eval_games"])
        obs_mode = self.cfg["obs_mode"]
        noise_cfg = None
        if obs_mode == "realistic":
            noise_cfg = {
                "belief_blend": float(self.cfg["belief_blend"]),
                "belief_noise": float(self.cfg["belief_noise"]),
                "seed": 10_000_000 + it,
            }
        kwargs = dict(
            rules_profile=self.profile,
            seed=10_000_000 + it,
            max_turns=int(self.cfg["max_turns"]),
            device=self.device,
            obs_mode=obs_mode,
            noise_cfg=noise_cfg,
        )
        vs_random = evaluate_vs_bots(self.policy, random_bot.pick_action, n, **kwargs)
        vs_greedy = evaluate_vs_bots(self.policy, greedy_bot.pick_action, n, **kwargs)
        self.writer.add_scalar("eval/win_rate_vs_random", vs_random["win_rate"], it)
        self.writer.add_scalar("eval/win_rate_vs_greedy", vs_greedy["win_rate"], it)
        msg = (
            f"[eval {it:5d}] vs_random={vs_random['win_rate']:.2f} "
            f"vs_greedy={vs_greedy['win_rate']:.2f}"
        )

        prev = latest_checkpoint(self.ckpt_dir)
        if prev is not None:
            vs_prev = evaluate_vs_checkpoint(self.policy, prev, n, **kwargs)
            self.writer.add_scalar("eval/win_rate_vs_prev_checkpoint", vs_prev["win_rate"], it)
            msg += f" vs_prev={vs_prev['win_rate']:.2f}"
        print(msg)

    def _checkpoint(self, metrics: Dict) -> None:
        cfg_serializable = {
            k: (v.to_dict() if isinstance(v, RulesProfile) else v)
            for k, v in self.cfg.items()
        }
        path = save_checkpoint(
            self.ckpt_dir,
            self.policy,
            self.trainer.optimizer,
            self.iteration,
            config=cfg_serializable,
            metrics=metrics,
            obs_mode=self.cfg["obs_mode"],
        )
        print(f"[ckpt] saved {path}")

    def close(self) -> None:
        self.writer.close()
