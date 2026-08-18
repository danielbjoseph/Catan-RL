"""
Custom PPO (clipped surrogate) for the shared self-play policy.

Deliberately not stable-baselines3 (spec §0): we need full control over
multi-seat trajectory batching. The trainer consumes a flattened `Batch`
(already GAE-processed per seat) and runs several epochs of minibatch
updates over it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from .logger import StructuredLogger
from .models import ActorCritic
from .rollout import Batch


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_coef: float = 0.2
    epochs: int = 4
    minibatch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.999
    gae_lambda: float = 0.95
    hidden_sizes: Sequence[int] = (512, 512)
    target_kl: Optional[float] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PPOConfig":
        """Load from YAML, ignoring any keys that are not PPOConfig fields
        (train scripts keep run-level settings in the same file)."""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid}
        if "hidden_sizes" in kwargs:
            kwargs["hidden_sizes"] = tuple(kwargs["hidden_sizes"])
        return cls(**kwargs)


class PPOTrainer:
    def __init__(self, policy: ActorCritic, cfg: PPOConfig, device: str = "cpu", logger: Optional[StructuredLogger] = None):
        self.policy = policy.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
        self.logger = logger
        self.update_step = 0

    def update(self, batch: Batch) -> Dict[str, float]:
        """One PPO update (cfg.epochs passes of minibatches) over the batch."""
        cfg = self.cfg
        self.policy.train()

        n = len(batch)
        obs = batch.obs.to(self.device)
        masks = batch.masks.to(self.device)
        actions = batch.actions.to(self.device)
        old_logprobs = batch.logprobs.to(self.device)
        advantages = batch.advantages.to(self.device)
        returns = batch.returns.to(self.device)

        policy_losses, value_losses, entropies = [], [], []
        kls, clip_fracs = [], []

        early_stop = False
        for _epoch in range(cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, cfg.minibatch_size):
                idx = perm[start:start + cfg.minibatch_size]
                if len(idx) < 2:
                    continue

                mb_adv = advantages[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                new_logprob, entropy, new_value = self.policy.evaluate_actions(
                    obs[idx], masks[idx], actions[idx]
                )
                log_ratio = new_logprob - old_logprobs[idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = 0.5 * ((new_value - returns[idx]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())
                kls.append(approx_kl.item())
                clip_fracs.append(clip_frac.item())

                if cfg.target_kl is not None and approx_kl.item() > 1.5 * cfg.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        def _mean(xs):
            return float(sum(xs) / max(len(xs), 1))

        stats = {
            "policy_loss": _mean(policy_losses),
            "value_loss": _mean(value_losses),
            "entropy": _mean(entropies),
            "approx_kl": _mean(kls),
            "clip_fraction": _mean(clip_fracs),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        if self.logger is not None:
            for name, value in stats.items():
                self.logger.log_metric(name, value, step=self.update_step)
            self.update_step += 1

        return stats
