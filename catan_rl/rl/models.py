"""
Actor-critic network with legal-action masking.

The policy head always outputs 256 logits (the full action catalog);
illegal slots are masked to a large negative value before softmax so the
categorical distribution places (numerically) zero probability on them.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..env.actions import CATALOG_SIZE
from ..env.observation import OBS_DIM

_MASK_VALUE = -1e9  # large negative instead of -inf keeps softmax/entropy finite


def masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    """Set logits of illegal actions (mask == False) to a large negative value."""
    return torch.where(mask, logits, torch.full_like(logits, _MASK_VALUE))


def _init_layer(layer: nn.Linear, std: float = 2 ** 0.5) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """MLP trunk with a 256-logit policy head and a scalar value head."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = CATALOG_SIZE,
        hidden_sizes: Sequence[int] = (512, 512),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(_init_layer(nn.Linear(last, h)))
            layers.append(nn.Tanh())
            last = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = _init_layer(nn.Linear(last, n_actions), std=0.01)
        self.value_head = _init_layer(nn.Linear(last, 1), std=1.0)

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (logits (B, n_actions), value (B,))."""
        z = self.trunk(obs)
        return self.policy_head(z), self.value_head(z).squeeze(-1)

    def _dist(self, obs: Tensor, mask: Tensor) -> Tuple[torch.distributions.Categorical, Tensor]:
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=masked_logits(logits, mask))
        return dist, value

    @torch.no_grad()
    def act(
        self, obs: Tensor, mask: Tensor, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample (or argmax) an action. Returns (action, logprob, value), each (B,)."""
        dist, value = self._dist(obs, mask)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(
        self, obs: Tensor, mask: Tensor, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return (logprob, entropy, value) for given actions, with gradients."""
        dist, value = self._dist(obs, mask)
        return dist.log_prob(actions), dist.entropy(), value
