"""
Checkpoint save/load.

Each checkpoint is `ckpt_<iteration:06d>.pt` (torch state_dicts + arch) with a
human-readable `.json` sidecar carrying iteration, config, metrics, and arch.
Only state_dicts are serialized — never pickled module objects.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from .models import ActorCritic

_CKPT_RE = re.compile(r"ckpt_(\d+)\.pt$")


def _arch_of(policy: ActorCritic) -> Dict:
    hidden = [m.out_features for m in policy.trunk if isinstance(m, torch.nn.Linear)]
    return {
        "obs_dim": policy.obs_dim,
        "n_actions": policy.n_actions,
        "hidden_sizes": hidden,
    }


def save_checkpoint(
    directory: Union[str, Path],
    policy: ActorCritic,
    optimizer: Optional[torch.optim.Optimizer],
    iteration: int,
    config: Dict,
    metrics: Dict,
    obs_mode: str = "self_play",
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"ckpt_{iteration:06d}.pt"

    metadata = {
        "iteration": iteration,
        "config": config,
        "metrics": metrics,
        "arch": _arch_of(policy),
        "obs_mode": obs_mode,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch_version": str(torch.__version__),
    }
    payload = {
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "metadata": metadata,
    }
    torch.save(payload, path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str))
    return path


def load_checkpoint(
    path: Union[str, Path],
    policy: ActorCritic,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    """Load weights (and optimizer state) into existing objects; returns metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    policy.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload["metadata"]


def load_policy(path: Union[str, Path], device: str = "cpu") -> Tuple[ActorCritic, Dict]:
    """Reconstruct an ActorCritic from a checkpoint's stored architecture."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    arch = payload["metadata"]["arch"]
    policy = ActorCritic(
        obs_dim=arch["obs_dim"],
        n_actions=arch["n_actions"],
        hidden_sizes=tuple(arch["hidden_sizes"]),
    )
    policy.load_state_dict(payload["model"])
    policy.to(device)
    policy.eval()
    return policy, payload["metadata"]


def widen_policy(
    old: ActorCritic,
    new_obs_dim: int,
    new_n_actions: int,
    new_hidden_sizes: Optional[Sequence[int]] = None,
) -> ActorCritic:
    """Prefix-preserving widening: first trunk layer's new input columns are
    zero (new features ignored until learned); policy head's new rows get
    zero weights and bias -4.0 (near-zero initial probability, spec §3.4);
    all other weights copied. Requires new dims >= old dims and identical
    hidden_sizes.

    `new_hidden_sizes` is optional and only used to *validate* that a
    caller's intended target architecture matches `old`'s hidden layer
    sizes (this function cannot widen/reshape hidden layers) -- omit it to
    simply reuse `old`'s hidden_sizes, which is the common case.
    """
    old_hidden = tuple(m.out_features for m in old.trunk if isinstance(m, torch.nn.Linear))
    if new_hidden_sizes is not None and tuple(new_hidden_sizes) != old_hidden:
        raise ValueError(
            f"widen_policy cannot change hidden_sizes: old={old_hidden}, "
            f"requested={tuple(new_hidden_sizes)}"
        )
    if new_obs_dim < old.obs_dim:
        raise ValueError(f"new_obs_dim ({new_obs_dim}) must be >= old obs_dim ({old.obs_dim})")
    if new_n_actions < old.n_actions:
        raise ValueError(
            f"new_n_actions ({new_n_actions}) must be >= old n_actions ({old.n_actions})"
        )

    new = ActorCritic(obs_dim=new_obs_dim, n_actions=new_n_actions, hidden_sizes=old_hidden)

    old_linears = [m for m in old.trunk if isinstance(m, torch.nn.Linear)]
    new_linears = [m for m in new.trunk if isinstance(m, torch.nn.Linear)]

    with torch.no_grad():
        # First trunk layer: old input columns + bias copied; new (extra)
        # input columns zeroed so the new features have no effect until
        # trained.
        first_old, first_new = old_linears[0], new_linears[0]
        first_new.weight.zero_()
        first_new.weight[:, : old.obs_dim].copy_(first_old.weight)
        first_new.bias.copy_(first_old.bias)

        # Remaining trunk layers are unchanged in shape (hidden_sizes
        # match) -- copy exactly.
        for old_layer, new_layer in zip(old_linears[1:], new_linears[1:]):
            new_layer.weight.copy_(old_layer.weight)
            new_layer.bias.copy_(old_layer.bias)

        # Policy head: old rows + bias copied; new (extra) rows get zero
        # weight and bias -4.0, i.e. near-zero initial probability.
        new.policy_head.weight.zero_()
        new.policy_head.weight[: old.n_actions, :].copy_(old.policy_head.weight)
        new.policy_head.bias.fill_(-4.0)
        new.policy_head.bias[: old.n_actions].copy_(old.policy_head.bias)

        # Value head shape is unchanged -- copy exactly.
        new.value_head.weight.copy_(old.value_head.weight)
        new.value_head.bias.copy_(old.value_head.bias)

    return new


def list_checkpoints(directory: Union[str, Path]) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    ckpts = [p for p in directory.iterdir() if _CKPT_RE.search(p.name)]
    return sorted(ckpts, key=lambda p: int(_CKPT_RE.search(p.name).group(1)))


def latest_checkpoint(directory: Union[str, Path]) -> Optional[Path]:
    ckpts = list_checkpoints(directory)
    return ckpts[-1] if ckpts else None
