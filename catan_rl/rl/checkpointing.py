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
from typing import Dict, Optional, Tuple, Union

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


def list_checkpoints(directory: Union[str, Path]) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    ckpts = [p for p in directory.iterdir() if _CKPT_RE.search(p.name)]
    return sorted(ckpts, key=lambda p: int(_CKPT_RE.search(p.name).group(1)))


def latest_checkpoint(directory: Union[str, Path]) -> Optional[Path]:
    ckpts = list_checkpoints(directory)
    return ckpts[-1] if ckpts else None
