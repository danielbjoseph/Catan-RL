"""Structured logging for distributed training with Ray and torch.distributed."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredLogger:
    """Distributed-aware JSON logger for Ray workers and torch.distributed ranks."""

    def __init__(self, run_id: str, log_dir: Optional[str] = None, level: str = "INFO"):
        """Initialize logger.

        Args:
            run_id: Unique identifier for this training run.
            log_dir: Directory to write logs to. Defaults to runs/{run_id}/logs.
            level: Logging level ("DEBUG", "INFO", "WARNING"). (Unused in Phase 1, for future expansion.)
        """
        self.run_id = run_id
        self.level = level
        self.rank = self._detect_rank()
        self.world_size = self._detect_world_size()

        # Set log directory
        if log_dir is None:
            log_dir = f"runs/{run_id}/logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Log file per rank
        self.log_file = self.log_dir / f"rank-{self.rank}.jsonl"
        self._file_handle = None

    def _detect_rank(self) -> int:
        """Detect rank from Ray or torch.distributed environment."""
        # Check torch.distributed first (takes precedence)
        if "RANK" in os.environ:
            return int(os.environ["RANK"])

        # Check Ray next
        try:
            import ray
            if ray.is_initialized():
                worker_id = os.environ.get("RAY_WORKER_ID")
                if worker_id is not None:
                    # Simple mapping: worker ID to rank (0-indexed)
                    return int(worker_id)
        except (ImportError, RuntimeError):
            pass

        return 0

    def _detect_world_size(self) -> int:
        """Detect world size from torch.distributed or Ray environment."""
        # Check torch.distributed first
        if "WORLD_SIZE" in os.environ:
            return int(os.environ["WORLD_SIZE"])

        # Check Ray next
        try:
            import ray
            if ray.is_initialized():
                info = ray.cluster_resources()
                # Rough heuristic: total CPUs / CPUs per worker
                num_workers = os.environ.get("RAY_NUM_WORKERS")
                if num_workers is not None:
                    return int(num_workers)
        except (ImportError, RuntimeError):
            pass

        return 1

    def log_metric(
        self, name: str, value: float, step: Optional[int] = None, **tags
    ) -> None:
        """Log a scalar metric.

        Args:
            name: Metric name (e.g., "policy_loss", "throughput_games_per_sec").
            value: Metric value (float).
            step: Optional step/epoch number.
            **tags: Additional key-value tags (e.g., worker_id=0, phase="rollout").
        """
        log_entry = {
            "type": "metric",
            "timestamp": datetime.utcnow().isoformat(),
            "rank": self.rank,
            "world_size": self.world_size,
            "name": name,
            "value": value,
        }
        if step is not None:
            log_entry["step"] = step
        log_entry.update(tags)
        self._write_log(log_entry)

    def log_event(self, event_type: str, **data) -> None:
        """Log a discrete event.

        Args:
            event_type: Type of event (e.g., "training_started", "rollout_complete").
            **data: Event-specific key-value data.
        """
        log_entry = {
            "type": "event",
            "timestamp": datetime.utcnow().isoformat(),
            "rank": self.rank,
            "world_size": self.world_size,
            "event_type": event_type,
        }
        log_entry.update(data)
        self._write_log(log_entry)

    def _write_log(self, entry: Dict[str, Any]) -> None:
        """Write a log entry to file (newline-delimited JSON)."""
        with open(self.log_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")

    def get_rank(self) -> int:
        """Return this process's rank."""
        return self.rank

    def get_world_size(self) -> int:
        """Return total number of processes."""
        return self.world_size
