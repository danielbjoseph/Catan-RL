"""Log aggregation for multi-rank training runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class LogAggregator:
    """Aggregates per-rank JSONL logs into summary statistics."""

    def __init__(self, log_dir: str | Path):
        """Initialize aggregator with a log directory.

        Args:
            log_dir: Directory containing rank-*.jsonl files.
        """
        self.log_dir = Path(log_dir)
        self.logs_by_rank: Dict[int, List[Dict[str, Any]]] = {}
        self.metrics_by_rank: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.events_by_rank: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self._load_logs()

    def _load_logs(self) -> None:
        """Load all rank-*.jsonl files from log directory."""
        if not self.log_dir.exists():
            return

        rank_files = sorted(self.log_dir.glob("rank-*.jsonl"))

        for rank_file in rank_files:
            try:
                rank = int(rank_file.stem.split("-")[1])
            except (ValueError, IndexError):
                continue

            logs = []
            metrics = defaultdict(list)
            events = []

            try:
                with open(rank_file) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                            logs.append(entry)

                            if entry.get("type") == "metric":
                                metric_name = entry.get("name")
                                metric_value = entry.get("value")
                                if metric_name is not None and metric_value is not None:
                                    metrics[metric_name].append(metric_value)
                            elif entry.get("type") == "event":
                                events.append(entry)
                        except json.JSONDecodeError:
                            continue
            except (IOError, OSError):
                continue

            self.logs_by_rank[rank] = logs
            self.metrics_by_rank[rank] = metrics
            self.events_by_rank[rank] = events

    def get_rank_count(self) -> int:
        """Return number of ranks with logs."""
        return len(self.logs_by_rank)

    def get_ranks(self) -> List[int]:
        """Return sorted list of ranks."""
        return sorted(self.logs_by_rank.keys())

    def get_logs(self, rank: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all logs for a rank, or all logs if rank is None.

        Args:
            rank: Rank number, or None for all logs.

        Returns:
            List of log entries (dict).
        """
        if rank is None:
            all_logs = []
            for r in sorted(self.logs_by_rank.keys()):
                all_logs.extend(self.logs_by_rank[r])
            return all_logs
        return self.logs_by_rank.get(rank, [])

    def get_metrics(self, rank: Optional[int] = None) -> Dict[str, List[float]]:
        """Get all metrics for a rank, or aggregated across ranks if rank is None.

        Args:
            rank: Rank number, or None for aggregated metrics.

        Returns:
            Dict mapping metric name to list of values.
        """
        if rank is None:
            # Aggregate metrics across all ranks
            agg_metrics: Dict[str, List[float]] = defaultdict(list)
            for r in sorted(self.metrics_by_rank.keys()):
                for metric_name, values in self.metrics_by_rank[r].items():
                    agg_metrics[metric_name].extend(values)
            return dict(agg_metrics)
        return dict(self.metrics_by_rank.get(rank, {}))

    def get_events(self, rank: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all events for a rank, or all events if rank is None.

        Args:
            rank: Rank number, or None for all events.

        Returns:
            List of event entries (dict).
        """
        if rank is None:
            all_events = []
            for r in sorted(self.events_by_rank.keys()):
                all_events.extend(self.events_by_rank[r])
            return all_events
        return self.events_by_rank.get(rank, [])

    def get_metric_stats(
        self, metric_name: str, rank: Optional[int] = None
    ) -> Dict[str, float]:
        """Get statistics (min, max, mean) for a metric.

        Args:
            metric_name: Name of the metric.
            rank: Rank number, or None for aggregated stats.

        Returns:
            Dict with keys: min, max, mean, count.
        """
        if rank is None:
            metrics = self.get_metrics(rank=None)
            values = metrics.get(metric_name, [])
        else:
            metrics = self.get_metrics(rank=rank)
            values = metrics.get(metric_name, [])

        if not values:
            return {"min": None, "max": None, "mean": None, "count": 0}

        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "count": len(values),
        }

    def get_event_counts(
        self, rank: Optional[int] = None
    ) -> Dict[str, int]:
        """Get count of each event type.

        Args:
            rank: Rank number, or None for aggregated counts.

        Returns:
            Dict mapping event type to count.
        """
        events = self.get_events(rank=rank)
        counts: Dict[str, int] = defaultdict(int)
        for event in events:
            event_type = event.get("event_type", "unknown")
            counts[event_type] += 1
        return dict(counts)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all logs.

        Returns:
            Dict with overall statistics across all ranks.
        """
        summary: Dict[str, Any] = {
            "rank_count": self.get_rank_count(),
            "ranks": self.get_ranks(),
            "total_log_entries": len(self.get_logs(rank=None)),
            "total_metrics": len(self.get_metrics(rank=None)),
            "total_events": len(self.get_events(rank=None)),
            "event_type_counts": self.get_event_counts(rank=None),
        }

        # Add metric stats for each metric
        all_metrics = self.get_metrics(rank=None)
        metric_stats = {}
        for metric_name in sorted(all_metrics.keys()):
            metric_stats[metric_name] = self.get_metric_stats(metric_name, rank=None)
        summary["metric_stats"] = metric_stats

        return summary
