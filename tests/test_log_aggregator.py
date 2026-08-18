"""Tests for log aggregation."""

import json
import tempfile
from pathlib import Path

from catan_rl.rl.log_aggregator import LogAggregator
from catan_rl.rl.logger import StructuredLogger


def test_log_aggregator_single_rank():
    """Test aggregator with single rank logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create logs from a single rank
        logger = StructuredLogger(run_id="test-single", log_dir=tmpdir)
        logger.log_metric("loss", 0.5, step=1)
        logger.log_metric("loss", 0.3, step=2)
        logger.log_event("training_started", epoch=0)

        # Aggregate
        agg = LogAggregator(tmpdir)
        assert agg.get_rank_count() == 1
        assert agg.get_ranks() == [0]
        assert len(agg.get_logs()) == 3

        # Check metrics
        metrics = agg.get_metrics()
        assert "loss" in metrics
        assert metrics["loss"] == [0.5, 0.3]

        # Check events
        events = agg.get_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "training_started"


def test_log_aggregator_multiple_ranks():
    """Test aggregator with multiple rank logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create logs for multiple ranks by manually writing files
        for rank in range(2):
            rank_file = Path(tmpdir) / f"rank-{rank}.jsonl"
            with open(rank_file, "w") as f:
                # Write some metrics
                for i in range(3):
                    entry = {
                        "type": "metric",
                        "rank": rank,
                        "name": "loss",
                        "value": 0.5 - rank * 0.1 - i * 0.05,
                    }
                    f.write(json.dumps(entry) + "\n")

                # Write an event
                event = {
                    "type": "event",
                    "rank": rank,
                    "event_type": "training_started",
                }
                f.write(json.dumps(event) + "\n")

        # Aggregate
        agg = LogAggregator(tmpdir)
        assert agg.get_rank_count() == 2
        assert agg.get_ranks() == [0, 1]
        assert len(agg.get_logs()) == 8  # 3 metrics + 1 event per rank = 4 per rank

        # Check aggregated metrics
        metrics = agg.get_metrics()
        assert "loss" in metrics
        assert len(metrics["loss"]) == 6  # 3 per rank

        # Check per-rank metrics
        rank0_metrics = agg.get_metrics(rank=0)
        assert len(rank0_metrics["loss"]) == 3

        # Check events
        events = agg.get_events()
        assert len(events) == 2


def test_log_aggregator_metric_stats():
    """Test metric statistics calculation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rank_file = Path(tmpdir) / "rank-0.jsonl"
        with open(rank_file, "w") as f:
            for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
                entry = {"type": "metric", "name": "test_metric", "value": value}
                f.write(json.dumps(entry) + "\n")

        agg = LogAggregator(tmpdir)
        stats = agg.get_metric_stats("test_metric")

        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["count"] == 5


def test_log_aggregator_event_counts():
    """Test event count aggregation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rank_file = Path(tmpdir) / "rank-0.jsonl"
        with open(rank_file, "w") as f:
            # Write events
            for event_type in ["started", "started", "completed"]:
                entry = {"type": "event", "event_type": event_type}
                f.write(json.dumps(entry) + "\n")

        agg = LogAggregator(tmpdir)
        counts = agg.get_event_counts()

        assert counts["started"] == 2
        assert counts["completed"] == 1


def test_log_aggregator_empty_dir():
    """Test aggregator with empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agg = LogAggregator(tmpdir)
        assert agg.get_rank_count() == 0
        assert agg.get_ranks() == []
        assert len(agg.get_logs()) == 0


def test_log_aggregator_nonexistent_dir():
    """Test aggregator with non-existent directory."""
    agg = LogAggregator("/nonexistent/path")
    assert agg.get_rank_count() == 0
    assert agg.get_ranks() == []


def test_log_aggregator_summary():
    """Test summary generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a logger and log some entries
        logger = StructuredLogger(run_id="test-summary", log_dir=tmpdir)
        logger.log_metric("loss", 0.5, step=1)
        logger.log_metric("acc", 0.9, step=1)
        logger.log_event("training_started", epoch=0)
        logger.log_event("training_completed", epoch=1)

        # Aggregate
        agg = LogAggregator(tmpdir)
        summary = agg.get_summary()

        assert summary["rank_count"] == 1
        assert summary["total_log_entries"] == 4
        assert summary["total_metrics"] == 2
        assert summary["total_events"] == 2
        assert "started" in summary["event_type_counts"]
        assert "completed" in summary["event_type_counts"]
        assert "loss" in summary["metric_stats"]
        assert "acc" in summary["metric_stats"]


def test_log_aggregator_malformed_json():
    """Test that aggregator handles malformed JSON gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rank_file = Path(tmpdir) / "rank-0.jsonl"
        with open(rank_file, "w") as f:
            # Write valid entry
            f.write(json.dumps({"type": "metric", "name": "loss", "value": 0.5}) + "\n")
            # Write malformed entry
            f.write("invalid json\n")
            # Write another valid entry
            f.write(json.dumps({"type": "event", "event_type": "done"}) + "\n")

        agg = LogAggregator(tmpdir)
        logs = agg.get_logs()
        # Should have 2 valid entries (skipped malformed)
        assert len(logs) == 2


def test_log_aggregator_missing_fields():
    """Test that aggregator handles entries with missing fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rank_file = Path(tmpdir) / "rank-0.jsonl"
        with open(rank_file, "w") as f:
            # Entry with missing value
            f.write(json.dumps({"type": "metric", "name": "loss"}) + "\n")
            # Entry with missing name
            f.write(json.dumps({"type": "metric", "value": 0.5}) + "\n")
            # Valid entry
            f.write(json.dumps({"type": "metric", "name": "valid", "value": 1.0}) + "\n")

        agg = LogAggregator(tmpdir)
        metrics = agg.get_metrics()
        # Should have only the valid metric
        assert "valid" in metrics
        assert metrics["valid"] == [1.0]
