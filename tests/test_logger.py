import tempfile
import json
from pathlib import Path
from catan_rl.rl.logger import StructuredLogger

def test_logger_single_process():
    """Test logger in single-process mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = StructuredLogger(run_id="test-run", log_dir=tmpdir)

        assert logger.get_rank() == 0
        assert logger.get_world_size() == 1

        logger.log_metric("loss", 0.5, step=1)
        logger.log_event("training_started", epoch=0, lr=3e-4)

        log_file = Path(tmpdir) / "rank-0.jsonl"
        assert log_file.exists()

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2

        metric_log = json.loads(lines[0])
        assert metric_log["type"] == "metric"
        assert metric_log["name"] == "loss"
        assert metric_log["value"] == 0.5
        assert metric_log["step"] == 1

        event_log = json.loads(lines[1])
        assert event_log["type"] == "event"
        assert event_log["event_type"] == "training_started"
        assert event_log["epoch"] == 0

def test_logger_creates_directory():
    """Test that logger creates log directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_dir = Path(tmpdir) / "logs" / "nested"
        logger = StructuredLogger(run_id="test", log_dir=str(nested_dir))
        logger.log_metric("test", 1.0)

        assert (nested_dir / "rank-0.jsonl").exists()

def test_logger_default_log_dir():
    """Test that logger uses default 'runs/<run_id>/logs' if no log_dir provided."""
    import shutil
    try:
        logger = StructuredLogger(run_id="test-default")
        logger.log_metric("test", 1.0)

        expected_path = Path("runs/test-default/logs/rank-0.jsonl")
        assert expected_path.exists()
    finally:
        shutil.rmtree("runs/test-default", ignore_errors=True)
