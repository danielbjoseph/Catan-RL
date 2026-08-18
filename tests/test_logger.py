import tempfile
import json
from pathlib import Path
from catan_rl.rl.logger import StructuredLogger
import torch

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

def test_logger_env_var_detection():
    """Test rank/world_size detection from environment variables."""
    import os
    import tempfile
    import shutil

    # Save original env vars
    original_rank = os.environ.get("RANK")
    original_world_size = os.environ.get("WORLD_SIZE")

    try:
        os.environ["RANK"] = "2"
        os.environ["WORLD_SIZE"] = "4"

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredLogger(run_id="test-env", log_dir=tmpdir)
            assert logger.get_rank() == 2
            assert logger.get_world_size() == 4
    finally:
        # Restore original env vars
        if original_rank is not None:
            os.environ["RANK"] = original_rank
        else:
            os.environ.pop("RANK", None)

        if original_world_size is not None:
            os.environ["WORLD_SIZE"] = original_world_size
        else:
            os.environ.pop("WORLD_SIZE", None)

def test_ppo_trainer_logging():
    """Test that PPO trainer logs metrics."""
    from catan_rl.rl.ppo import PPOTrainer, PPOConfig
    from catan_rl.rl.models import ActorCritic
    from catan_rl.rl.logger import StructuredLogger
    from catan_rl.rl.rollout import Batch

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = StructuredLogger(run_id="test-ppo", log_dir=tmpdir)

        # Create a minimal policy for testing
        policy = ActorCritic(obs_dim=5, n_actions=3, hidden_sizes=(64,))
        cfg = PPOConfig(lr=1e-4, epochs=1, minibatch_size=4)
        trainer = PPOTrainer(policy, cfg, device="cpu", logger=logger)

        # Create a minimal batch (5 samples)
        batch = Batch(
            obs=torch.randn(5, 5),
            masks=torch.ones(5, 3, dtype=torch.bool),
            actions=torch.randint(0, 3, (5,)),
            logprobs=torch.randn(5),
            values=torch.randn(5),
            advantages=torch.randn(5),
            returns=torch.randn(5),
            seat_ids=torch.zeros(5, dtype=torch.long),
            episode_ids=torch.zeros(5, dtype=torch.long),
            stats={},
        )

        # Run one update
        stats = trainer.update(batch)

        # Check that metrics were logged
        log_file = Path(tmpdir) / "rank-0.jsonl"
        assert log_file.exists()

        lines = log_file.read_text().strip().split("\n")
        # Should have logs for each metric
        metric_names = set()
        for line in lines:
            entry = json.loads(line)
            if entry["type"] == "metric":
                metric_names.add(entry["name"])

        expected_metrics = {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "learning_rate"}
        assert expected_metrics.issubset(metric_names)
