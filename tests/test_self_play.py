"""End-to-end test: tiny self-play training run writes checkpoints + TB logs."""

from pathlib import Path

import pytest

from catan_rl.env.rules_profile import RulesProfile
from catan_rl.rl.checkpointing import list_checkpoints
from catan_rl.rl.self_play import SelfPlayTrainer

FAST = RulesProfile(name="fast", dev_cards_enabled=False, win_vp=8)

TINY_CONFIG = {
    "experiment_name": "tiny_test",
    "seed": 0,
    "iterations": 2,
    "games_per_iteration": 2,
    "eval_interval": 2,
    "eval_games": 1,
    "checkpoint_interval": 2,
    "rules_profile": FAST,
    "max_turns": 300,
    "hidden_sizes": [32, 32],
    "minibatch_size": 128,
    "epochs": 2,
    "device": "cpu",
}


@pytest.mark.parametrize("obs_mode", ["self_play", "realistic", "global"])
def test_tiny_training_run(tmp_path: Path, obs_mode: str):
    config = dict(TINY_CONFIG, obs_mode=obs_mode, experiment_name=f"tiny_test_{obs_mode}")
    trainer = SelfPlayTrainer(config, run_dir=tmp_path / "run")
    trainer.train()
    trainer.close()

    # Checkpoint written
    ckpts = list_checkpoints(tmp_path / "run" / "checkpoints")
    assert len(ckpts) >= 1

    # TensorBoard events written and contain required scalars
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(tmp_path / "run"))
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    required = {
        "train/policy_loss", "train/value_loss", "train/entropy",
        "train/approx_kl", "train/clip_fraction", "train/learning_rate",
        "game/mean_episode_length", "game/win_rate_seat0", "game/win_rate_seat3",
        "game/mean_vp_at_end", "game/games_completed",
        "eval/win_rate_vs_random", "eval/win_rate_vs_greedy",
    }
    missing = required - tags
    assert not missing, f"missing TB scalars: {missing}"

    # Two iterations of train scalars
    assert len(acc.Scalars("train/policy_loss")) == 2


def test_resume_from_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run"
    t1 = SelfPlayTrainer(TINY_CONFIG, run_dir=run_dir)
    t1.train()
    t1.close()
    it_after_first = t1.iteration

    t2 = SelfPlayTrainer(TINY_CONFIG, run_dir=run_dir, resume=True)
    assert t2.iteration == it_after_first
    t2.close()
