"""Tests for the custom PPO trainer."""

from pathlib import Path

import pytest
import torch

from catan_rl.rl.models import ActorCritic
from catan_rl.rl.ppo import PPOConfig, PPOTrainer
from catan_rl.rl.rollout import Batch

OBS_DIM_SMALL = 32
N_ACTIONS = 256
STAT_KEYS = {"policy_loss", "value_loss", "entropy", "approx_kl",
             "clip_fraction", "learning_rate"}


def _synthetic_batch(n=512, seed=0, favored_action=5):
    """Batch where `favored_action` always has positive advantage."""
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, OBS_DIM_SMALL, generator=g)
    masks = torch.zeros(n, N_ACTIONS, dtype=torch.bool)
    masks[:, :10] = True
    actions = torch.randint(0, 10, (n,), generator=g)
    advantages = torch.where(actions == favored_action, 1.0, -0.2)
    return Batch(
        obs=obs,
        masks=masks,
        actions=actions,
        logprobs=torch.full((n,), -torch.log(torch.tensor(10.0))),
        values=torch.zeros(n),
        advantages=advantages,
        returns=advantages.clone(),
        seat_ids=torch.zeros(n, dtype=torch.long),
        episode_ids=torch.zeros(n, dtype=torch.long),
        stats={},
    )


@pytest.fixture
def trainer():
    torch.manual_seed(0)
    policy = ActorCritic(obs_dim=OBS_DIM_SMALL, hidden_sizes=(32, 32))
    cfg = PPOConfig(lr=1e-3, epochs=2, minibatch_size=128)
    return PPOTrainer(policy, cfg)


class TestPPOConfig:
    def test_defaults(self):
        cfg = PPOConfig()
        assert cfg.clip_coef == 0.2
        assert cfg.gamma == 0.999

    def test_from_yaml(self, tmp_path: Path):
        p = tmp_path / "ppo.yaml"
        p.write_text("lr: 0.001\nclip_coef: 0.3\nhidden_sizes: [128, 128]\n")
        cfg = PPOConfig.from_yaml(p)
        assert cfg.lr == 0.001
        assert cfg.clip_coef == 0.3
        assert tuple(cfg.hidden_sizes) == (128, 128)
        assert cfg.epochs == 4  # untouched default

    def test_from_yaml_ignores_run_settings(self, tmp_path: Path):
        """Trainer-level keys like experiment_name must not break parsing."""
        p = tmp_path / "ppo.yaml"
        p.write_text("lr: 0.001\nexperiment_name: foo\niterations: 10\n")
        cfg = PPOConfig.from_yaml(p)
        assert cfg.lr == 0.001


class TestUpdate:
    def test_returns_all_stats_finite(self, trainer):
        stats = trainer.update(_synthetic_batch())
        assert STAT_KEYS <= set(stats)
        for k in STAT_KEYS:
            assert stats[k] == stats[k], f"{k} is NaN"  # NaN check

    def test_favored_action_probability_increases(self, trainer):
        batch = _synthetic_batch()
        obs, masks = batch.obs[:64], batch.masks[:64]

        def favored_prob():
            with torch.no_grad():
                logits, _ = trainer.policy(obs)
                from catan_rl.rl.models import masked_logits
                probs = torch.softmax(masked_logits(logits, masks), dim=-1)
            return probs[:, 5].mean().item()

        before = favored_prob()
        for _ in range(20):
            trainer.update(_synthetic_batch())
        after = favored_prob()
        assert after > before, f"favored action prob did not increase: {before} -> {after}"

    def test_value_regression(self, trainer):
        """Value head should move toward the constant return target."""
        batch = _synthetic_batch()
        batch.returns.fill_(3.0)
        batch.advantages.fill_(0.0)
        for _ in range(30):
            trainer.update(batch)
        with torch.no_grad():
            _, v = trainer.policy(batch.obs[:64])
        assert (v - 3.0).abs().mean() < 1.0
