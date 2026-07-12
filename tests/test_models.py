"""Tests for the masked actor-critic network."""

import numpy as np
import pytest
import torch

from catan_rl.env.observation import OBS_DIM
from catan_rl.rl.models import ActorCritic, masked_logits


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return ActorCritic(obs_dim=OBS_DIM, n_actions=256, hidden_sizes=(64, 64))


def _rand_obs(batch=4):
    return torch.randn(batch, OBS_DIM)


def _mask_with_legal(legal_indices, batch=4):
    mask = torch.zeros(batch, 256, dtype=torch.bool)
    mask[:, legal_indices] = True
    return mask


class TestForward:
    def test_shapes(self, net):
        logits, value = net(_rand_obs(3))
        assert logits.shape == (3, 256)
        assert value.shape == (3,)

    def test_masked_logits_kills_illegal(self):
        logits = torch.zeros(2, 256)
        mask = _mask_with_legal([0, 5, 100], batch=2)
        ml = masked_logits(logits, mask)
        probs = torch.softmax(ml, dim=-1)
        illegal_prob = probs[:, ~mask[0]].sum()
        assert illegal_prob < 1e-6


class TestAct:
    def test_never_samples_illegal(self, net):
        legal = [3, 77, 200]
        obs = _rand_obs(1).repeat(512, 1)
        mask = _mask_with_legal(legal, batch=512)
        actions, logprobs, values = net.act(obs, mask)
        assert actions.shape == (512,)
        assert set(actions.tolist()) <= set(legal)
        assert torch.isfinite(logprobs).all()
        assert torch.isfinite(values).all()

    def test_deterministic_is_argmax_over_legal(self, net):
        obs = _rand_obs(1)
        mask = _mask_with_legal([10, 20, 30], batch=1)
        action, _, _ = net.act(obs, mask, deterministic=True)
        logits, _ = net(obs)
        ml = masked_logits(logits, mask)
        assert action.item() == ml.argmax(dim=-1).item()
        assert action.item() in (10, 20, 30)

    def test_single_legal_action_logprob_zero(self, net):
        obs = _rand_obs(1)
        mask = _mask_with_legal([42], batch=1)
        action, logprob, _ = net.act(obs, mask)
        assert action.item() == 42
        assert abs(logprob.item()) < 1e-5


class TestEvaluateActions:
    def test_matches_act_logprob(self, net):
        obs = _rand_obs(8)
        mask = _mask_with_legal([1, 2, 3, 74, 128], batch=8)
        actions, logprobs, values = net.act(obs, mask)
        lp2, entropy, v2 = net.evaluate_actions(obs, mask, actions)
        assert torch.allclose(logprobs, lp2, atol=1e-5)
        assert torch.allclose(values, v2, atol=1e-5)
        assert entropy.shape == (8,)

    def test_entropy_bounded_by_log_n_legal(self, net):
        obs = _rand_obs(4)
        legal = [0, 1, 2, 3, 4]
        mask = _mask_with_legal(legal, batch=4)
        actions, _, _ = net.act(obs, mask)
        _, entropy, _ = net.evaluate_actions(obs, mask, actions)
        assert (entropy <= np.log(len(legal)) + 1e-5).all()
        assert (entropy >= 0).all()
