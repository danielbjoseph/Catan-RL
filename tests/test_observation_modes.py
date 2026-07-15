"""
Tests for realistic + global observation modes.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from catan_rl.bots.random_bot import pick_action
from catan_rl.env.belief import BeliefTracker
from catan_rl.env.board import BoardConfig
from catan_rl.env.game_state import GameState
from catan_rl.env.observation import (
    OBS_DIM,
    OBS_DIM_GLOBAL,
    OBS_DIM_PERFECT,
    OBS_DIM_REALISTIC,
    apply_belief_noise,
    make_observation,
    obs_dim_for_mode,
)
from catan_rl.env.rules import apply_action
from catan_rl.env.rules_profile import STANDARD

SEED = 0


def _play_random(seed=SEED, n_plies=50):
    rng = random.Random(seed)
    config = BoardConfig.standard(seed=seed)
    state = GameState.new_game(config, n_players=4, seed=seed, profile=STANDARD)
    tracker = BeliefTracker(state)
    plies = 0
    while not state.is_terminal and plies < n_plies:
        action = pick_action(state, rng)
        before = state.clone()
        apply_action(state, action, rng)
        tracker.on_action(before, action, state)
        plies += 1
    return state, tracker


# ---------------------------------------------------------------------------
# 1. Dimensions
# ---------------------------------------------------------------------------

class TestDims:
    @pytest.mark.parametrize("mode", ["self_play", "perfect", "realistic", "global"])
    def test_dim_matches_obs_dim_for_mode(self, mode):
        state, tracker = _play_random()
        belief = tracker if mode == "realistic" else None
        obs = make_observation(state, observer=0, mode=mode, belief=belief)
        assert obs.shape[0] == obs_dim_for_mode(mode)

    def test_constants(self):
        assert OBS_DIM == 1520
        assert OBS_DIM_PERFECT == 1565
        assert OBS_DIM_REALISTIC == 1549
        assert OBS_DIM_GLOBAL == 1576


# ---------------------------------------------------------------------------
# 2. Regression: bases must remain byte-identical
# ---------------------------------------------------------------------------

class TestRegressionBasesUntouched:
    def test_realistic_prefix_equals_self_play(self):
        state, tracker = _play_random(seed=1, n_plies=40)
        obs_sp = make_observation(state, observer=0, mode="self_play")
        obs_real = make_observation(state, observer=0, mode="realistic", belief=tracker)
        assert obs_sp.shape[0] == OBS_DIM
        assert np.array_equal(obs_real[:OBS_DIM], obs_sp)

    def test_global_prefix_equals_perfect(self):
        state, tracker = _play_random(seed=1, n_plies=40)
        obs_pf = make_observation(state, observer=0, mode="perfect")
        obs_glob = make_observation(state, observer=0, mode="global")
        assert obs_pf.shape[0] == OBS_DIM_PERFECT
        assert np.array_equal(obs_glob[:OBS_DIM_PERFECT], obs_pf)

    def test_other_observers_too(self):
        state, tracker = _play_random(seed=2, n_plies=40)
        for observer in range(4):
            obs_sp = make_observation(state, observer=observer, mode="self_play")
            obs_real = make_observation(state, observer=observer, mode="realistic", belief=tracker)
            assert np.array_equal(obs_real[:OBS_DIM], obs_sp)

            obs_pf = make_observation(state, observer=observer, mode="perfect")
            obs_glob = make_observation(state, observer=observer, mode="global")
            assert np.array_equal(obs_glob[:OBS_DIM_PERFECT], obs_pf)


# ---------------------------------------------------------------------------
# 3. Realistic belief features
# ---------------------------------------------------------------------------

class TestRealisticBeliefFeatures:
    def _opponent_block(self, obs, rel_i):
        off = OBS_DIM + (rel_i - 1) * 6
        return obs[off:off + 5], obs[off + 5]

    def test_matches_tracker_when_noise_cfg_none(self):
        state, tracker = _play_random(seed=3, n_plies=30)
        obs = make_observation(state, observer=0, mode="realistic", belief=tracker, noise_cfg=None)
        for rel_i in range(1, 4):
            pid = (0 + rel_i) % 4
            res_block, unc = self._opponent_block(obs, rel_i)
            np.testing.assert_array_equal(res_block, (tracker.expected(pid) / 19.0).astype(np.float32))
            assert unc == pytest.approx(tracker.uncertainty(pid), abs=1e-6)

    def test_matches_tracker_when_blend_and_sigma_zero(self):
        state, tracker = _play_random(seed=3, n_plies=30)
        noise_cfg = {"belief_blend": 0.0, "belief_noise": 0.0, "seed": 7}
        obs = make_observation(state, observer=0, mode="realistic", belief=tracker, noise_cfg=noise_cfg)
        for rel_i in range(1, 4):
            pid = (0 + rel_i) % 4
            res_block, unc = self._opponent_block(obs, rel_i)
            np.testing.assert_array_equal(res_block, (tracker.expected(pid) / 19.0).astype(np.float32))

    def test_dev_deck_and_bank_blocks(self):
        state, tracker = _play_random(seed=4, n_plies=30)
        obs = make_observation(state, observer=0, mode="realistic", belief=tracker)
        dev_off = OBS_DIM + 18
        comp, count = tracker.dev_deck_estimate(0, state)
        np.testing.assert_allclose(obs[dev_off:dev_off + 5], comp / 14.0, atol=1e-6)
        assert obs[dev_off + 5] == pytest.approx(count / 25.0, abs=1e-6)

        bank_off = dev_off + 6
        expected_bank = np.array(state.bank, dtype=np.float32) / 19.0
        np.testing.assert_allclose(obs[bank_off:bank_off + 5], expected_bank, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Noise determinism
# ---------------------------------------------------------------------------

class TestApplyBeliefNoise:
    def test_no_noise_passthrough(self):
        vec = np.array([2.0, 1.0, 0.0, 3.0, 0.0], dtype=np.float32)
        out = apply_belief_noise(vec, 6, blend=0.0, sigma=0.0, key=(42, 5, 0))
        assert np.array_equal(out, vec)

    def test_same_key_identical(self):
        vec = np.array([2.0, 1.0, 0.0, 3.0, 0.0], dtype=np.float32)
        out1 = apply_belief_noise(vec, 6, blend=0.3, sigma=0.5, key=(42, 5, 0))
        out2 = apply_belief_noise(vec, 6, blend=0.3, sigma=0.5, key=(42, 5, 0))
        assert np.array_equal(out1, out2)

    def test_different_observer_key_differs(self):
        vec = np.array([2.0, 1.0, 0.0, 3.0, 0.0], dtype=np.float32)
        out0 = apply_belief_noise(vec, 6, blend=0.3, sigma=0.5, key=(42, 5, 0))
        out1 = apply_belief_noise(vec, 6, blend=0.3, sigma=0.5, key=(42, 5, 1))
        assert not np.array_equal(out0, out1)

    def test_sum_preserved(self):
        vec = np.array([2.0, 1.0, 0.0, 3.0, 0.0], dtype=np.float32)
        out = apply_belief_noise(vec, 6, blend=0.3, sigma=0.5, key=(42, 5, 0))
        assert float(out.sum()) == pytest.approx(6.0, abs=1e-4)

    def test_non_negative(self):
        vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        out = apply_belief_noise(vec, 3, blend=0.0, sigma=2.0, key=(1, 1, 1))
        assert np.all(out >= 0.0)

    def test_make_observation_deterministic_with_noise_cfg(self):
        state, tracker = _play_random(seed=5, n_plies=30)
        noise_cfg = {"belief_blend": 0.3, "belief_noise": 0.5, "seed": 99}
        obs1 = make_observation(state, observer=0, mode="realistic", belief=tracker, noise_cfg=noise_cfg)
        obs2 = make_observation(state, observer=0, mode="realistic", belief=tracker, noise_cfg=noise_cfg)
        assert np.array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# 5. Global deck features
# ---------------------------------------------------------------------------

class TestGlobalDeckFeatures:
    def test_matches_dev_deck_multiset(self):
        state, _ = _play_random(seed=6, n_plies=30)
        obs = make_observation(state, observer=0, mode="global")
        base = OBS_DIM_PERFECT
        counts = np.zeros(5, dtype=np.float32)
        for c in state.dev_deck:
            counts[int(c)] += 1
        np.testing.assert_allclose(obs[base:base + 5], counts / 14.0, atol=1e-6)
        assert obs[base + 5] == pytest.approx(len(state.dev_deck) / 25.0, abs=1e-6)

        bank_off = base + 6
        expected_bank = np.array(state.bank, dtype=np.float32) / 19.0
        np.testing.assert_allclose(obs[bank_off:bank_off + 5], expected_bank, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Errors
# ---------------------------------------------------------------------------

class TestErrors:
    def test_realistic_requires_belief(self):
        state, _ = _play_random(seed=7, n_plies=10)
        with pytest.raises(ValueError):
            make_observation(state, observer=0, mode="realistic")

    def test_obs_dim_for_mode_unknown_raises(self):
        with pytest.raises(ValueError):
            obs_dim_for_mode("bogus")

    def test_make_observation_unknown_mode_dim_helper_still_raises(self):
        with pytest.raises(ValueError):
            obs_dim_for_mode("not_a_mode")
