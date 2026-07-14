"""
Smoke tests for CatanAECEnv and CatanGymEnv.

Verifies:
  - many games run to completion without error
  - legal action masks prevent impossible moves
  - observations are consistent each step
  - rewards are correct at game end
  - terminal detection works
"""

import random

import numpy as np
import pytest

from catan_rl.env.action_mask import legal_action_mask
from catan_rl.env.actions import CATALOG_SIZE
from catan_rl.env.observation import OBS_DIM, OBS_DIM_PERFECT
from catan_rl.env.pettingzoo_env import CatanAECEnv
from catan_rl.env.gym_wrapper import CatanGymEnv


def _random_action(obs_dict: dict) -> int:
    mask = obs_dict["action_mask"]
    legal = np.where(mask)[0]
    if len(legal) == 0:
        return 0
    return int(np.random.choice(legal))


# ---------------------------------------------------------------------------
# AEC environment smoke tests
# ---------------------------------------------------------------------------

class TestAECSmoke:
    def _run_game(self, seed: int) -> dict:
        """Run one full game to completion; return stats."""
        env = CatanAECEnv()
        env.reset(seed=seed)
        steps = 0
        while not all(env.terminations.values()) and not all(env.truncations.values()):
            agent = env.agent_selection
            obs_dict = env.observe(agent)
            action = _random_action(obs_dict)
            env.step(action)
            steps += 1
        return {
            "steps": steps,
            "terminations": dict(env.terminations),
            "truncations": dict(env.truncations),
            "cumulative_rewards": dict(env._cumulative_rewards),
        }

    def test_single_game_completes(self):
        stats = self._run_game(seed=0)
        assert stats["steps"] > 0

    def test_multiple_games_complete(self):
        for seed in range(5):
            stats = self._run_game(seed=seed)
            assert stats["steps"] > 0, f"Game {seed} completed in 0 steps"

    def test_exactly_one_winner(self):
        for seed in range(5):
            env = CatanAECEnv()
            env.reset(seed=seed)
            while not all(env.terminations.values()) and not all(env.truncations.values()):
                agent = env.agent_selection
                obs_dict = env.observe(agent)
                env.step(_random_action(obs_dict))
            terms = env.terminations
            truncs = env.truncations
            if all(terms.values()):
                # Should have one winner with +1 and others -1
                rewards = list(env._cumulative_rewards.values())
                winners = [r for r in rewards if r > 0]
                losers  = [r for r in rewards if r < 0]
                assert len(winners) == 1, f"Expected 1 winner, got rewards {rewards}"
                assert len(losers) == 3

    def test_observations_correct_shape(self):
        env = CatanAECEnv(obs_mode="self_play")
        env.reset(seed=1)
        for agent in env.agents:
            obs_dict = env.observe(agent)
            assert obs_dict["observation"].shape == (OBS_DIM,)
            assert obs_dict["action_mask"].shape == (CATALOG_SIZE,)

    def test_perfect_mode_obs_shape(self):
        env = CatanAECEnv(obs_mode="perfect")
        env.reset(seed=2)
        for agent in env.agents:
            obs_dict = env.observe(agent)
            assert obs_dict["observation"].shape == (OBS_DIM_PERFECT,)

    def test_mask_always_has_legal_action(self):
        env = CatanAECEnv()
        env.reset(seed=3)
        violations = 0
        steps = 0
        while not all(env.terminations.values()) and not all(env.truncations.values()):
            agent = env.agent_selection
            obs_dict = env.observe(agent)
            mask = obs_dict["action_mask"]
            if not mask.any():
                violations += 1
            env.step(_random_action(obs_dict))
            steps += 1
        assert violations == 0, f"{violations}/{steps} steps had empty mask"

    def test_action_mask_only_true_for_current_player(self):
        env = CatanAECEnv()
        env.reset(seed=4)
        agent = env.agent_selection
        for other in env.agents:
            if other != agent:
                other_obs = env.observe(other)
                assert not other_obs["action_mask"].any(), (
                    f"Non-current player {other} should have empty mask"
                )

    def test_reset_clears_state(self):
        env = CatanAECEnv()
        env.reset(seed=5)
        # Run a few steps
        for _ in range(20):
            if all(env.terminations.values()):
                break
            agent = env.agent_selection
            env.step(_random_action(env.observe(agent)))
        # Reset to a fresh game
        env.reset(seed=0)
        assert env._state.turn_number == 0
        assert all(r == 0.0 for r in env._cumulative_rewards.values())

    def test_no_nan_in_observations(self):
        env = CatanAECEnv()
        env.reset(seed=7)
        for _ in range(50):
            if all(env.terminations.values()):
                break
            agent = env.agent_selection
            obs_dict = env.observe(agent)
            assert not np.any(np.isnan(obs_dict["observation"])), "NaN in observation"
            env.step(_random_action(obs_dict))

    def test_render(self):
        env = CatanAECEnv()
        env.reset(seed=8)
        text = env.render()
        assert isinstance(text, str)
        assert "Phase" in text

    def test_last_returns_correct_types(self):
        env = CatanAECEnv()
        env.reset(seed=9)
        obs_dict, reward, term, trunc, info = env.last()
        assert isinstance(obs_dict, dict)
        assert "observation" in obs_dict
        assert "action_mask" in obs_dict
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)

    def test_ten_games_no_crash(self):
        for seed in range(10):
            env = CatanAECEnv()
            env.reset(seed=seed)
            while not all(env.terminations.values()) and not all(env.truncations.values()):
                agent = env.agent_selection
                env.step(_random_action(env.observe(agent)))


# ---------------------------------------------------------------------------
# Gym wrapper smoke tests
# ---------------------------------------------------------------------------

class TestGymWrapper:
    def test_reset_returns_obs_and_info(self):
        env = CatanGymEnv()
        obs, info = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (OBS_DIM,)
        assert "action_mask" in info
        assert info["action_mask"].shape == (CATALOG_SIZE,)

    def test_step_returns_correct_types(self):
        env = CatanGymEnv()
        obs, info = env.reset(seed=1)
        mask = info["action_mask"]
        action = int(np.where(mask)[0][0])
        obs2, reward, term, trunc, info2 = env.step(action)
        assert obs2.shape == (OBS_DIM,)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)

    def test_full_game_completes(self):
        env = CatanGymEnv()
        obs, info = env.reset(seed=2)
        done = False
        steps = 0
        while not done and steps < 2000:
            mask = info["action_mask"]
            legal = np.where(mask)[0]
            action = int(np.random.choice(legal)) if len(legal) else 0
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        assert done, "Game should complete within step limit"

    def test_reward_is_nonzero_at_end(self):
        # Seeded action RNG (global: the wrapper's opponent policy uses
        # np.random too). An unseeded run can legitimately truncate at the
        # turn cap (reward 0), which is not what this test is about.
        np.random.seed(1)
        rng = np.random.default_rng(1)
        env = CatanGymEnv()
        obs, info = env.reset(seed=3)
        final_reward = 0.0
        done = terminated = False
        while not done:
            mask = info["action_mask"]
            legal = np.where(mask)[0]
            action = int(rng.choice(legal)) if len(legal) else 0
            obs, reward, terminated, trunc, info = env.step(action)
            final_reward = reward
            done = terminated or trunc
        assert terminated, "Seeded game should terminate with a winner, not truncate"
        assert final_reward != 0.0, "Terminal reward should be non-zero"

    def test_no_nan_in_gym_obs(self):
        env = CatanGymEnv()
        obs, info = env.reset(seed=4)
        assert not np.any(np.isnan(obs))
        for _ in range(30):
            mask = info["action_mask"]
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            obs, reward, term, trunc, info = env.step(int(np.random.choice(legal)))
            assert not np.any(np.isnan(obs)), "NaN in gym observation"
            if term or trunc:
                break
