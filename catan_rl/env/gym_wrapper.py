"""
Thin Gym-compatible wrapper around CatanAECEnv.

Presents the environment as a single-agent Gym env from the perspective of
whichever player is currently acting.  Suitable for current-player RL training.

Observation space: Box(shape=(OBS_DIM,), dtype=float32)
Action space:      Discrete(256)

The observation dict returned by the AEC env is unwrapped so that:
  obs  = np.ndarray  shape (OBS_DIM,)
  info = {"action_mask": np.ndarray(256, bool), "current_player": int}
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .observation import obs_dim_for_mode
from .actions import CATALOG_SIZE
from .pettingzoo_env import CatanAECEnv


class CatanGymEnv:
    """
    Single-agent Gym-style wrapper for CatanAECEnv.

    Step interface:
      obs, reward, terminated, truncated, info = env.step(action_index)

    The wrapper steps the AEC env until it is the *same* player's turn again
    (or the game ends), so from the caller's perspective it feels like a
    single-agent environment.  All intermediate transitions by other players
    are handled internally with the `opponent_policy` callable.

    Parameters
    ----------
    opponent_policy : callable(obs_dict) -> int, optional
        Policy used for opponents.  Defaults to random-legal play.
    obs_mode : str
        "self_play" or "perfect".
    """

    def __init__(
        self,
        opponent_policy=None,
        obs_mode: str = "self_play",
        reward_win: float = 1.0,
        reward_loss: float = -1.0,
        max_turns: int = 500,
        rules_profile=None,
        belief_blend: float = 0.25,
        belief_noise: float = 0.5,
    ):
        self._aec = CatanAECEnv(
            obs_mode=obs_mode,
            reward_win=reward_win,
            reward_loss=reward_loss,
            max_turns=max_turns,
            rules_profile=rules_profile,
            belief_blend=belief_blend,
            belief_noise=belief_noise,
        )
        self._opponent_policy = opponent_policy or _random_legal_policy
        self.obs_dim = obs_dim_for_mode(obs_mode)
        self.action_space_size = CATALOG_SIZE
        self._controlled_agent: Optional[str] = None

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, dict]:
        self._aec.reset(seed=seed)
        # Player 0 is always the controlled agent in this wrapper
        self._controlled_agent = "player_0"
        # Step opponents until it is player_0's turn (setup phase starts with player_0)
        self._step_opponents_until_our_turn()
        obs_dict, _, _, _, info = self._aec.last()
        return obs_dict["observation"], {
            "action_mask": obs_dict["action_mask"],
            "current_player": 0,
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._aec.step(action)
        self._step_opponents_until_our_turn()
        obs_dict, reward, terminated, truncated, info = self._aec.last()
        obs = obs_dict["observation"] if obs_dict is not None else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, reward, terminated, truncated, {
            "action_mask": obs_dict["action_mask"] if obs_dict else np.zeros(CATALOG_SIZE, dtype=bool),
            "current_player": 0,
        }

    def render(self) -> Optional[str]:
        return self._aec.render()

    # ------------------------------------------------------------------

    def _step_opponents_until_our_turn(self) -> None:
        """Advance the AEC env through all opponent turns."""
        aec = self._aec
        while (
            not all(aec.terminations.values())
            and not all(aec.truncations.values())
            and aec.agent_selection != self._controlled_agent
        ):
            agent = aec.agent_selection
            obs_dict = aec.observe(agent)
            action = self._opponent_policy(obs_dict)
            aec.step(action)


def _random_legal_policy(obs_dict: dict) -> int:
    """Pick a random legal action from the mask."""
    mask: np.ndarray = obs_dict["action_mask"]
    legal = np.where(mask)[0]
    if len(legal) == 0:
        return 0
    return int(np.random.choice(legal))
