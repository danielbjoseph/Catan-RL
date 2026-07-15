"""
PettingZoo-style AEC (Agent Environment Cycle) environment for Catan.

API surface:
  env = CatanAECEnv(obs_mode="self_play", rules_profile="standard")
  env.reset(seed=42)
  while not all(env.terminations.values()):
      agent = env.agent_selection
      obs, rew, term, trunc, info = env.last()
      action = policy(obs, info["action_mask"])   # catalog index int
      env.step(action)

agent names: "player_0" .. "player_3"

Rewards:
  +1 to the winner, -1 to all others, emitted at game end.
  0 at every intermediate step.

Infos always contain "action_mask": np.ndarray of shape (256,) bool.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .action_mask import legal_action_mask
from .actions import CATALOG, CATALOG_SIZE
from .belief import BeliefTracker
from .board import BoardConfig
from .game_state import GameState, Phase
from .observation import obs_dim_for_mode, make_observation
from .rules import apply_action

_AGENTS = [f"player_{i}" for i in range(4)]


class CatanAECEnv:
    """
    PettingZoo-style AEC environment for 4-player Catan.

    Parameters
    ----------
    obs_mode : "self_play" (Mode B, default) or "perfect" (Mode A)
    reward_win : float   reward given to the winner (default +1)
    reward_loss : float  reward given to losers (default -1)
    max_turns : int      hard turn limit before truncation (default 500)
    """

    metadata = {"name": "catan_aec_v0", "is_parallelizable": False}

    def __init__(
        self,
        obs_mode: str = "self_play",
        reward_win: float = 1.0,
        reward_loss: float = -1.0,
        max_turns: int = 500,
        rules_profile=None,
        belief_blend: float = 0.25,
        belief_noise: float = 0.5,
    ):
        from .rules_profile import RulesProfile

        self.rules_profile = RulesProfile.get(rules_profile)
        self.obs_mode = obs_mode
        self.reward_win = reward_win
        self.reward_loss = reward_loss
        self.max_turns = max_turns
        self.belief_blend = belief_blend
        self.belief_noise = belief_noise

        self.possible_agents: List[str] = list(_AGENTS)
        self.observation_space_dim: int = obs_dim_for_mode(obs_mode)
        self.action_space_size: int = CATALOG_SIZE

        # Mutable env state — initialised in reset()
        self._state: Optional[GameState] = None
        self._rng: Optional[random.Random] = None
        self._belief: Optional[BeliefTracker] = None
        self._seed: int = 0
        self.agents: List[str] = []
        self.agent_selection: str = ""
        self.rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, Dict[str, Any]] = {}
        self._cumulative_rewards: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Core PettingZoo AEC methods
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> None:
        self._seed = seed if seed is not None else 0
        self._rng = random.Random(seed)
        board_seed = self._rng.randint(0, 2**31)
        config = BoardConfig.standard(seed=board_seed)
        game_seed = self._rng.randint(0, 2**31)
        self._state = GameState.new_game(
            config, n_players=4, seed=game_seed, profile=self.rules_profile
        )
        self._belief = BeliefTracker(self._state) if self.obs_mode == "realistic" else None

        self.agents = list(self.possible_agents)
        self._sync_agent_selection()

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {"action_mask": self._make_mask(i)} for i, a in enumerate(self.agents)}

    def step(self, action: int) -> None:
        """
        Apply a catalog action index for the current agent.

        After stepping, rewards/terminations/truncations are updated and
        agent_selection advances to the next acting player.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        state = self._state
        agent = self.agent_selection

        # Clear this agent's per-step reward
        self.rewards = {a: 0.0 for a in self.agents}

        if self.terminations[agent] or self.truncations[agent]:
            # Dead-step: do nothing (PZ convention)
            self._accumulate_rewards()
            return

        # Validate and apply the action
        if not (0 <= action < CATALOG_SIZE):
            raise ValueError(f"action {action} out of range [0, {CATALOG_SIZE})")
        catalog_action = CATALOG[action]
        if self._belief is not None:
            before = state.clone()
            apply_action(state, catalog_action, self._rng)
            self._belief.on_action(before, catalog_action, state)
        else:
            apply_action(state, catalog_action, self._rng)

        # Check terminal conditions
        if state.is_terminal:
            winner = state.winner
            for i, a in enumerate(self.agents):
                if i == winner:
                    self.rewards[a] = self.reward_win
                else:
                    self.rewards[a] = self.reward_loss
                self.terminations[a] = True
            self._accumulate_rewards()
            return

        if state.turn_number >= self.max_turns:
            for a in self.agents:
                self.truncations[a] = True
            self._accumulate_rewards()
            return

        # Advance to the next acting agent and update infos
        self._sync_agent_selection()
        self._update_infos()
        self._accumulate_rewards()

    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        """
        Return a dict with keys "observation" and "action_mask" for the given agent.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before observe().")
        pid = int(agent.split("_")[1])
        if self.obs_mode == "realistic":
            noise_cfg = {
                "belief_blend": self.belief_blend,
                "belief_noise": self.belief_noise,
                "seed": self._seed,
            }
            obs = make_observation(
                self._state, observer=pid, mode=self.obs_mode,
                belief=self._belief, noise_cfg=noise_cfg,
            )
        else:
            obs = make_observation(self._state, observer=pid, mode=self.obs_mode)
        mask = self._make_mask(pid)
        return {"observation": obs, "action_mask": mask}

    def last(self, observe: bool = True) -> Tuple[Optional[Dict], float, bool, bool, dict]:
        """
        Return (observation, cumulative_reward, termination, truncation, info)
        for the current agent_selection. Mirrors PettingZoo's last() convention.
        """
        agent = self.agent_selection
        obs = self.observe(agent) if observe else None
        return (
            obs,
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def render(self, mode: str = "text") -> Optional[str]:
        if self._state is None:
            return None
        return self._text_render()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_agent_selection(self) -> None:
        self.agent_selection = f"player_{self._state.current_player}"

    def _make_mask(self, pid: int) -> np.ndarray:
        """Action mask for the given player. Only legal when it is their turn."""
        if self._state is None or self._state.current_player != pid or self._state.is_terminal:
            return np.zeros(CATALOG_SIZE, dtype=bool)
        return legal_action_mask(self._state)

    def _update_infos(self) -> None:
        for i, a in enumerate(self.agents):
            self.infos[a] = {"action_mask": self._make_mask(i)}

    def _accumulate_rewards(self) -> None:
        for a in self.agents:
            self._cumulative_rewards[a] += self.rewards[a]

    def _text_render(self) -> str:
        s = self._state
        lines = [
            f"Turn {s.turn_number}  Phase: {s.phase.name}  Current: player_{s.current_player}",
            f"Robber: hex {s.robber_hex}",
        ]
        if s.dice:
            lines.append(f"Dice: {s.dice[0]}+{s.dice[1]}={sum(s.dice)}")
        for p in s.players:
            res = dict(zip(["W","Br","Sh","Wh","Or"], p.resources))
            lines.append(
                f"  P{p.player_id}: VP={p.public_vp}  res={res}  "
                f"rds={p.roads_built} sets={p.settlements_built} cities={p.cities_built}"
            )
        if s.winner is not None:
            lines.append(f"*** GAME OVER — winner: player_{s.winner} ***")
        return "\n".join(lines)
