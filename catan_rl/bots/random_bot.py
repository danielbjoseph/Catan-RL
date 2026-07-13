"""Random-legal bot: picks uniformly from all legal actions."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..env.game_state import GameState
    from ..env.actions import Action


def pick_action(state: "GameState", rng: random.Random | None = None) -> "Action":
    from ..env.validators import legal_actions
    if rng is None:
        rng = random.Random()
    actions = legal_actions(state)
    if not actions:
        raise RuntimeError(f"No legal actions in phase {state.phase} for player {state.current_player}")
    return rng.choice(actions)
