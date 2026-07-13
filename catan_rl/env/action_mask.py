"""
Legal action mask generator.

legal_action_mask(state) -> np.ndarray of shape (256,) dtype bool

True at index i means CATALOG[i] is legal in the current state.
Illegal slots (including the two padding slots 254-255) remain False.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .actions import CATALOG_SIZE
from .validators import legal_actions

if TYPE_CHECKING:
    from .game_state import GameState


def legal_action_mask(state: "GameState") -> np.ndarray:
    """Return a bool array of shape (CATALOG_SIZE,) with True at legal action indices."""
    mask = np.zeros(CATALOG_SIZE, dtype=bool)
    for action in legal_actions(state):
        if 0 <= action.catalog_index < CATALOG_SIZE:
            mask[action.catalog_index] = True
    return mask
