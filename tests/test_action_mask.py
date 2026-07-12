"""
Tests for the legal action mask generator.
"""

import random

import numpy as np
import pytest

from catan_rl.env.action_mask import legal_action_mask
from catan_rl.env.actions import CATALOG, CATALOG_SIZE, ActionType
from catan_rl.env.board import BoardConfig
from catan_rl.env.game_state import GameState, Phase
from catan_rl.env.rules import apply_action
from catan_rl.env.validators import legal_actions


def make_state(seed=0):
    config = BoardConfig.standard(seed=seed)
    return GameState.new_game(config, n_players=4, seed=seed)


def fast_forward_setup(state, rng=None):
    if rng is None:
        rng = random.Random(42)
    while state.phase not in (Phase.ROLL, Phase.GAME_OVER):
        acts = legal_actions(state)
        apply_action(state, rng.choice(acts), rng)


class TestMaskShape:
    def test_shape(self):
        state = make_state()
        mask = legal_action_mask(state)
        assert mask.shape == (CATALOG_SIZE,)

    def test_dtype(self):
        state = make_state()
        mask = legal_action_mask(state)
        assert mask.dtype == bool

    def test_padding_always_false(self):
        state = make_state()
        mask = legal_action_mask(state)
        assert not mask[254], "Padding slot 254 must never be legal"
        assert not mask[255], "Padding slot 255 must never be legal"


class TestMaskCorrectness:
    def test_setup_only_settlements_legal(self):
        state = make_state()
        assert state.phase == Phase.SETUP_SETTLEMENT_1
        mask = legal_action_mask(state)
        # Only settlement slots (74-127) should be True
        assert mask[0] == False, "ROLL_DICE illegal during setup"
        assert mask[1] == False, "END_TURN illegal during setup"
        assert any(mask[74:128]), "At least one settlement slot must be legal"
        assert not any(mask[2:74]), "Road slots must be illegal during settlement phase"

    def test_mask_matches_legal_actions_count(self):
        state = make_state()
        rng = random.Random(3)
        fast_forward_setup(state, rng)
        acts = legal_actions(state)
        mask = legal_action_mask(state)
        assert mask.sum() == len(acts), (
            f"Mask has {mask.sum()} True slots but {len(acts)} legal actions"
        )

    def test_mask_matches_legal_actions_indices(self):
        state = make_state()
        rng = random.Random(4)
        fast_forward_setup(state, rng)
        acts = legal_actions(state)
        expected_indices = {a.catalog_index for a in acts}
        mask = legal_action_mask(state)
        actual_indices = set(np.where(mask)[0].tolist())
        assert expected_indices == actual_indices

    def test_roll_dice_legal_at_roll_phase(self):
        state = make_state()
        rng = random.Random(1)
        fast_forward_setup(state, rng)
        assert state.phase == Phase.ROLL
        mask = legal_action_mask(state)
        assert mask[0] == True, "ROLL_DICE (index 0) must be legal in ROLL phase"

    def test_end_turn_always_available_in_main(self):
        state = make_state()
        rng = random.Random(2)
        fast_forward_setup(state, rng)
        apply_action(state, legal_actions(state)[0], rng)  # roll
        if state.phase == Phase.MAIN:
            mask = legal_action_mask(state)
            assert mask[1] == True, "END_TURN (index 1) must be legal in MAIN phase"

    def test_mask_has_at_least_one_true(self):
        state = make_state()
        rng = random.Random(6)
        # Run several turns and check mask is never all-False while game is running
        for _ in range(40):
            if state.phase == Phase.GAME_OVER:
                break
            mask = legal_action_mask(state)
            assert mask.any(), f"Mask is all-False in phase {state.phase}"
            acts = legal_actions(state)
            apply_action(state, rng.choice(acts), rng)

    def test_mask_consistent_across_phases(self):
        """Run a full game and verify mask always matches legal_actions."""
        state = make_state(seed=99)
        rng = random.Random(99)
        mismatches = 0
        for _ in range(300):
            if state.phase == Phase.GAME_OVER:
                break
            acts = legal_actions(state)
            mask = legal_action_mask(state)
            expected = {a.catalog_index for a in acts}
            actual = set(np.where(mask)[0].tolist())
            if expected != actual:
                mismatches += 1
            apply_action(state, rng.choice(acts), rng)
        assert mismatches == 0, f"{mismatches} steps had mask/legal_actions mismatch"
