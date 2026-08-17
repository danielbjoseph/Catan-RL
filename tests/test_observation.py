"""
Tests for the observation generator.
"""

import random

import numpy as np
import pytest

from catan_rl.env.board import BoardConfig
from catan_rl.env.game_state import GameState, Phase
from catan_rl.env.observation import (
    OBS_DIM, OBS_DIM_PERFECT, _N_HEX, _N_VERTEX, _N_EDGE, _SEG_TRADE,
    make_observation,
)
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


class TestObservationShape:
    def test_self_play_dim(self):
        state = make_state()
        obs = make_observation(state, observer=0, mode="self_play")
        assert obs.shape == (OBS_DIM,), f"Expected {OBS_DIM}, got {obs.shape}"

    def test_perfect_dim(self):
        state = make_state()
        obs = make_observation(state, observer=0, mode="perfect")
        assert obs.shape == (OBS_DIM_PERFECT,), f"Expected {OBS_DIM_PERFECT}, got {obs.shape}"

    def test_dtype(self):
        state = make_state()
        obs = make_observation(state, observer=0)
        assert obs.dtype == np.float32

    def test_no_nan(self):
        state = make_state()
        for pid in range(4):
            obs = make_observation(state, observer=pid)
            assert not np.any(np.isnan(obs)), "NaN found in observation"

    def test_all_players_same_shape(self):
        state = make_state()
        shapes = [make_observation(state, observer=i).shape for i in range(4)]
        assert len(set(shapes)) == 1


class TestObservationValues:
    def test_robber_one_hot(self):
        state = make_state()
        obs = make_observation(state, observer=0)
        # Robber segment starts after hex_resources(114) + hex_tokens(19) + port_vertices(378) = 511
        robber_start = 114 + 19 + 378
        robber = obs[robber_start: robber_start + _N_HEX]
        assert robber.sum() == pytest.approx(1.0), "Robber must be one-hot"
        assert robber[state.robber_hex] == pytest.approx(1.0)

    def test_hex_resource_one_hot(self):
        state = make_state()
        obs = make_observation(state, observer=0)
        hex_res = obs[: _N_HEX * 6].reshape(_N_HEX, 6)
        row_sums = hex_res.sum(axis=1)
        assert np.allclose(row_sums, 1.0), "Each hex must have exactly one resource type set"

    def test_port_vertices_one_hot(self):
        state = make_state()
        obs = make_observation(state, observer=0)
        port_start = 114 + 19
        port = obs[port_start: port_start + _N_VERTEX * 7].reshape(_N_VERTEX, 7)
        row_sums = port.sum(axis=1)
        assert np.allclose(row_sums, 1.0), "Each vertex must have exactly one port type"

    def test_values_bounded(self):
        state = make_state()
        rng = random.Random(7)
        fast_forward_setup(state, rng)
        # Play a few turns
        for _ in range(20):
            if state.phase == Phase.GAME_OVER:
                break
            acts = legal_actions(state)
            apply_action(state, rng.choice(acts), rng)
        obs = make_observation(state, observer=0)
        assert obs.min() >= -0.01, "Observation values should be non-negative"
        assert obs.max() <= 1.01, "Observation values should be at most 1.0"

    def test_rotation_consistency(self):
        """Observations from different players should differ only in relative encoding."""
        state = make_state()
        rng = random.Random(3)
        fast_forward_setup(state, rng)
        obs0 = make_observation(state, observer=0)
        obs1 = make_observation(state, observer=1)
        # They should have the same shape but different content
        assert obs0.shape == obs1.shape
        assert not np.allclose(obs0, obs1), "Observations for different players should differ"

    def test_perfect_reveals_more_info(self):
        state = make_state()
        rng = random.Random(5)
        fast_forward_setup(state, rng)
        # Give player 0 some resources
        state.players[0].resources[0] = 3
        obs_sp = make_observation(state, observer=1, mode="self_play")
        obs_pf = make_observation(state, observer=1, mode="perfect")
        # Perfect mode is longer
        assert len(obs_pf) > len(obs_sp)
        # The extra section should encode player 0's resources. It starts
        # right after the shared base (OBS_DIM minus the trailing trade
        # block, which is appended last and isn't part of the base).
        extra = obs_pf[OBS_DIM - _SEG_TRADE:]
        # Player 0 has 3 wood; first opponent from observer=1 is player 2... wait
        # From observer=1, opponents in order are: player_2 (rel 1), player_3 (rel 2), player_0 (rel 3)
        # Player 0 is at relative index 3 -> extra[2*15 .. 2*15+5]
        player0_resources_in_extra = extra[2 * 15: 2 * 15 + 5]
        # wood (index 0) should be 3/19
        assert player0_resources_in_extra[0] == pytest.approx(3 / 19.0, abs=1e-5)

    def test_setup_flag_set(self):
        state = make_state()
        assert state.phase in (Phase.SETUP_SETTLEMENT_1, Phase.SETUP_ROAD_1,
                                Phase.SETUP_SETTLEMENT_2, Phase.SETUP_ROAD_2)
        obs = make_observation(state, observer=0)
        # is_setup flag is the last element of the turn-context segment,
        # which is now followed by the trailing 28-dim trade block.
        assert obs[-(1 + _SEG_TRADE)] == pytest.approx(1.0), "is_setup should be 1.0 during setup"

    def test_setup_flag_cleared(self):
        state = make_state()
        rng = random.Random(1)
        fast_forward_setup(state, rng)
        assert state.phase == Phase.ROLL
        obs = make_observation(state, observer=0)
        assert obs[-(1 + _SEG_TRADE)] == pytest.approx(0.0), "is_setup should be 0.0 after setup"
