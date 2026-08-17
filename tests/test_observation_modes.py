"""
Tests for realistic + global observation modes.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from catan_rl.bots.random_bot import pick_action
from catan_rl.env.actions import Resource, propose_trade_action, ACCEPT_TRADE, DECLINE_TRADE
from catan_rl.env.belief import BeliefTracker
from catan_rl.env.board import BoardConfig
from catan_rl.env.game_state import GameState, Phase
from catan_rl.env.observation import (
    OBS_DIM,
    OBS_DIM_GLOBAL,
    OBS_DIM_PERFECT,
    OBS_DIM_REALISTIC,
    apply_belief_noise,
    make_observation,
    obs_dim_for_mode,
)
from catan_rl.env.pettingzoo_env import CatanAECEnv
from catan_rl.env.rules import apply_action
from catan_rl.env.rules_profile import STANDARD

_SEG_TRADE = 28

SEED = 0

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden_observations.npz"
GOLDEN_SEED = 123
GOLDEN_PLIES = 60
GOLDEN_OBSERVER = 1


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


def _golden_state():
    """Deterministic mid-game state recipe shared by the golden-fixture
    generator and the regression test. Do not change: the stored reference
    vectors in fixtures/golden_observations.npz were produced from exactly
    this recipe."""
    rng = random.Random(GOLDEN_SEED)
    config = BoardConfig.standard(seed=GOLDEN_SEED)
    state = GameState.new_game(config, n_players=4, seed=GOLDEN_SEED, profile=STANDARD)
    for _ in range(GOLDEN_PLIES):
        if state.is_terminal:
            break
        action = pick_action(state, rng)
        apply_action(state, action, rng)
    return state


def _generate_golden_fixture():
    """Regenerate the golden fixture. Run manually ONLY when the base
    observation encoding is intentionally changed:

        .venv/Scripts/python.exe -c "from tests.test_observation_modes import _generate_golden_fixture; _generate_golden_fixture()"
    """
    state = _golden_state()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        GOLDEN_PATH,
        self_play=make_observation(state, observer=GOLDEN_OBSERVER, mode="self_play"),
        perfect=make_observation(state, observer=GOLDEN_OBSERVER, mode="perfect"),
    )


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
        assert OBS_DIM == 1548
        assert OBS_DIM_PERFECT == 1593
        assert OBS_DIM_REALISTIC == 1577
        assert OBS_DIM_GLOBAL == 1604


# ---------------------------------------------------------------------------
# 1.5 Pending-trade observation block (appended at the absolute end of every
#     mode's vector; identical computation regardless of mode).
# ---------------------------------------------------------------------------

def _trading_state(seed=0):
    config = BoardConfig.standard(seed=seed)
    state = GameState.new_game(config, n_players=4, seed=seed, profile="standard_trading")
    state.phase = Phase.MAIN
    state.current_player = 0
    state.rolled_this_turn = True
    return state


class TestTradeBlock:
    @pytest.mark.parametrize("mode", ["self_play", "perfect", "realistic", "global"])
    def test_trade_block_zero_when_no_pending_trade(self, mode):
        state, tracker = _play_random(seed=9, n_plies=5)
        state.pending_trade = None
        belief = tracker if mode == "realistic" else None
        obs = make_observation(state, observer=0, mode=mode, belief=belief)
        trade = obs[-_SEG_TRADE:]
        expected = np.zeros(_SEG_TRADE, dtype=np.float32)
        expected[16] = 1.0
        expected[19] = 1.0
        expected[22] = 1.0
        expected[25] = 1.0
        np.testing.assert_array_equal(trade, expected)

    def test_trade_block_encodes_pending_trade(self):
        state = _trading_state()
        state.players[0].resources = [2, 0, 0, 0, 0]
        for pid in (1, 2, 3):
            state.players[pid].resources = [0, 1, 0, 0, 0]  # all hold the wanted brick
        rng = random.Random(0)
        apply_action(state, propose_trade_action(Resource.WOOD, Resource.BRICK, 2), rng)
        assert state.phase == Phase.TRADE_RESPONSE and state.current_player == 1
        apply_action(state, DECLINE_TRADE, rng)
        assert state.current_player == 2
        assert state.pending_trade["responses"] == {1: False, 2: None, 3: None}

        observer = 2
        obs = make_observation(state, observer=observer, mode="self_play")
        trade = obs[-_SEG_TRADE:]

        assert trade[0] == 1.0  # active
        # proposer (0) relative to observer (2): rel = (0-2)%4 = 2
        assert trade[1 + 2] == 1.0
        assert trade[1:5].sum() == 1.0
        # give=WOOD(0), get=BRICK(1)
        assert trade[5 + 0] == 1.0
        assert trade[5:10].sum() == 1.0
        assert trade[10 + 1] == 1.0
        assert trade[10:15].sum() == 1.0
        # give_n = 2 -> 2/2.0 = 1.0
        assert trade[15] == pytest.approx(1.0)

        # response block: rel_i = (pid - observer) % 4
        # rel_i=0 -> pid=2 (observer itself, still pending) -> +0
        assert trade[16 + 0] == 1.0
        # rel_i=1 -> pid=3 (still pending) -> +0
        assert trade[19 + 0] == 1.0
        # rel_i=2 -> pid=0 (the proposer's own slot, no response entry) -> +0
        assert trade[22 + 0] == 1.0
        # rel_i=3 -> pid=1 (declined) -> +1
        assert trade[25 + 1] == 1.0
        # every other response-block entry is zero
        resp_hot = {16, 19, 22, 26}
        for i in range(16, 28):
            if i not in resp_hot:
                assert trade[i] == 0.0

        # accepted state (+2): player 2 accepts; its rel slot (rel_i=0 for
        # observer 2) flips from pending to accepted
        apply_action(state, ACCEPT_TRADE, rng)
        trade = make_observation(state, observer=observer, mode="self_play")[-_SEG_TRADE:]
        assert trade[16 + 2] == 1.0
        assert trade[16 + 0] == 0.0

    def test_trade_block_rotation(self):
        state = _trading_state()
        state.players[0].resources = [2, 0, 0, 0, 0]
        for pid in (1, 2, 3):
            state.players[pid].resources = [0, 1, 0, 0, 0]
        rng = random.Random(0)
        apply_action(state, propose_trade_action(Resource.WOOD, Resource.BRICK, 2), rng)

        # observer = proposer (0): rel = (0-0)%4 = 0
        obs0 = make_observation(state, observer=0, mode="self_play")
        trade0 = obs0[-_SEG_TRADE:]
        assert trade0[1 + 0] == 1.0
        assert trade0[1:5].sum() == 1.0

        # observer = 3: rel = (0-3)%4 = 1
        obs3 = make_observation(state, observer=3, mode="self_play")
        trade3 = obs3[-_SEG_TRADE:]
        assert trade3[1 + 1] == 1.0
        assert trade3[1:5].sum() == 1.0

        assert not np.array_equal(trade0, trade3)


# ---------------------------------------------------------------------------
# 2. Regression: bases must remain byte-identical
# ---------------------------------------------------------------------------

class TestGoldenVectorRegression:
    """Compare freshly-computed base observations against STORED reference
    vectors captured before any mode was added. Catches any drift in the
    shared self_play/perfect construction path, which same-run prefix
    comparisons alone cannot (both sides would drift together)."""

    def test_golden_fixture_exists(self):
        assert GOLDEN_PATH.is_file(), (
            f"missing golden fixture {GOLDEN_PATH}; regenerate ONLY on an "
            "intentional base-encoding change via _generate_golden_fixture()"
        )

    def test_self_play_matches_golden(self):
        golden = np.load(GOLDEN_PATH)
        obs = make_observation(_golden_state(), observer=GOLDEN_OBSERVER, mode="self_play")
        ref = golden["self_play"]
        assert ref.shape == (OBS_DIM,)
        assert np.array_equal(obs, ref), "self_play base drifted from stored golden vector"

    def test_perfect_matches_golden(self):
        golden = np.load(GOLDEN_PATH)
        obs = make_observation(_golden_state(), observer=GOLDEN_OBSERVER, mode="perfect")
        ref = golden["perfect"]
        assert ref.shape == (OBS_DIM_PERFECT,)
        assert np.array_equal(obs, ref), "perfect base drifted from stored golden vector"


class TestRegressionBasesUntouched:
    """The shared BASE segment (board + public + self-private + turn context,
    i.e. everything before mode-specific extras and before the trailing
    28-dim trade block) must be untouched between self_play/realistic and
    perfect/global. The trailing trade block is appended last for every mode
    and is computed identically regardless of mode, so it too must match
    exactly between mode pairs for the same state/observer -- only the
    middle mode-specific extras segment may legitimately differ."""

    def test_realistic_prefix_equals_self_play(self):
        state, tracker = _play_random(seed=1, n_plies=40)
        obs_sp = make_observation(state, observer=0, mode="self_play")
        obs_real = make_observation(state, observer=0, mode="realistic", belief=tracker)
        assert obs_sp.shape[0] == OBS_DIM
        base_dim = OBS_DIM - _SEG_TRADE
        assert np.array_equal(obs_real[:base_dim], obs_sp[:base_dim])
        assert np.array_equal(obs_real[-_SEG_TRADE:], obs_sp[-_SEG_TRADE:])

    def test_global_prefix_equals_perfect(self):
        state, tracker = _play_random(seed=1, n_plies=40)
        obs_pf = make_observation(state, observer=0, mode="perfect")
        obs_glob = make_observation(state, observer=0, mode="global")
        assert obs_pf.shape[0] == OBS_DIM_PERFECT
        base_dim = OBS_DIM_PERFECT - _SEG_TRADE
        assert np.array_equal(obs_glob[:base_dim], obs_pf[:base_dim])
        assert np.array_equal(obs_glob[-_SEG_TRADE:], obs_pf[-_SEG_TRADE:])

    def test_other_observers_too(self):
        state, tracker = _play_random(seed=2, n_plies=40)
        base_dim = OBS_DIM - _SEG_TRADE
        base_dim_perfect = OBS_DIM_PERFECT - _SEG_TRADE
        for observer in range(4):
            obs_sp = make_observation(state, observer=observer, mode="self_play")
            obs_real = make_observation(state, observer=observer, mode="realistic", belief=tracker)
            assert np.array_equal(obs_real[:base_dim], obs_sp[:base_dim])
            assert np.array_equal(obs_real[-_SEG_TRADE:], obs_sp[-_SEG_TRADE:])

            obs_pf = make_observation(state, observer=observer, mode="perfect")
            obs_glob = make_observation(state, observer=observer, mode="global")
            assert np.array_equal(obs_glob[:base_dim_perfect], obs_pf[:base_dim_perfect])
            assert np.array_equal(obs_glob[-_SEG_TRADE:], obs_pf[-_SEG_TRADE:])


# ---------------------------------------------------------------------------
# 3. Realistic belief features
# ---------------------------------------------------------------------------

class TestRealisticBeliefFeatures:
    def _opponent_block(self, obs, rel_i):
        base_dim = OBS_DIM - _SEG_TRADE
        off = base_dim + (rel_i - 1) * 6
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

    def test_different_opponents_get_independent_noise(self, monkeypatch):
        """Regression: the noise RNG key used to be (seed, turn, observer)
        with no opponent component, so if two opponents happened to share
        the same underlying expected-hand vector they'd get the *exact
        same* noise draw applied. Force that scenario by making the
        tracker return an identical raw vector for every opponent, and
        confirm the three resulting (noised) blocks are not all identical."""
        state, tracker = _play_random(seed=3, n_plies=30)
        same_vec = np.array([2.0, 1.0, 0.0, 3.0, 1.0], dtype=np.float32)
        monkeypatch.setattr(tracker, "expected", lambda pid: same_vec.copy())
        noise_cfg = {"belief_blend": 0.3, "belief_noise": 0.8, "seed": 11}
        obs = make_observation(
            state, observer=0, mode="realistic", belief=tracker, noise_cfg=noise_cfg,
        )
        blocks = [self._opponent_block(obs, rel_i)[0] for rel_i in range(1, 4)]
        assert not (
            np.array_equal(blocks[0], blocks[1]) and np.array_equal(blocks[1], blocks[2])
        ), "all three opponents got the identical noise draw"

    def test_dev_deck_and_bank_blocks(self):
        state, tracker = _play_random(seed=4, n_plies=30)
        obs = make_observation(state, observer=0, mode="realistic", belief=tracker)
        dev_off = (OBS_DIM - _SEG_TRADE) + 18
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

    def test_different_pid_same_key_differs(self):
        """Two opponents observed on the same turn by the same observer
        (identical key) must still get independent noise draws once `pid`
        is mixed in."""
        vec = np.array([2.0, 1.0, 0.0, 3.0, 0.0], dtype=np.float32)
        out0 = apply_belief_noise(vec, 6, blend=0.3, sigma=0.5, key=(42, 5, 0), pid=1)
        out1 = apply_belief_noise(vec, 6, blend=0.3, sigma=0.5, key=(42, 5, 0), pid=2)
        assert not np.array_equal(out0, out1)

    def test_same_pid_same_key_identical(self):
        vec = np.array([2.0, 1.0, 0.0, 3.0, 0.0], dtype=np.float32)
        out0 = apply_belief_noise(vec, 6, blend=0.3, sigma=0.5, key=(42, 5, 0), pid=3)
        out1 = apply_belief_noise(vec, 6, blend=0.3, sigma=0.5, key=(42, 5, 0), pid=3)
        assert np.array_equal(out0, out1)

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
        base = OBS_DIM_PERFECT - _SEG_TRADE
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

    def test_make_observation_unknown_mode_raises(self):
        state, _ = _play_random(seed=7, n_plies=10)
        with pytest.raises(ValueError):
            make_observation(state, observer=0, mode="bogus")


# ---------------------------------------------------------------------------
# 7. AEC env plumbing for all four observation modes
# ---------------------------------------------------------------------------

def _random_legal_action(env: CatanAECEnv, rng: random.Random, agent: str):
    mask = env.observe(agent)["action_mask"]
    legal = np.where(mask)[0]
    if len(legal) == 0:
        return None
    return int(rng.choice(legal))


class TestAECEnvObservationModes:
    @pytest.mark.parametrize("mode", ["self_play", "perfect", "realistic", "global"])
    def test_random_plies_correct_dims_no_nans(self, mode):
        env = CatanAECEnv(obs_mode=mode, rules_profile=STANDARD)
        env.reset(seed=5)
        rng = random.Random(5)
        expected_dim = obs_dim_for_mode(mode)

        plies = 0
        while plies < 150 and not (
            all(env.terminations.values()) or all(env.truncations.values())
        ):
            agent = env.agent_selection
            obs_dict = env.observe(agent)
            obs = obs_dict["observation"]
            assert obs.shape == (expected_dim,)
            assert np.all(np.isfinite(obs))

            action = _random_legal_action(env, rng, agent)
            if action is None:
                break
            env.step(action)
            plies += 1

        assert plies >= 100, f"expected at least 100 plies, only got {plies}"

    def test_realistic_env_owns_a_belief_tracker(self):
        env = CatanAECEnv(obs_mode="realistic", rules_profile=STANDARD)
        env.reset(seed=5)
        assert env._belief is not None

        rng = random.Random(5)
        for _ in range(30):
            agent = env.agent_selection
            action = _random_legal_action(env, rng, agent)
            if action is None:
                break
            env.step(action)
        # A well-formed BeliefTracker still answers queries after several steps.
        assert env._belief.expected(0).shape == (5,)

    @pytest.mark.parametrize("mode", ["self_play", "perfect", "global"])
    def test_non_realistic_modes_do_not_own_a_tracker(self, mode):
        env = CatanAECEnv(obs_mode=mode, rules_profile=STANDARD)
        env.reset(seed=5)
        assert env._belief is None

    def test_realistic_observations_differ_between_observers_same_ply(self):
        """Rotation sanity: two different observers at the same ply must not
        see byte-identical vectors (self-private + belief blocks differ)."""
        env = CatanAECEnv(obs_mode="realistic", rules_profile=STANDARD)
        env.reset(seed=5)
        rng = random.Random(5)
        plies = 0
        while plies < 40 and not (
            all(env.terminations.values()) or all(env.truncations.values())
        ):
            agent = env.agent_selection
            action = _random_legal_action(env, rng, agent)
            if action is None:
                break
            env.step(action)
            plies += 1

        obs0 = env.observe("player_0")["observation"]
        obs1 = env.observe("player_1")["observation"]
        assert obs0.shape == obs1.shape == (OBS_DIM_REALISTIC,)
        assert not np.array_equal(obs0, obs1)
