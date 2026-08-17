"""Tests for the rules profile system (standard / simplified_v1 / custom)."""

import random

import numpy as np
import pytest

from catan_rl.env.action_mask import legal_action_mask
from catan_rl.env.actions import CATALOG
from catan_rl.env.board import BoardConfig
from catan_rl.env.game_state import GameState, Phase
from catan_rl.env.pettingzoo_env import CatanAECEnv
from catan_rl.env.rules import apply_action
from catan_rl.env.rules_profile import RulesProfile, STANDARD, SIMPLIFIED_V1
from catan_rl.bots.random_bot import pick_action

# Catalog slots for dev-card actions: BUY_DEV_CARD(230) .. PLAY_VICTORY_POINT(253)
DEV_SLOTS = slice(230, 254)


def _new_state(profile=None, seed=0):
    config = BoardConfig.standard(seed=seed)
    return GameState.new_game(config, n_players=4, seed=seed, profile=profile)


class TestProfileResolution:
    def test_builtin_constants(self):
        assert STANDARD.dev_cards_enabled is True
        assert STANDARD.win_vp == 10
        assert SIMPLIFIED_V1.dev_cards_enabled is False
        assert SIMPLIFIED_V1.win_vp == 10

    def test_get_from_string(self):
        assert RulesProfile.get("simplified_v1") == SIMPLIFIED_V1
        assert RulesProfile.get("standard") == STANDARD

    def test_get_passthrough_and_default(self):
        p = RulesProfile(name="custom", dev_cards_enabled=False, win_vp=8)
        assert RulesProfile.get(p) is p
        assert RulesProfile.get(None) == STANDARD

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError):
            RulesProfile.get("nonexistent_profile")

    def test_load_yaml_configs(self):
        std = RulesProfile.load("standard")
        simp = RulesProfile.load("simplified_v1")
        assert std == STANDARD
        assert simp == SIMPLIFIED_V1


class TestSimplifiedRules:
    def test_default_profile_is_standard(self):
        state = _new_state()
        assert state.profile == STANDARD
        assert len(state.dev_deck) == 25

    def test_simplified_has_empty_dev_deck(self):
        state = _new_state(profile=SIMPLIFIED_V1)
        assert state.dev_deck == []

    def test_simplified_never_offers_dev_actions(self):
        """Play 300 random plies in simplified mode; dev slots must stay masked."""
        rng = random.Random(7)
        state = _new_state(profile=SIMPLIFIED_V1, seed=7)
        for _ in range(300):
            if state.is_terminal:
                break
            mask = legal_action_mask(state)
            assert not mask[DEV_SLOTS].any(), (
                f"dev-card slot legal in simplified mode, phase={state.phase}"
            )
            apply_action(state, pick_action(state, rng), rng)

    def test_simplified_blocks_buy_even_with_resources(self):
        state = _new_state(profile=SIMPLIFIED_V1)
        # Fast-forward past setup by force
        state.phase = Phase.MAIN
        p = state.current
        p.resources = [5, 5, 5, 5, 5]
        mask = legal_action_mask(state)
        assert not mask[230]  # BUY_DEV_CARD


class TestWinVP:
    def test_custom_win_vp_8(self):
        profile = RulesProfile(name="fast", dev_cards_enabled=False, win_vp=8)
        state = _new_state(profile=profile)
        state.phase = Phase.MAIN
        p = state.current
        p.settlements_built = 4
        p.cities_built = 2  # public VP = 4 + 4 = 8
        rng = random.Random(0)
        apply_action(state, CATALOG[1], rng)  # END_TURN triggers win check
        assert state.winner == 0
        assert state.is_terminal

    def test_standard_win_vp_10_not_at_8(self):
        state = _new_state()
        state.phase = Phase.MAIN
        p = state.current
        p.settlements_built = 4
        p.cities_built = 2  # 8 VP
        rng = random.Random(0)
        apply_action(state, CATALOG[1], rng)
        assert state.winner is None


class TestSerialization:
    def test_profile_round_trip(self):
        profile = RulesProfile(name="fast", dev_cards_enabled=False, win_vp=8)
        state = _new_state(profile=profile)
        d = state.to_dict()
        restored = GameState.from_dict(d, state.config)
        assert restored.profile == profile


class TestTrading:
    def test_trading_profiles_builtin(self):
        p = RulesProfile.get("standard_trading")
        assert p.trades_enabled and p.dev_cards_enabled and p.max_trades_per_turn == 3
        q = RulesProfile.get("simplified_trading_v1")
        assert q.trades_enabled and not q.dev_cards_enabled

    def test_existing_profiles_no_trading(self):
        assert not RulesProfile.get("standard").trades_enabled
        assert not RulesProfile.get("simplified_v1").trades_enabled

    def test_from_dict_backcompat_missing_trade_keys(self):
        p = RulesProfile.from_dict({"name": "standard", "dev_cards_enabled": True, "win_vp": 10})
        assert p.trades_enabled is False and p.max_trades_per_turn == 3

    def test_round_trip_with_trading(self):
        p = RulesProfile.get("standard_trading")
        assert RulesProfile.from_dict(p.to_dict()) == p


class TestEnvIntegration:
    def test_aec_env_accepts_profile(self):
        env = CatanAECEnv(rules_profile="simplified_v1")
        env.reset(seed=3)
        rng = np.random.default_rng(3)
        for _ in range(200):
            if all(env.terminations.values()) or all(env.truncations.values()):
                break
            obs = env.observe(env.agent_selection)
            mask = obs["action_mask"]
            assert not mask[DEV_SLOTS].any()
            legal = np.where(mask)[0]
            env.step(int(rng.choice(legal)))
