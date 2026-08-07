"""Tests for opponent-pool training (personality and checkpoint seats)."""

import numpy as np
import pytest
import torch

from catan_rl.env.actions import CATALOG_SIZE
from catan_rl.env.observation import OBS_DIM, obs_dim_for_mode
from catan_rl.env.rules_profile import RulesProfile
from catan_rl.rl.checkpointing import save_checkpoint
from catan_rl.rl.models import ActorCritic, act_prefix_sliced
from catan_rl.rl.rollout import collect_rollouts

FAST_TRADING = RulesProfile(
    name="fast_trading", dev_cards_enabled=False, win_vp=6, trades_enabled=True
)


def _tiny_policy(seed=0, obs_mode="self_play"):
    torch.manual_seed(seed)
    return ActorCritic(obs_dim=obs_dim_for_mode(obs_mode), hidden_sizes=(8, 8))


class TestDefaultPureSelfPlay:
    def test_default_none_is_pure_self_play(self):
        policy = _tiny_policy()
        batch = collect_rollouts(
            policy, n_games=2, rules_profile=FAST_TRADING, seed=1, max_turns=60,
        )
        assert set(batch.seat_ids.tolist()) == {0, 1, 2, 3}
        assert "policy_win_rate" not in batch.stats
        assert "opponent_win_rates" not in batch.stats


class TestOpponentPool:
    def test_pool_only_policy_seat_transitions(self):
        policy = _tiny_policy()
        opponents = {
            "pool": [
                {"type": "personality", "name": "never_trader", "weight": 1},
                {"type": "personality", "name": "never_trader", "weight": 1},
                {"type": "personality", "name": "never_trader", "weight": 1},
            ]
        }
        n_games = 4
        batch = collect_rollouts(
            policy, n_games=n_games, rules_profile=FAST_TRADING, seed=3, max_turns=60,
            opponents=opponents, n_policy_seats=1,
        )
        for seat, episode in zip(batch.seat_ids.tolist(), batch.episode_ids.tolist()):
            assert seat == episode % 4  # rotating policy seat = (g + 0) % 4

    def test_pool_sampling_reproducible(self):
        def run():
            policy = _tiny_policy()
            opponents = {
                "pool": [
                    {"type": "personality", "name": "opportunist", "weight": 2},
                    {"type": "personality", "name": "stall_the_leader", "weight": 1},
                    {"type": "self", "weight": 1},
                ]
            }
            return collect_rollouts(
                policy, n_games=4, rules_profile=FAST_TRADING, seed=5, max_turns=60,
                opponents=opponents, n_policy_seats=1,
            )

        b1 = run()
        b2 = run()
        assert b1.stats["opponent_win_rates"] == b2.stats["opponent_win_rates"]
        assert b1.stats["policy_win_rate"] == b2.stats["policy_win_rate"]

    def test_stats_have_opponent_win_rates(self):
        policy = _tiny_policy()
        opponents = {"pool": [{"type": "personality", "name": "opportunist", "weight": 1}]}
        batch = collect_rollouts(
            policy, n_games=3, rules_profile=FAST_TRADING, seed=2, max_turns=60,
            opponents=opponents, n_policy_seats=1,
        )
        assert "policy_win_rate" in batch.stats
        assert 0.0 <= batch.stats["policy_win_rate"] <= 1.0
        assert set(batch.stats["opponent_win_rates"].keys()) <= {"personality:opportunist"}
        for rate in batch.stats["opponent_win_rates"].values():
            assert 0.0 <= rate <= 1.0

    def test_n_policy_seats_rotation(self):
        """n_policy_seats=2: policy seats for game g are {g%4, (g+1)%4}."""
        policy = _tiny_policy()
        opponents = {"pool": [{"type": "personality", "name": "never_trader", "weight": 1}]}
        n_games = 4
        batch = collect_rollouts(
            policy, n_games=n_games, rules_profile=FAST_TRADING, seed=9, max_turns=60,
            opponents=opponents, n_policy_seats=2,
        )
        for seat, episode in zip(batch.seat_ids.tolist(), batch.episode_ids.tolist()):
            assert seat in {episode % 4, (episode + 1) % 4}

    def test_checkpoint_opponent_non_realistic_mode_runs(self, tmp_path):
        """A self_play-mode checkpoint opponent needs no belief tracker and
        can be used regardless of the env's own obs_mode."""
        ckpt_policy = _tiny_policy(seed=7, obs_mode="self_play")
        opt = torch.optim.Adam(ckpt_policy.parameters())
        ckpt_path = save_checkpoint(
            tmp_path, ckpt_policy, opt, 0, {}, {}, obs_mode="self_play"
        )

        policy = _tiny_policy(seed=1)
        opponents = {"pool": [{"type": "checkpoint", "path": str(ckpt_path), "weight": 1}]}
        batch = collect_rollouts(
            policy, n_games=1, rules_profile=FAST_TRADING, seed=4, max_turns=60,
            opponents=opponents, n_policy_seats=1,
        )
        assert len(batch) > 0
        assert "checkpoint:" + ckpt_path.stem in batch.stats["opponent_win_rates"]


class TestRealisticCheckpointGuard:
    def test_realistic_checkpoint_mode_mismatch_raises(self, tmp_path):
        ckpt_policy = _tiny_policy(seed=1, obs_mode="realistic")
        opt = torch.optim.Adam(ckpt_policy.parameters())
        ckpt_path = save_checkpoint(
            tmp_path, ckpt_policy, opt, 0, {}, {}, obs_mode="realistic"
        )

        policy = _tiny_policy(seed=2, obs_mode="self_play")
        opponents = {"pool": [{"type": "checkpoint", "path": str(ckpt_path), "weight": 1}]}
        with pytest.raises(ValueError):
            collect_rollouts(
                policy, n_games=1, rules_profile=FAST_TRADING, seed=1, max_turns=10,
                obs_mode="self_play", opponents=opponents, n_policy_seats=1,
            )


class TestActPrefixSliced:
    def test_act_prefix_sliced_old_head_declines_trades(self):
        torch.manual_seed(0)
        old_policy = ActorCritic(obs_dim=1520, n_actions=256, hidden_sizes=(8, 8))
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        mask = np.zeros(CATALOG_SIZE, dtype=bool)
        mask[296] = True  # ACCEPT_TRADE
        mask[297] = True  # DECLINE_TRADE
        action = act_prefix_sliced(old_policy, obs, mask)
        assert action == 297

    def test_act_prefix_sliced_normal_case_picks_legal_action(self):
        torch.manual_seed(0)
        policy = ActorCritic(obs_dim=OBS_DIM, n_actions=CATALOG_SIZE, hidden_sizes=(8, 8))
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        mask = np.zeros(CATALOG_SIZE, dtype=bool)
        mask[0] = True
        mask[1] = True
        action = act_prefix_sliced(policy, obs, mask, deterministic=True)
        assert action in (0, 1)
