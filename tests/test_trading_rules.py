"""Tests for the P2P trade sub-phase state machine (propose/respond/resolve)."""

import random

from catan_rl.env.action_mask import legal_action_mask
from catan_rl.env.actions import (
    Resource, propose_trade_action, ACCEPT_TRADE, DECLINE_TRADE, END_TURN,
)
from catan_rl.env.board import BoardConfig
from catan_rl.env.game_state import GameState, Phase
from catan_rl.env.rules import apply_action
from catan_rl.env.validators import legal_actions
from catan_rl.bots.random_bot import pick_action


def _trading_state(seed=0):
    config = BoardConfig.standard(seed=seed)
    state = GameState.new_game(config, n_players=4, seed=seed, profile="standard_trading")
    state.phase = Phase.MAIN
    state.current_player = 0
    state.rolled_this_turn = True
    return state


def test_propose_requires_holdings():
    state = _trading_state()
    state.players[0].resources = [1, 0, 0, 0, 0]
    legal = {a.catalog_index for a in legal_actions(state)}
    assert propose_trade_action(Resource.WOOD, Resource.BRICK, 1).catalog_index in legal
    assert propose_trade_action(Resource.WOOD, Resource.BRICK, 2).catalog_index not in legal  # only 1 wood
    assert propose_trade_action(Resource.ORE, Resource.WOOD, 1).catalog_index not in legal    # no ore


def test_no_propose_when_trades_disabled():
    config = BoardConfig.standard(seed=0)
    state = GameState.new_game(config, n_players=4, seed=0, profile="standard")
    state.phase = Phase.MAIN
    state.current_player = 0
    state.rolled_this_turn = True
    state.players[0].resources = [5, 5, 5, 5, 5]
    legal = {a.catalog_index for a in legal_actions(state)}
    assert not any(idx >= 256 for idx in legal)
    assert propose_trade_action(Resource.WOOD, Resource.BRICK, 1).catalog_index not in legal


def test_full_response_walk_first_accepter_wins():
    state = _trading_state()
    state.players[0].resources = [2, 0, 0, 0, 0]
    for pid in (1, 2, 3):
        state.players[pid].resources = [0, 1, 0, 0, 0]  # all hold the wanted brick
    rng = random.Random(0)
    apply_action(state, propose_trade_action(Resource.WOOD, Resource.BRICK, 2), rng)
    assert state.phase == Phase.TRADE_RESPONSE and state.current_player == 1
    # In TRADE_RESPONSE the legal set is exactly {ACCEPT_TRADE, DECLINE_TRADE}.
    assert sorted(a.catalog_index for a in legal_actions(state)) == [296, 297]
    apply_action(state, DECLINE_TRADE, rng)
    assert state.current_player == 2
    apply_action(state, ACCEPT_TRADE, rng)
    assert state.current_player == 3            # 3 still gets to respond
    apply_action(state, ACCEPT_TRADE, rng)
    # first accepter in order (2) executes
    assert state.phase == Phase.MAIN and state.current_player == 0
    assert state.players[0].resources == [0, 1, 0, 0, 0]
    assert state.players[2].resources == [2, 0, 0, 0, 0]
    assert state.players[3].resources == [0, 1, 0, 0, 0]  # untouched


def test_responder_order_wraps_past_seat_zero():
    # Proposer 2: response walk must wrap 3 -> 0 -> 1, and among accepters
    # seat 0 (earlier in wrap order) beats seat 1.
    state = _trading_state()
    state.current_player = 2
    state.players[2].resources = [2, 0, 0, 0, 0]
    for pid in (0, 1, 3):
        state.players[pid].resources = [0, 1, 0, 0, 0]
    rng = random.Random(0)
    apply_action(state, propose_trade_action(Resource.WOOD, Resource.BRICK, 2), rng)
    assert state.phase == Phase.TRADE_RESPONSE and state.current_player == 3
    apply_action(state, DECLINE_TRADE, rng)
    assert state.current_player == 0
    apply_action(state, ACCEPT_TRADE, rng)
    assert state.current_player == 1
    apply_action(state, ACCEPT_TRADE, rng)
    # First accepter in wrap order (0) executes; 1 untouched.
    assert state.phase == Phase.MAIN and state.current_player == 2
    assert state.players[2].resources == [0, 1, 0, 0, 0]
    assert state.players[0].resources == [2, 0, 0, 0, 0]
    assert state.players[1].resources == [0, 1, 0, 0, 0]


def test_auto_decline_skips_broke_responders():
    # only player 3 holds the get resource -> phase jumps straight to player 3
    state = _trading_state()
    state.players[0].resources = [2, 0, 0, 0, 0]
    state.players[1].resources = [0, 0, 0, 0, 0]
    state.players[2].resources = [0, 0, 0, 0, 0]
    state.players[3].resources = [0, 1, 0, 0, 0]
    rng = random.Random(0)
    apply_action(state, propose_trade_action(Resource.WOOD, Resource.BRICK, 2), rng)
    assert state.phase == Phase.TRADE_RESPONSE
    assert state.current_player == 3
    assert state.pending_trade["responses"][1] is False
    assert state.pending_trade["responses"][2] is False
    assert state.pending_trade["responses"][3] is None


def test_all_auto_declined_returns_to_main_no_transfer():
    state = _trading_state()
    state.players[0].resources = [2, 0, 0, 0, 0]
    state.players[1].resources = [0, 0, 0, 0, 0]
    state.players[2].resources = [0, 0, 0, 0, 0]
    state.players[3].resources = [0, 0, 0, 0, 0]
    rng = random.Random(0)
    apply_action(state, propose_trade_action(Resource.WOOD, Resource.BRICK, 2), rng)
    assert state.phase == Phase.MAIN
    assert state.current_player == 0
    assert state.pending_trade is None
    assert state.players[0].resources == [2, 0, 0, 0, 0]  # untouched, no transfer


def test_trade_cap_per_turn():
    # after max_trades_per_turn proposals (each fully declined), PROPOSE slots leave the mask
    state = _trading_state()
    state.players[0].resources = [10, 0, 0, 0, 0]
    for pid in (1, 2, 3):
        state.players[pid].resources = [0, 0, 0, 0, 0]  # nobody can accept -> auto full decline
    rng = random.Random(0)
    max_trades = state.profile.max_trades_per_turn
    for i in range(max_trades):
        legal = {a.catalog_index for a in legal_actions(state)}
        assert propose_trade_action(Resource.WOOD, Resource.BRICK, 1).catalog_index in legal
        apply_action(state, propose_trade_action(Resource.WOOD, Resource.BRICK, 1), rng)
        assert state.phase == Phase.MAIN  # fully auto-declined, resolves immediately
    assert state.trades_proposed_this_turn == max_trades
    legal = {a.catalog_index for a in legal_actions(state)}
    assert not any(idx >= 256 for idx in legal)


def test_end_turn_resets_cap_counter():
    state = _trading_state()
    state.players[0].resources = [10, 0, 0, 0, 0]
    for pid in (1, 2, 3):
        state.players[pid].resources = [0, 0, 0, 0, 0]
    rng = random.Random(0)
    max_trades = state.profile.max_trades_per_turn
    for i in range(max_trades):
        apply_action(state, propose_trade_action(Resource.WOOD, Resource.BRICK, 1), rng)
    assert state.trades_proposed_this_turn == max_trades
    apply_action(state, END_TURN, rng)
    assert state.trades_proposed_this_turn == 0


def test_resource_conservation_across_trades():
    # sum over players+bank of each resource is invariant through propose/respond/execute
    state = _trading_state()
    state.players[0].resources = [2, 0, 0, 0, 0]
    for pid in (1, 2, 3):
        state.players[pid].resources = [0, 1, 0, 0, 0]

    def totals(state):
        totals = list(state.bank)
        for p in state.players:
            for i in range(5):
                totals[i] += p.resources[i]
        return totals

    before = totals(state)
    rng = random.Random(0)
    apply_action(state, propose_trade_action(Resource.WOOD, Resource.BRICK, 2), rng)
    assert totals(state) == before
    apply_action(state, DECLINE_TRADE, rng)
    assert totals(state) == before
    apply_action(state, ACCEPT_TRADE, rng)
    assert totals(state) == before
    apply_action(state, ACCEPT_TRADE, rng)
    assert totals(state) == before


def test_clone_and_serialization_round_trip_mid_trade():
    # clone() deep-copies pending_trade; to_dict/from_dict round-trips it exactly;
    # from_dict of a dict WITHOUT the new keys yields pending_trade None (backcompat)
    state = _trading_state()
    state.players[0].resources = [2, 0, 0, 0, 0]
    for pid in (1, 2, 3):
        state.players[pid].resources = [0, 1, 0, 0, 0]
    rng = random.Random(0)
    apply_action(state, propose_trade_action(Resource.WOOD, Resource.BRICK, 2), rng)
    assert state.pending_trade is not None

    cloned = state.clone()
    assert cloned.pending_trade == state.pending_trade
    assert cloned.pending_trade is not state.pending_trade
    assert cloned.pending_trade["responses"] is not state.pending_trade["responses"]
    cloned.pending_trade["responses"][2] = True
    assert state.pending_trade["responses"][2] is None  # original untouched

    d = state.to_dict()
    assert set(d["pending_trade"]["responses"].keys()) == {"1", "2", "3"}
    restored = GameState.from_dict(d, state.config)
    assert restored.pending_trade == state.pending_trade
    assert all(isinstance(k, int) for k in restored.pending_trade["responses"].keys())
    assert restored.trades_proposed_this_turn == state.trades_proposed_this_turn

    # Backcompat: a dict missing the new keys entirely.
    d_old = dict(d)
    del d_old["pending_trade"]
    del d_old["trades_proposed_this_turn"]
    restored_old = GameState.from_dict(d_old, state.config)
    assert restored_old.pending_trade is None
    assert restored_old.trades_proposed_this_turn == 0


def test_notrade_profile_bit_identical():
    # run 200 random-legal plies with seed on "standard" profile: legal_action_mask
    # never sets any index >= 256, and Phase.TRADE_RESPONSE never occurs
    config = BoardConfig.standard(seed=11)
    state = GameState.new_game(config, n_players=4, seed=11, profile="standard")
    rng = random.Random(11)
    for _ in range(200):
        if state.is_terminal:
            break
        mask = legal_action_mask(state)
        assert not mask[256:].any()
        assert state.phase != Phase.TRADE_RESPONSE
        apply_action(state, pick_action(state, rng), rng)
