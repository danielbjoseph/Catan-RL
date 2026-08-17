"""Tests for trade personality presets (catan_rl/bots/personalities.py)."""

import random

import pytest

from catan_rl.bots.personalities import (
    PERSONALITIES,
    make_personality_bot,
    resource_pips,
    trade_margin,
)
from catan_rl.bots import greedy_bot, heuristic_bot, resolve_bot
from catan_rl.env.actions import ActionType, Resource
from catan_rl.env.board import BoardConfig
from catan_rl.env.game_state import GameState, Phase
from catan_rl.env.rules import apply_action

MAX_PLIES = 3000


def _response_state(seed=0):
    """MAIN-phase-adjacent TRADE_RESPONSE state: player 0 proposes 2 wood
    for 1 brick, player 1 is the responder on the clock."""
    config = BoardConfig.standard(seed=seed)
    state = GameState.new_game(config, n_players=4, seed=seed, profile="standard_trading")
    state.phase = Phase.TRADE_RESPONSE
    state.current_player = 1
    state.pending_trade = {
        "proposer": 0,
        "give": int(Resource.WOOD),
        "get": int(Resource.BRICK),
        "give_n": 2,
        "responses": {1: None, 2: None, 3: None},
    }
    state.players[0].resources = [2, 0, 0, 0, 0]
    state.players[1].resources = [0, 1, 0, 0, 0]
    return state


def _totals(state):
    totals = list(state.bank)
    for p in state.players:
        for i in range(5):
            totals[i] += p.resources[i]
    return totals


def test_never_trader_never_proposes_or_accepts():
    bot = make_personality_bot(PERSONALITIES["never_trader"])
    for seed in range(5):
        rng = random.Random(seed)
        config = BoardConfig.standard(seed=seed)
        state = GameState.new_game(config, n_players=4, seed=seed, profile="standard_trading")
        plies = 0
        while not state.is_terminal and plies < MAX_PLIES:
            action = bot(state, rng)
            assert action.action_type not in (ActionType.PROPOSE_TRADE, ActionType.ACCEPT_TRADE), (
                f"never_trader took {action} at ply {plies} (seed {seed})"
            )
            apply_action(state, action, rng)
            plies += 1


def test_never_trader_declines_even_wildly_favorable_offer():
    # Direct TRADE_RESPONSE coverage: full never_trader games never reach
    # the response phase (nobody proposes), so the game-walk test above
    # can't exercise accept_margin=inf. Pin it here: 2 wood for 1 brick at
    # uniform values is a clear +1 margin, and never_trader still declines.
    bot = make_personality_bot(PERSONALITIES["never_trader"])
    for seed in range(3):
        state = _response_state()
        assert bot(state, random.Random(seed)).action_type == ActionType.DECLINE_TRADE


def test_stall_the_leader_declines_sole_leader_below_block_window():
    # Proposer is the unique VP leader at 5 VP -- well below the near-win
    # line (win_vp - leader_block_vp = 8) -- with an attractive offer.
    # The sole-leader clause alone must produce the decline.
    bot = make_personality_bot(PERSONALITIES["stall_the_leader"])
    state = _response_state()
    state.players[0].settlements_built = 5
    assert bot(state, random.Random(0)).action_type == ActionType.DECLINE_TRADE


def test_stall_the_leader_declines_leader_adjacent_only():
    personality = PERSONALITIES["stall_the_leader"]
    bot = make_personality_bot(personality)
    rng = random.Random(0)

    # Proposer one VP short of winning (9 = win_vp - 1 >= win_vp - leader_block_vp = 8).
    near_win = _response_state()
    near_win.players[0].settlements_built = 1
    near_win.players[0].cities_built = 4
    action = bot(near_win, rng)
    assert action.action_type == ActionType.DECLINE_TRADE

    # Proposer far behind (0 VP), attractive margin -> accepts on the merits.
    far_behind = _response_state()
    action = bot(far_behind, rng)
    assert action.action_type == ActionType.ACCEPT_TRADE


def test_desperado_threshold_moves_with_deficit():
    personality = PERSONALITIES["desperado"]
    bot = make_personality_bot(personality)
    rng = random.Random(0)

    def _marginal_state(leader_vp):
        state = _response_state()
        # Give the responder resources so gaining/losing 1 unit doesn't
        # cross a build-affordability line (no bonus contamination), and
        # place no settlements so all pips are 0 (uniform value 1.0).
        state.players[1].resources = [2, 2, 1, 0, 0]
        state.pending_trade = {
            "proposer": 0,
            "give": int(Resource.SHEEP),
            "get": int(Resource.WOOD),
            "give_n": 1,
            "responses": {1: None, 2: None, 3: None},
        }
        if leader_vp:
            state.players[2].settlements_built = leader_vp
        return state

    # Sanity: this trade's margin is ~0 (equal pip-less values).
    margin = trade_margin(_marginal_state(0), 1, {int(Resource.SHEEP): 1}, {int(Resource.WOOD): 1})
    assert margin == 0.0

    tied = _marginal_state(leader_vp=0)
    assert bot(tied, rng).action_type == ActionType.DECLINE_TRADE

    behind_by_4 = _marginal_state(leader_vp=4)
    assert bot(behind_by_4, rng).action_type == ActionType.ACCEPT_TRADE


def test_fair_dealer_band():
    personality = PERSONALITIES["fair_dealer"]
    bot = make_personality_bot(personality)
    rng = random.Random(0)

    near_even = _response_state()
    near_even.players[1].resources = [2, 2, 1, 0, 0]
    near_even.pending_trade = {
        "proposer": 0,
        "give": int(Resource.SHEEP),
        "get": int(Resource.WOOD),
        "give_n": 1,
        "responses": {1: None, 2: None, 3: None},
    }
    assert bot(near_even, rng).action_type == ActionType.ACCEPT_TRADE

    # Lopsided in the RESPONDER's favor: a floor-based (margin >= 0) rule
    # would accept this; the band rejects it because |margin| > band.
    lopsided = _response_state()
    lopsided.players[1].resources = [2, 2, 1, 0, 0]
    lopsided.pending_trade = {
        "proposer": 0,
        "give": int(Resource.WOOD),
        "get": int(Resource.SHEEP),
        "give_n": 2,
        "responses": {1: None, 2: None, 3: None},
    }
    assert bot(lopsided, rng).action_type == ActionType.DECLINE_TRADE

    # Lopsided AGAINST the responder: a one-sided rule (margin <= band,
    # no abs) would accept this negative-margin offer; the band declines.
    against = _response_state(seed=0)
    against.players[1].settlement_vertices = {5}  # ore pips 5, brick pips 2
    against.players[1].resources = [0, 1, 0, 0, 0]
    against.pending_trade = {
        "proposer": 0,
        "give": int(Resource.ORE),
        "get": int(Resource.BRICK),
        "give_n": 1,
        "responses": {1: None, 2: None, 3: None},
    }
    margin = trade_margin(against, 1, {int(Resource.ORE): 1}, {int(Resource.BRICK): 1})
    assert margin < -personality.accept_band  # ~ -1/6
    assert bot(against, rng).action_type == ActionType.DECLINE_TRADE


def test_opportunist_accepts_negative_margin():
    personality = PERSONALITIES["opportunist"]
    bot = make_personality_bot(personality)
    rng = random.Random(0)

    # Board seed 0, vertex 5: pips = [0, 2(brick), 0, 0, 5(ore)] -- ore is
    # cheap to this player (value 1/6), brick is dear (value 1/3), so
    # gaining ore for losing brick is a negative-margin trade for them.
    state = _response_state(seed=0)
    state.players[1].settlement_vertices = {5}
    pips = resource_pips(state, 1)
    assert pips[int(Resource.ORE)] > pips[int(Resource.BRICK)] > 0

    state.pending_trade = {
        "proposer": 0,
        "give": int(Resource.ORE),
        "get": int(Resource.BRICK),
        "give_n": 1,
        "responses": {1: None, 2: None, 3: None},
    }
    state.players[1].resources = [0, 1, 0, 0, 0]  # holds the brick they'd give up

    margin = trade_margin(state, 1, {int(Resource.ORE): 1}, {int(Resource.BRICK): 1})
    assert -0.5 <= margin < 0.0

    action = bot(state, rng)
    assert action.action_type == ActionType.ACCEPT_TRADE


def test_mixed_personality_games_terminate_and_conserve():
    presets = ["opportunist", "stall_the_leader", "fair_dealer", "desperado"]
    bots = [make_personality_bot(PERSONALITIES[p]) for p in presets]
    executed_trades = 0

    for seed in range(3):
        rng = random.Random(seed)
        config = BoardConfig.standard(seed=seed)
        state = GameState.new_game(config, n_players=4, seed=seed, profile="standard_trading")
        plies = 0
        while not state.is_terminal and plies < MAX_PLIES:
            before = _totals(state)
            proposer_before = None
            if state.phase == Phase.TRADE_RESPONSE and state.pending_trade is not None:
                proposer_before = (
                    state.pending_trade["proposer"],
                    list(state.players[state.pending_trade["proposer"]].resources),
                )
            action = bots[state.current_player](state, rng)
            apply_action(state, action, rng)
            assert _totals(state) == before, f"conservation violated at ply {plies} (seed {seed})"

            if proposer_before is not None and state.pending_trade is None and state.phase == Phase.MAIN:
                proposer_id, resources_before = proposer_before
                if state.players[proposer_id].resources != resources_before:
                    executed_trades += 1
            plies += 1

        assert state.is_terminal or plies == MAX_PLIES

    assert executed_trades >= 1, "no trade executed across the whole batch"


def test_plain_bots_decline():
    # Multiple seeds: with the DECLINE branch removed, the rng.choice
    # fallback happens to return DECLINE for Random(0), so a single-seed
    # assertion would pass vacuously.
    for bot in (heuristic_bot, greedy_bot):
        for seed in range(5):
            state = _response_state()
            action = bot.pick_action(state, random.Random(seed))
            assert action.action_type == ActionType.DECLINE_TRADE


def test_resolve_bot_names():
    valid_names = ["random", "greedy", "heuristic", *PERSONALITIES]
    for name in valid_names:
        bot = resolve_bot(name)
        assert callable(bot)

    with pytest.raises(ValueError) as excinfo:
        resolve_bot("nope")
    message = str(excinfo.value)
    for name in valid_names:
        assert name in message


def test_resource_pips_city_doubles():
    state = _response_state(seed=0)
    p = state.players[1]
    p.settlement_vertices = {5}
    base = resource_pips(state, 1).copy()
    assert base.sum() > 0
    p.settlement_vertices = set()
    p.city_vertices = {5}
    assert (resource_pips(state, 1) == 2 * base).all()


def test_build_need_bonus_applies_once():
    # Gaining 1 ore at 2 wheat + 2 ore newly affords a city -> +0.5, applied
    # once per resource regardless of quantity gained.
    state = _response_state(seed=0)
    p = state.players[1]
    p.settlement_vertices = {5}  # enables the city-placement precondition
    ore = int(Resource.ORE)
    value_ore = 1.0 / (1.0 + resource_pips(state, 1)[ore])

    p.resources = [0, 0, 0, 2, 2]  # 2 wheat, 2 ore: one ore short of a city
    with_bonus = trade_margin(state, 1, {ore: 1}, {})
    p.resources = [0, 0, 0, 2, 0]  # 2 ore short: gaining 1 doesn't afford it
    without_bonus = trade_margin(state, 1, {ore: 1}, {})
    assert with_bonus == pytest.approx(without_bonus + 0.5)

    p.resources = [0, 0, 0, 2, 2]
    gaining_two = trade_margin(state, 1, {ore: 2}, {})
    assert gaining_two == pytest.approx(with_bonus + value_ore)  # bonus not doubled
