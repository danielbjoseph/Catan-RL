"""
Tests for BeliefTracker: the table-view belief model that reconstructs what a
perfect-memory human observer at the table could deduce about each player's
hand from public events only.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from catan_rl.env.belief import BeliefTracker
from catan_rl.env.board import BoardConfig
from catan_rl.env.game_state import GameState
from catan_rl.env.actions import (
    Action, ActionType, Resource, DevCard,
    ROLL_DICE, road_action, steal_action, discard_action, monopoly_action,
)
from catan_rl.env.rules import apply_action
from catan_rl.env.rules_profile import STANDARD
from catan_rl.bots.random_bot import pick_action

SEED = 0


def _fresh_state(n_players: int = 4, seed: int = SEED) -> GameState:
    config = BoardConfig.standard(seed=seed)
    return GameState.new_game(config, n_players=n_players, seed=seed, profile=STANDARD)


# ---------------------------------------------------------------------------
# 1. Production / build events reconcile exactly
# ---------------------------------------------------------------------------

def test_production_reconciles_exactly():
    state = _fresh_state()
    tracker = BeliefTracker(state)

    before = state.clone()
    after = state.clone()
    after.players[0].resources = [3, 0, 0, 0, 0]

    tracker.on_action(before, ROLL_DICE, after)

    assert np.allclose(tracker.expected(0), [3, 0, 0, 0, 0])
    assert tracker.uncertainty(0) == 0.0


def test_build_reconciles_exactly():
    state = _fresh_state()
    tracker = BeliefTracker(state)

    # Seed player 0 with known resources via a "production" event first.
    before0 = state.clone()
    after0 = state.clone()
    after0.players[0].resources = [3, 3, 0, 0, 0]
    tracker.on_action(before0, ROLL_DICE, after0)

    # Now build a road: costs 1 wood + 1 brick.
    before1 = after0.clone()
    after1 = after0.clone()
    after1.players[0].resources = [2, 2, 0, 0, 0]
    after1.players[0].road_vertices = {0}
    after1.players[0].roads_built = 1

    tracker.on_action(before1, road_action(0), after1)

    assert np.allclose(tracker.expected(0), [2, 2, 0, 0, 0])
    assert tracker.uncertainty(0) == 0.0


# ---------------------------------------------------------------------------
# 2. Steal moves proportional mass and raises uncertainty on both parties
# ---------------------------------------------------------------------------

def test_steal_moves_proportional_mass_and_raises_uncertainty():
    state = _fresh_state()
    tracker = BeliefTracker(state)

    before = state.clone()
    before.current_player = 0
    before.players[0].resources = [0, 0, 0, 0, 0]
    before.players[1].resources = [2, 2, 0, 0, 0]  # total 4

    after = before.clone()
    after.players[0].resources = [1, 0, 0, 0, 0]   # thief total 1
    after.players[1].resources = [1, 2, 0, 0, 0]   # victim total 3

    # Prime tracker's belief for these players to match "before" exactly.
    tracker.on_action(state.clone(), ROLL_DICE, before)

    tracker.on_action(before, steal_action(1), after)

    # Expected composition of one random card from victim's hand: [0.5, 0.5, 0, 0, 0]
    assert np.allclose(tracker.expected(1), [1.5, 1.5, 0, 0, 0])
    assert np.allclose(tracker.expected(0), [0.5, 0.5, 0, 0, 0])
    assert tracker.uncertainty(0) > 0.0
    assert tracker.uncertainty(1) > 0.0


# ---------------------------------------------------------------------------
# 3. Monopoly reconciles the stolen type exactly for all players, even when
#    prior hidden events (steal) had introduced error in that resource type.
# ---------------------------------------------------------------------------

def test_monopoly_reconciles_exactly_despite_prior_uncertainty():
    state = _fresh_state()
    tracker = BeliefTracker(state)

    before = state.clone()
    before.current_player = 0
    before.players[0].resources = [0, 0, 0, 0, 0]
    before.players[1].resources = [2, 2, 0, 0, 0]
    tracker.on_action(state.clone(), ROLL_DICE, before)

    after_steal = before.clone()
    after_steal.players[0].resources = [1, 0, 0, 0, 0]  # true wood = 1
    after_steal.players[1].resources = [1, 2, 0, 0, 0]  # true wood = 1
    tracker.on_action(before, steal_action(1), after_steal)

    # Sanity: tracker now believes wood is split 0.5 / 1.5, not the true 1 / 1.
    assert not np.isclose(tracker.expected(0)[0], 1.0)
    assert not np.isclose(tracker.expected(1)[0], 1.0)

    # Player 2 plays monopoly on WOOD.
    before_mono = after_steal.clone()
    before_mono.current_player = 2
    before_mono.players[2].resources = [0, 0, 0, 0, 0]
    before_mono.players[3].resources = [0, 0, 0, 0, 0]

    after_mono = before_mono.clone()
    # true wood: p0=1, p1=1, p3=0 -> all taken by p2
    after_mono.players[0].resources = [0, 0, 0, 0, 0]
    after_mono.players[1].resources = [0, 2, 0, 0, 0]
    after_mono.players[2].resources = [2, 0, 0, 0, 0]
    after_mono.players[3].resources = [0, 0, 0, 0, 0]

    tracker.on_action(before_mono, monopoly_action(Resource.WOOD), after_mono)

    assert tracker.expected(0)[0] == pytest.approx(0.0, abs=1e-6)
    assert tracker.expected(1)[0] == pytest.approx(0.0, abs=1e-6)
    assert tracker.expected(2)[0] == pytest.approx(2.0, abs=1e-6)
    assert tracker.expected(3)[0] == pytest.approx(0.0, abs=1e-6)


def test_monopoly_pins_reconciled_column_during_renormalization():
    """Regression: renormalization after a monopoly must not move the
    just-reconciled column off ground truth, nor corrupt it via a blanket
    whole-vector rescale.

    Trace: p0 steals from p1 (introduces hidden-event error in both wood
    beliefs), p1 then publicly gains 2 ore, then p0 plays monopoly on WOOD.
    The monopolist's believed vector totals 2.5 after the exact-wood override
    while their true hand is 2, so a blanket rescale would drag wood from the
    exact value 2 down to 1.6. The wood column must stay pinned; only the
    non-reconciled columns may absorb the sum correction.
    """
    state = _fresh_state()
    tracker = BeliefTracker(state)

    # Prime: p1 publicly gains [2, 2, 0, 0, 0].
    before = state.clone()
    before.players[1].resources = [2, 2, 0, 0, 0]
    tracker.on_action(state.clone(), ROLL_DICE, before)

    # p0 steals one hidden card from p1 (truth: it was wood).
    before.current_player = 0
    after_steal = before.clone()
    after_steal.players[0].resources = [1, 0, 0, 0, 0]
    after_steal.players[1].resources = [1, 2, 0, 0, 0]
    tracker.on_action(before, steal_action(1), after_steal)
    # beliefs now: p0 [0.5, 0.5, 0, 0, 0], p1 [1.5, 1.5, 0, 0, 0]

    # p1 publicly gains 2 ore (production) -> p1 ore belief is exact (2.0).
    after_prod = after_steal.clone()
    after_prod.players[1].resources = [1, 2, 0, 0, 2]
    tracker.on_action(after_steal, ROLL_DICE, after_prod)
    assert tracker.expected(1)[4] == pytest.approx(2.0, abs=1e-6)

    # p0 plays monopoly on WOOD: takes p1's 1 wood -> p0 true [2,0,0,0,0].
    before_mono = after_prod.clone()
    before_mono.current_player = 0
    after_mono = before_mono.clone()
    after_mono.players[0].resources = [2, 0, 0, 0, 0]
    after_mono.players[1].resources = [0, 2, 0, 0, 2]
    tracker.on_action(before_mono, monopoly_action(Resource.WOOD), after_mono)

    # (a) The reconciled column is exactly ground truth for every player.
    for pid, true_wood in [(0, 2.0), (1, 0.0), (2, 0.0), (3, 0.0)]:
        assert tracker.expected(pid)[0] == pytest.approx(true_wood, abs=1e-6), (
            f"player {pid} wood belief not pinned to ground truth"
        )

    # (b) Sum invariant holds for every player.
    for pid in range(4):
        hand = after_mono.players[pid].total_resources
        assert float(tracker.expected(pid).sum()) == pytest.approx(hand, abs=1e-4)

    # (c) Monopolist's full vector is derivable: hand 2, wood pinned at 2,
    # so every other column must be exactly 0.
    assert np.allclose(tracker.expected(0), [2, 0, 0, 0, 0], atol=1e-6)

    # p1's non-reconciled columns absorb the correction but keep their
    # relative proportions (brick:ore was 1.5:2 before renormalization).
    # NOTE: no assertion that ore == 2 — residual error in non-reconciled
    # columns is acceptable table-view behavior.
    p1 = tracker.expected(1)
    assert p1[1] / p1[4] == pytest.approx(1.5 / 2.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. Discard subtracts proportionally and raises uncertainty
# ---------------------------------------------------------------------------

def test_discard_subtracts_proportionally():
    state = _fresh_state()
    tracker = BeliefTracker(state)

    before = state.clone()
    before.current_player = 0
    before.players[0].resources = [2, 2, 0, 0, 0]  # total 4
    tracker.on_action(state.clone(), ROLL_DICE, before)

    after = before.clone()
    after.players[0].resources = [1, 2, 0, 0, 0]  # total 3 (discarded 1, type hidden)

    tracker.on_action(before, discard_action(Resource.WOOD), after)

    assert np.allclose(tracker.expected(0), [1.5, 1.5, 0, 0, 0])
    assert tracker.uncertainty(0) > 0.0


# ---------------------------------------------------------------------------
# 5. Property test: play a full seeded random game, feed every ply.
# ---------------------------------------------------------------------------

def test_property_full_random_game_invariants():
    rng = random.Random(SEED)
    config = BoardConfig.standard(seed=SEED)
    state = GameState.new_game(config, n_players=4, seed=SEED, profile=STANDARD)
    tracker = BeliefTracker(state)

    max_plies = 3000
    plies = 0
    while not state.is_terminal and plies < max_plies:
        action = pick_action(state, rng)
        before = state.clone()
        apply_action(state, action, rng)
        after = state
        tracker.on_action(before, action, after)

        for i in range(state.n_players):
            hand = state.players[i].total_resources
            expected_i = tracker.expected(i)
            assert abs(float(expected_i.sum()) - hand) < 1e-4, (
                f"ply {plies}: player {i} expected sum {expected_i.sum()} != hand {hand}"
            )
            if tracker.uncertainty(i) == 0.0:
                assert np.allclose(expected_i, state.players[i].resources, atol=1e-4), (
                    f"ply {plies}: player {i} has zero uncertainty but expected "
                    f"{expected_i} != true {state.players[i].resources}"
                )
        plies += 1

    assert plies > 50, "game ended too quickly to be a meaningful property test"


# ---------------------------------------------------------------------------
# 6. dev_deck_estimate: full-knowledge case
# ---------------------------------------------------------------------------

def test_dev_deck_estimate_full_knowledge():
    state = _fresh_state()
    tracker = BeliefTracker(state)

    # Observer (player 0) holds all cards that have been drawn; no opponent
    # holds any unplayed dev cards. Fabricate a specific remaining deck.
    remaining = [DevCard.KNIGHT, DevCard.KNIGHT, DevCard.VICTORY_POINT]
    state.dev_deck = list(remaining)

    # Cards drawn so far: 14+2+2+2+5 - 3 = 22 cards drawn total, all of which
    # are accounted for by the observer (held + newly bought + played) since
    # no opponent holds any unplayed dev cards. Per-resource drawn counts are
    # [12, 2, 2, 2, 4] (initial minus the fabricated remaining deck above).
    state.players[0].dev_cards = [3, 1, 1, 1, 1]          # held, playable
    state.players[0].dev_cards_new = [1, 0, 0, 0, 0]      # bought this turn
    state.players[0].played_dev_cards = [8, 1, 1, 1, 3]   # played
    for pid in (1, 2, 3):
        state.players[pid].dev_cards = [0, 0, 0, 0, 0]
        state.players[pid].dev_cards_new = [0, 0, 0, 0, 0]
        state.players[pid].played_dev_cards = [0, 0, 0, 0, 0]

    composition, count = tracker.dev_deck_estimate(0, state)

    assert count == 3
    expected = np.array([2, 0, 0, 0, 1], dtype=np.float32)
    assert np.allclose(composition, expected, atol=1e-4)
    assert composition.shape == (5,)
