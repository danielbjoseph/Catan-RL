"""
Official-rulebook regression tests for catan_rl.env.rules.

These tests encode specific rules from the official Catan rulebook that are
easy to get subtly wrong in an engine implementation (e.g. "first come, first
served" bank depletion vs. the official "nobody gets it if the bank can't
cover everyone" rule). This file grows across the rules-audit task series.
"""

import random

from catan_rl.env.action_mask import legal_action_mask
from catan_rl.env.actions import (
    ActionType, DevCard, Resource, END_TURN, PLAY_KNIGHT, ROLL_DICE,
    move_robber_action, settlement_action, steal_action,
)
from catan_rl.env.board import BoardConfig, HexType
from catan_rl.env.game_state import GameState, Phase
from catan_rl.env.rules import _produce_resources, apply_action
from catan_rl.env.rules_profile import SIMPLIFIED_V1
from catan_rl.env.scoring import check_winner, compute_longest_road, update_longest_road
from catan_rl.env.validators import _steal_actions, legal_actions
from catan_rl.bots.random_bot import pick_action


def _clean_vertex(geo, config, v, hex_id, token):
    """True if `v` isn't also adjacent to a *different* hex sharing `token`.

    Standard board generation doesn't forbid two same-numbered hexes from
    touching the same vertex, so without this check a test that places a
    settlement on `v` could accidentally draw production from an unrelated
    hex and contaminate the resource counts being asserted on.
    """
    return all(
        hx == hex_id or config.hex_tokens[hx] != token
        for hx in geo.vertex_to_hexes[v]
    )


def _find_clean_production_site(config: BoardConfig, min_vertices: int = 2):
    """
    Find a non-desert hex whose token has at least `min_vertices` vertices
    that are "clean" (see `_clean_vertex`), so a test can place settlements
    there and know exactly what resource/count is owed. Returns
    (hex_id, token, resource, clean_vertices).
    """
    geo = config.geometry
    for hi, hex_type in enumerate(config.hex_resources):
        if hex_type == HexType.DESERT:
            continue
        token = config.hex_tokens[hi]
        if token == 0:
            continue
        clean = [v for v in geo.hex_to_vertices[hi] if _clean_vertex(geo, config, v, hi, token)]
        if len(clean) >= min_vertices:
            return hi, token, hex_type.to_resource(), clean
    raise AssertionError("No hex found with enough clean (non-token-colliding) vertices")


def _make_state(seed: int = 0):
    config = BoardConfig.standard(seed=seed)
    state = GameState.new_game(config, n_players=4, seed=seed)
    state.phase = Phase.ROLL
    state.current_player = 0
    # Setup phase bookkeeping isn't relevant to production tests.
    state._setup_forward_idx = 4
    state._setup_backward_idx = -1
    # Start every player with a clean slate; tests place settlements manually.
    for p in state.players:
        p.settlement_vertices = set()
        p.city_vertices = set()
        p.resources = [0] * 5
    return state, config


class TestBankShortageOnProduction:
    """Official rule: if the bank cannot fully supply *all* players owed a
    resource on a roll, *no* player receives that resource type — unless
    exactly one player is owed it, in which case they take what's left."""

    def test_two_players_owed_more_than_bank_has_both_get_zero(self):
        """(a) Two players owed wood, bank has only 1 -> both get 0, bank unchanged."""
        state, config = _make_state(seed=0)
        hex_id, token, resource, vertices = _find_clean_production_site(config, min_vertices=2)

        v0, v1 = vertices[0], vertices[1]
        state.players[0].settlement_vertices.add(v0)
        state.players[1].settlement_vertices.add(v1)

        state.bank = [19, 19, 19, 19, 19]
        state.bank[int(resource)] = 1  # not enough for two players each owed 1

        bank_before = list(state.bank)
        _produce_resources(state, token)

        assert state.players[0].resources[int(resource)] == 0
        assert state.players[1].resources[int(resource)] == 0
        assert state.bank == bank_before

    def test_single_demander_takes_remaining_supply(self):
        """(b) One player owed 3 (city + settlement), bank has 2 -> gets 2, bank 0."""
        state, config = _make_state(seed=0)
        hex_id, token, resource, vertices = _find_clean_production_site(config, min_vertices=2)

        v0, v1 = vertices[0], vertices[1]
        # Player 0 has a city on v0 (2) and a settlement on v1 (1) => owed 3.
        state.players[0].city_vertices.add(v0)
        state.players[0].settlement_vertices.add(v1)

        state.bank = [19, 19, 19, 19, 19]
        state.bank[int(resource)] = 2  # less than the 3 owed

        _produce_resources(state, token)

        assert state.players[0].resources[int(resource)] == 2
        assert state.bank[int(resource)] == 0

    def test_sufficient_supply_unchanged_behavior(self):
        """(c) Bank has plenty -> every demander gets exactly what they're owed."""
        state, config = _make_state(seed=0)
        hex_id, token, resource, vertices = _find_clean_production_site(config, min_vertices=2)

        v0, v1 = vertices[0], vertices[1]
        state.players[0].settlement_vertices.add(v0)
        state.players[1].city_vertices.add(v1)

        state.bank = [19, 19, 19, 19, 19]
        bank_before = list(state.bank)

        _produce_resources(state, token)

        assert state.players[0].resources[int(resource)] == 1
        assert state.players[1].resources[int(resource)] == 2
        assert state.bank[int(resource)] == bank_before[int(resource)] - 3


class TestVictoryPointDevCardsAutoCount:
    """Official rule: VP dev cards are never "played" — they count toward
    the win condition automatically as soon as they're held, while staying
    excluded from public VP (visible building-based score)."""

    def test_newly_bought_vp_card_detected_as_winner_on_next_win_check(self):
        """A player at 9 public VP who receives a VP card (still in
        dev_cards_new, i.e. bought this turn and not yet "playable") should
        be declared the winner the next time a win check runs, without ever
        playing the card."""
        state, config = _make_state(seed=0)
        state.phase = Phase.MAIN
        p = state.players[0]
        p.settlements_built = 1
        p.settlement_vertices = {0}
        p.cities_built = 4
        p.city_vertices = {1, 2, 3, 4}
        assert p.public_vp == 9

        assert check_winner(state) is None

        p.receive_dev_card(DevCard.VICTORY_POINT)  # lands in dev_cards_new
        assert p.dev_cards_new[int(DevCard.VICTORY_POINT)] == 1

        apply_action(state, END_TURN)

        assert state.winner == 0
        assert state.phase == Phase.GAME_OVER

    def test_play_victory_point_slot_never_legal_over_random_game(self):
        """Catalog slot 253 (PLAY_VICTORY_POINT) must never appear as a
        legal action across a full random-legal standard-profile game."""
        rng = random.Random(0)
        config = BoardConfig.standard(seed=0)
        state = GameState.new_game(config, n_players=4, seed=0)

        for _ in range(200):
            mask = legal_action_mask(state)
            assert not mask[253], "PLAY_VICTORY_POINT (slot 253) must be permanently masked"
            if state.phase == Phase.GAME_OVER:
                break
            action = pick_action(state, rng)
            apply_action(state, action, rng)

    def test_public_vp_excludes_vp_cards(self):
        """public_vp must reflect only settlements/cities; VP dev cards
        (held, newly bought, or historically played) live in hidden_vp."""
        state, config = _make_state(seed=0)
        p = state.players[0]
        p.settlements_built = 2
        p.cities_built = 1
        p.dev_cards[int(DevCard.VICTORY_POINT)] = 3
        p.dev_cards_new[int(DevCard.VICTORY_POINT)] = 2
        p.played_dev_cards[int(DevCard.VICTORY_POINT)] = 1

        assert p.public_vp == 4
        assert p.hidden_vp == 6
        assert p.total_vp == 10


class TestDevCardBeforeRoll:
    """Official rule: one dev card may be played at any time during your
    turn, including before the roll. Knight-before-roll is the strategically
    important case (unblocks your own hexes before production happens)."""

    def _no_target_hex(self, state) -> int:
        """A hex the robber isn't already on (safe to move to; _make_state
        clears all settlements so no hex has adjacent occupied vertices)."""
        return 0 if state.robber_hex != 0 else 1

    def test_knight_legal_in_roll_phase(self):
        """A player holding a knight (standard profile) sees PLAY_KNIGHT
        legal while still in the ROLL phase, before rolling."""
        state, config = _make_state(seed=0)
        p0 = state.players[0]
        p0.dev_cards[int(DevCard.KNIGHT)] = 1

        assert state.phase == Phase.ROLL
        acts = legal_actions(state)
        assert any(a.action_type == ActionType.PLAY_KNIGHT for a in acts)

        mask = legal_action_mask(state)
        assert mask[231], "PLAY_KNIGHT (catalog slot 231) must be legal pre-roll"

    def test_knight_preroll_no_steal_target_returns_to_roll(self):
        """Playing knight pre-roll with no adjacent opponents: ROLL ->
        (play knight) -> ROBBER -> back to ROLL, and ROLL_DICE is still
        required/legal afterward."""
        state, config = _make_state(seed=0)
        p0 = state.players[0]
        p0.dev_cards[int(DevCard.KNIGHT)] = 1

        apply_action(state, PLAY_KNIGHT)
        assert state.phase == Phase.ROBBER
        assert p0.has_played_dev_card
        assert p0.army_size == 1

        apply_action(state, move_robber_action(self._no_target_hex(state)))
        assert state.phase == Phase.ROLL

        acts = legal_actions(state)
        assert any(a.action_type == ActionType.ROLL_DICE for a in acts)
        assert ROLL_DICE in acts

        # Must still be able to roll to finish the turn.
        apply_action(state, ROLL_DICE, random.Random(0))
        assert state.rolled_this_turn is True

    def test_knight_preroll_with_steal_target_returns_to_roll(self):
        """Playing knight pre-roll with an adjacent opponent: ROLL ->
        ROBBER -> STEAL -> back to ROLL (not MAIN)."""
        state, config = _make_state(seed=1)
        geo = config.geometry
        p0 = state.players[0]
        p1 = state.players[1]
        p0.dev_cards[int(DevCard.KNIGHT)] = 1
        p1.resources = [1, 0, 0, 0, 0]

        hex_id = self._no_target_hex(state)
        v = geo.hex_to_vertices[hex_id][0]
        p1.settlement_vertices.add(v)

        apply_action(state, PLAY_KNIGHT)
        assert state.phase == Phase.ROBBER

        apply_action(state, move_robber_action(hex_id))
        assert state.phase == Phase.STEAL

        apply_action(state, steal_action(1), random.Random(0))
        assert state.phase == Phase.ROLL

    def test_knight_postroll_returns_to_main(self):
        """Playing knight after rolling still returns to MAIN, as before."""
        state, config = _make_state(seed=0)
        p0 = state.players[0]
        p0.dev_cards[int(DevCard.KNIGHT)] = 1
        state.rolled_this_turn = True
        state.phase = Phase.MAIN

        apply_action(state, PLAY_KNIGHT)
        assert state.phase == Phase.ROBBER

        apply_action(state, move_robber_action(self._no_target_hex(state)))
        assert state.phase == Phase.MAIN

    def test_card_bought_this_turn_not_playable_preroll(self):
        """A knight bought this turn (still in dev_cards_new) is not
        playable pre-roll, same gating as MAIN."""
        state, config = _make_state(seed=0)
        p0 = state.players[0]
        p0.dev_cards_new[int(DevCard.KNIGHT)] = 1

        acts = legal_actions(state)
        assert not any(a.action_type == ActionType.PLAY_KNIGHT for a in acts)
        assert not legal_action_mask(state)[231]

    def test_preroll_dev_card_play_consumes_one_per_turn_allowance(self):
        """Playing a dev card pre-roll uses up the one-per-turn allowance:
        nothing playable in MAIN afterward that same turn."""
        state, config = _make_state(seed=0)
        p0 = state.players[0]
        p0.dev_cards[int(DevCard.KNIGHT)] = 1
        p0.dev_cards[int(DevCard.MONOPOLY)] = 1

        apply_action(state, PLAY_KNIGHT)
        assert p0.has_played_dev_card
        apply_action(state, move_robber_action(self._no_target_hex(state)))
        assert state.phase == Phase.ROLL

        # Simulate finishing the roll (already exercised end-to-end above);
        # here we just need to land in MAIN with the allowance already spent.
        state.rolled_this_turn = True
        state.phase = Phase.MAIN

        acts = legal_actions(state)
        dev_play_acts = [
            a for a in acts
            if a.action_type in (
                ActionType.PLAY_KNIGHT, ActionType.PLAY_ROAD_BUILDING,
                ActionType.PLAY_YEAR_OF_PLENTY, ActionType.PLAY_MONOPOLY,
            )
        ]
        assert dev_play_acts == []

    def test_simplified_v1_roll_phase_only_roll_dice(self):
        """simplified_v1 has dev cards disabled entirely: ROLL phase offers
        only ROLL_DICE, even if a card is (artificially) present."""
        config = BoardConfig.standard(seed=0)
        state = GameState.new_game(config, n_players=4, seed=0, profile=SIMPLIFIED_V1)
        state.phase = Phase.ROLL
        state.current_player = 0
        state._setup_forward_idx = 4
        state._setup_backward_idx = -1
        state.players[0].dev_cards[int(DevCard.KNIGHT)] = 1

        acts = legal_actions(state)
        assert acts == [ROLL_DICE]


class TestStealTargetMustHoldCards:
    """Official rule: the robber may only steal from a player who is
    actually holding resource cards. A player with a building adjacent to
    the robbed hex but zero cards in hand is not a legal steal target, and
    if that's the *only* adjacent opponent, the STEAL phase is skipped
    entirely (same as if no opponent were adjacent at all)."""

    def test_move_robber_to_hex_with_only_empty_handed_opponent_skips_steal(self):
        """(a) The only adjacent opponent holds 0 resources -> no STEAL
        phase; post-roll, the turn returns straight to MAIN."""
        state, config = _make_state(seed=0)
        geo = config.geometry
        hex_id = 0 if state.robber_hex != 0 else 1
        v = geo.hex_to_vertices[hex_id][0]
        state.players[1].settlement_vertices.add(v)
        state.players[1].resources = [0] * 5
        state.rolled_this_turn = True
        state.phase = Phase.ROBBER

        apply_action(state, move_robber_action(hex_id))

        assert state.phase != Phase.STEAL
        assert state.phase == Phase.MAIN
        assert state.pending_steal_hex is None

    def test_move_robber_no_target_returns_to_roll_when_preroll(self):
        """Same scenario, but pre-roll (e.g. knight played before rolling):
        must return to ROLL, not MAIN, per rolled_this_turn."""
        state, config = _make_state(seed=0)
        geo = config.geometry
        hex_id = 0 if state.robber_hex != 0 else 1
        v = geo.hex_to_vertices[hex_id][0]
        state.players[1].settlement_vertices.add(v)
        state.players[1].resources = [0] * 5
        state.rolled_this_turn = False
        state.phase = Phase.ROBBER

        apply_action(state, move_robber_action(hex_id))

        assert state.phase != Phase.STEAL
        assert state.phase == Phase.ROLL

    def test_steal_actions_excludes_empty_handed_includes_card_holder_same_hex(self):
        """(b) Two opponents adjacent to the same hex: one holds 0 cards,
        the other holds cards. _steal_actions must offer only the holder."""
        state, config = _make_state(seed=0)
        geo = config.geometry
        hex_id = 0 if state.robber_hex != 0 else 1
        vertices = geo.hex_to_vertices[hex_id]
        v_empty, v_holder = vertices[0], vertices[1]
        state.players[1].settlement_vertices.add(v_empty)
        state.players[1].resources = [0] * 5
        state.players[2].settlement_vertices.add(v_holder)
        state.players[2].resources = [1, 0, 0, 0, 0]
        state.pending_steal_hex = hex_id
        state.phase = Phase.STEAL

        acts = _steal_actions(state)
        target_pids = {a.player_id for a in acts}
        assert target_pids == {2}


# ---------------------------------------------------------------------------
# Longest-road revocation
# ---------------------------------------------------------------------------

def _greedy_road_chain(geo, start_v, length, used_edges, avoid_vertices):
    """Walk from start_v adding up to `length` unused edges to a straight
    path, never stepping onto a vertex in avoid_vertices. Mutates
    used_edges in place. Returns (edges, vertices); board geometry is fixed
    (independent of seed), so the lengths requested below are known-good."""
    edges = []
    vertices = [start_v]
    current = start_v
    for _ in range(length):
        found = False
        for e in sorted(geo.vertex_to_edges[current]):
            if e in used_edges:
                continue
            va, vb = geo.edge_to_vertices[e]
            other = vb if va == current else va
            if other in avoid_vertices:
                continue
            edges.append(e)
            vertices.append(other)
            used_edges.add(e)
            current = other
            found = True
            break
        if not found:
            break
    return edges, vertices


class TestLongestRoadRevocation:
    """Official rule: longest road is recomputed on every road/settlement
    change; the holder keeps the card only while still >= LONGEST_ROAD_MIN
    (5) and unbeaten. If an opponent's settlement splits the holder's road
    below 5, the card is revoked -- awarded to the unique remaining player
    at >= 5, or to nobody on a tie / if nobody qualifies."""

    def _split_scenario(self, seed=0):
        """Build a 4-player board, roads only (no settlements/resources):
          - player 1: a straight 6-edge chain, split at its 4th vertex
          - player 0: an independent straight 5-edge chain elsewhere
          - player 3: an independent straight 5-edge chain elsewhere
        None of the three chains touch. Returns (state, config, mid_vertex)
        where mid_vertex is the vertex that splits player 1's chain into
        two 3-edge halves.
        """
        config = BoardConfig.standard(seed=seed)
        state = GameState.new_game(config, n_players=4, seed=seed)
        geo = config.geometry
        for p in state.players:
            p.settlement_vertices = set()
            p.city_vertices = set()
            p.road_vertices = set()
            p.roads_built = 0
            p.resources = [0] * 5

        used_edges = set()
        blocked = set()

        e1, v1 = _greedy_road_chain(geo, 0, 6, used_edges, blocked)
        assert len(e1) == 6, "geometry changed: 6-edge chain from vertex 0 unavailable"
        blocked |= set(v1)

        e0, v0 = _greedy_road_chain(geo, 53, 5, used_edges, blocked)
        assert len(e0) == 5, "geometry changed: 5-edge chain from vertex 53 unavailable"
        blocked |= set(v0)

        e3, v3 = _greedy_road_chain(geo, 27, 5, used_edges, blocked)
        assert len(e3) == 5, "geometry changed: 5-edge chain from vertex 27 unavailable"
        blocked |= set(v3)

        state.players[1].road_vertices = set(e1)
        state.players[1].roads_built = len(e1)
        state.players[0].road_vertices = set(e0)
        state.players[0].roads_built = len(e0)
        state.players[3].road_vertices = set(e3)
        state.players[3].roads_built = len(e3)

        mid_v = v1[3]
        assert mid_v not in v0 and mid_v not in v3

        return state, config, mid_v

    def _split_player1_road(self, state, mid_v):
        """Player 2 builds a settlement at mid_v, splitting player 1's road
        (this is the same apply_action path a real settlement build takes,
        so it exercises the update_longest_road call already wired into
        _build_settlement)."""
        state.phase = Phase.MAIN
        state.current_player = 2
        state.players[2].resources = [1, 1, 1, 1, 0]
        apply_action(state, settlement_action(mid_v))

    def test_holder_revoked_below_5_no_other_qualifier(self):
        """(c) Opponent settlement splits the holder's road below 5; nobody
        else has >=5 -> holder becomes None."""
        state, config, mid_v = self._split_scenario(seed=0)
        state.players[0].road_vertices = set()
        state.players[0].roads_built = 0
        state.players[3].road_vertices = set()
        state.players[3].roads_built = 0

        update_longest_road(state)
        assert state.longest_road_holder == 1, "setup: player 1 should start as holder"
        assert compute_longest_road(1, state) == 6

        self._split_player1_road(state, mid_v)

        assert compute_longest_road(1, state) < 5
        assert state.longest_road_holder is None

    def test_holder_revoked_transfers_to_unique_qualifier(self):
        """(d) Same split, but player 3 uniquely still has >=5 -> the card
        transfers to player 3."""
        state, config, mid_v = self._split_scenario(seed=0)
        state.players[0].road_vertices = set()
        state.players[0].roads_built = 0
        # player 3 keeps its untouched 5-edge chain

        update_longest_road(state)
        assert state.longest_road_holder == 1, "setup: player 1 should start as holder"

        self._split_player1_road(state, mid_v)

        assert compute_longest_road(1, state) < 5
        assert compute_longest_road(3, state) >= 5
        assert state.longest_road_holder == 3

    def test_tie_after_split_nobody_holds_it(self):
        """(e) No prior holder. Player 2's settlement splits player 1's
        road (a distractor -- player 1 was never a contender here), while
        players 0 and 3 are tied at exactly 5, the new maximum. On a tie,
        nobody gets the card."""
        state, config, mid_v = self._split_scenario(seed=0)
        assert state.longest_road_holder is None

        self._split_player1_road(state, mid_v)

        lengths = [compute_longest_road(pid, state) for pid in range(state.n_players)]
        assert lengths[0] == 5 and lengths[3] == 5, f"setup drifted: {lengths}"
        assert lengths[0] == lengths[3] == max(lengths)
        assert state.longest_road_holder is None


# ---------------------------------------------------------------------------
# Audit evidence tests: rules that were correct but previously untested
# ---------------------------------------------------------------------------

class TestDevCardDeckAndPurchase:
    """Official rules: the dev deck has 25 cards (14 knights, 2 road
    building, 2 year of plenty, 2 monopoly, 5 VP); a dev card costs
    1 sheep + 1 wheat + 1 ore; no card can be bought once the deck is empty."""

    def test_deck_composition_14_2_2_2_5(self):
        from collections import Counter
        from catan_rl.env.game_state import _DEV_DECK
        counts = Counter(_DEV_DECK)
        assert counts[DevCard.KNIGHT] == 14
        assert counts[DevCard.ROAD_BUILDING] == 2
        assert counts[DevCard.YEAR_OF_PLENTY] == 2
        assert counts[DevCard.MONOPOLY] == 2
        assert counts[DevCard.VICTORY_POINT] == 5
        assert len(_DEV_DECK) == 25

    def test_buy_dev_card_costs_sheep_wheat_ore(self):
        state, config = _make_state(seed=0)
        state.phase = Phase.MAIN
        state.rolled_this_turn = True
        p = state.players[0]
        p.resources = [0, 0, 1, 1, 1]  # exact cost
        bank_before = list(state.bank)
        deck_before = len(state.dev_deck)
        assert deck_before > 0

        acts = legal_actions(state)
        assert any(a.action_type == ActionType.BUY_DEV_CARD for a in acts)
        apply_action(state, next(a for a in acts if a.action_type == ActionType.BUY_DEV_CARD))

        assert p.resources == [0, 0, 0, 0, 0]
        assert state.bank[int(Resource.SHEEP)] == bank_before[int(Resource.SHEEP)] + 1
        assert state.bank[int(Resource.WHEAT)] == bank_before[int(Resource.WHEAT)] + 1
        assert state.bank[int(Resource.ORE)] == bank_before[int(Resource.ORE)] + 1
        assert len(state.dev_deck) == deck_before - 1
        assert sum(p.dev_cards_new) == 1, "bought card must land in dev_cards_new"

    def test_cannot_buy_when_deck_empty(self):
        state, config = _make_state(seed=0)
        state.phase = Phase.MAIN
        state.rolled_this_turn = True
        state.players[0].resources = [0, 0, 5, 5, 5]
        state.dev_deck = []

        acts = legal_actions(state)
        assert not any(a.action_type == ActionType.BUY_DEV_CARD for a in acts)


class TestPieceLimits:
    """Official rules: each player has exactly 15 roads, 5 settlements and
    4 cities; once all pieces of a kind are on the board, no more of that
    kind can be built regardless of resources."""

    def test_road_limit_15(self):
        state, config = _make_state(seed=0)
        state.phase = Phase.MAIN
        p = state.players[0]
        p.road_vertices = {0}
        p.resources = [5, 5, 0, 0, 0]

        p.roads_built = 14  # control: one piece left -> legal
        assert any(a.action_type == ActionType.BUILD_ROAD for a in legal_actions(state))

        p.roads_built = 15  # all pieces placed -> illegal
        assert not any(a.action_type == ActionType.BUILD_ROAD for a in legal_actions(state))

    def test_settlement_limit_5(self):
        state, config = _make_state(seed=0)
        state.phase = Phase.MAIN
        p = state.players[0]
        p.road_vertices = {0}  # reachable vertices exist and are unoccupied
        p.resources = [5, 5, 5, 5, 0]

        p.settlements_built = 4  # control
        assert any(a.action_type == ActionType.BUILD_SETTLEMENT for a in legal_actions(state))

        p.settlements_built = 5
        assert not any(a.action_type == ActionType.BUILD_SETTLEMENT for a in legal_actions(state))

    def test_city_limit_4(self):
        state, config = _make_state(seed=0)
        state.phase = Phase.MAIN
        p = state.players[0]
        p.settlement_vertices = {0}
        p.resources = [0, 0, 0, 5, 5]

        p.cities_built = 3  # control
        assert any(a.action_type == ActionType.BUILD_CITY for a in legal_actions(state))

        p.cities_built = 4
        assert not any(a.action_type == ActionType.BUILD_CITY for a in legal_actions(state))


class TestRoadBlockedByEnemyBuilding:
    """Official rule: a road may not be continued past an opponent's
    settlement or city (the building interrupts the route)."""

    def test_cannot_extend_road_through_enemy_settlement(self):
        state, config = _make_state(seed=0)
        geo = config.geometry
        state.phase = Phase.MAIN
        p0 = state.players[0]
        p0.road_vertices = {0}
        p0.resources = [5, 5, 0, 0, 0]

        va, vb = geo.edge_to_vertices[0]
        state.players[1].settlement_vertices.add(vb)  # enemy blocks endpoint vb

        legal_edges = {a.edge_id for a in legal_actions(state)
                       if a.action_type == ActionType.BUILD_ROAD}

        va_edges = {e for e in geo.vertex_to_edges[va] if e != 0}
        vb_edges = {e for e in geo.vertex_to_edges[vb] if e != 0}
        assert va_edges <= legal_edges, "open endpoint must remain extendable"
        assert not (vb_edges & legal_edges), \
            "edges past the enemy settlement must be illegal"
