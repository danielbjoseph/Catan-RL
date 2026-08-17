"""
BeliefTracker: table-view resource belief model.

Models what a perfect-memory human sitting AT THE TABLE could deduce about
each player's hand from public events only (no private/hole-card info for
anyone, including the observer's own opponents). This is distinct from the
"perfect" observation mode, which exposes ground truth; BeliefTracker instead
maintains, per player, an expected resource-composition vector plus a
running "hidden mass" counter used to derive an uncertainty score.

Update dispatch (see spec table):
  - DISCARD_RESOURCE and CHOOSE_STEAL_TARGET are the only "hidden" events:
    the resource *type* touched is not publicly observable, only the count.
  - Every other action (production, builds, buys, maritime trade, year of
    plenty, monopoly, robber moves, dev-card plays, end turn, ...) is a
    "public-delta" event: the exact per-player resource delta is public
    knowledge, so it is applied directly.
  - Monopoly gets one extra step on top of the public delta: since it fully
    reveals the true count of the monopolized resource for every player
    (victims driven to exactly 0, thief's gain publicly counted), the
    tracker overwrites that resource column with ground truth for all
    players, and the subsequent renormalization PINS that column: only the
    non-reconciled columns absorb the sum correction, so the reconciled
    column stays exact even when it carried residual error from an earlier
    hidden event. This is a genuine reconciliation event, but as a
    documented simplification we do NOT decrement the hidden-mass counter
    for it (the uncertainty metric stays conservative/elevated).

After every update, `expected[pid]` is renormalized (clipped to >= 0, then
rescaled) so that `expected[pid].sum() == players[pid].total_resources`,
which is always public knowledge (hand size is visible even when
composition isn't).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from .actions import Action, ActionType

if TYPE_CHECKING:
    from .game_state import GameState

N_RESOURCES = 5
N_DEV_CARDS = 5

# Initial dev-card deck composition, indexed by DevCard IntEnum order
# (KNIGHT, ROAD_BUILDING, YEAR_OF_PLENTY, MONOPOLY, VICTORY_POINT).
_INITIAL_DEV_COMPOSITION = np.array([14, 2, 2, 2, 5], dtype=np.float32)


class BeliefTracker:
    """Tracks a table-view belief over each player's hand composition."""

    def __init__(self, state: "GameState"):
        self.n_players = state.n_players
        self.reset(state)

    def reset(self, state: "GameState") -> None:
        """Re-anchor beliefs to the given state's (assumed fully known) hands."""
        self.n_players = state.n_players
        self._expected = np.array(
            [list(p.resources) for p in state.players], dtype=np.float32
        )
        self.hidden_mass = np.zeros(self.n_players, dtype=np.float32)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def on_action(self, state_before: "GameState", action: Action, state_after: "GameState") -> None:
        t = action.action_type

        if t == ActionType.DISCARD_RESOURCE:
            self._apply_discard(state_before, state_after)
        elif t == ActionType.CHOOSE_STEAL_TARGET:
            self._apply_steal(state_before, action, state_after)
        elif t == ActionType.PLAY_MONOPOLY:
            self._apply_public_delta(state_before, state_after)
            self._apply_monopoly_reconciliation(action, state_after)
            # The reconciled column is ground truth for every player; the
            # renormalization must not rescale it away from that.
            self._renormalize(state_after, pinned=int(action.resource))
            return
        else:
            self._apply_public_delta(state_before, state_after)

        self._renormalize(state_after)

    def _apply_public_delta(self, state_before: "GameState", state_after: "GameState") -> None:
        for i in range(self.n_players):
            before_vec = np.asarray(state_before.players[i].resources, dtype=np.float32)
            after_vec = np.asarray(state_after.players[i].resources, dtype=np.float32)
            self._expected[i] += after_vec - before_vec

    def _apply_monopoly_reconciliation(self, action: Action, state_after: "GameState") -> None:
        r = int(action.resource)
        for i in range(self.n_players):
            self._expected[i][r] = float(state_after.players[i].resources[r])

    def _apply_discard(self, state_before: "GameState", state_after: "GameState") -> None:
        pid = state_before.current_player
        before_total = state_before.players[pid].total_resources
        after_total = state_after.players[pid].total_resources
        amount = before_total - after_total
        if amount > 0 and before_total > 0:
            frac = amount / before_total
            self._expected[pid] = self._expected[pid] * (1.0 - frac)
            self.hidden_mass[pid] += amount

    def _apply_steal(self, state_before: "GameState", action: Action, state_after: "GameState") -> None:
        thief = state_before.current_player
        victim = action.player_id
        total_before_victim = state_before.players[victim].total_resources
        if total_before_victim > 0:
            card_composition = self._expected[victim] / total_before_victim
            self._expected[victim] = self._expected[victim] - card_composition
            self._expected[thief] = self._expected[thief] + card_composition
            self.hidden_mass[victim] += 1
            self.hidden_mass[thief] += 1

    def _renormalize(self, state: "GameState", pinned: int | None = None) -> None:
        """Clip beliefs >= 0 and rescale so each player's vector sums to
        their true (public) hand size.

        With ``pinned=r``, column r has just been set to ground truth for
        every player (monopoly reconciliation) and must stay exact: the sum
        correction is distributed across the non-pinned columns only, by
        scaling them to ``hand - true_r`` (uniform over the 4 non-pinned
        columns if they sum to ~0 but mass remains).
        """
        for i in range(self.n_players):
            hand = state.players[i].total_resources
            vec = np.clip(self._expected[i], 0.0, None)

            if pinned is None:
                total = float(vec.sum())
                if hand == 0:
                    vec = np.zeros(N_RESOURCES, dtype=np.float32)
                elif total <= 1e-9:
                    vec = np.full(N_RESOURCES, hand / N_RESOURCES, dtype=np.float32)
                else:
                    vec = vec * (hand / total)
            else:
                true_r = float(state.players[i].resources[pinned])
                target = hand - true_r  # >= 0: hand includes the true count
                others = np.ones(N_RESOURCES, dtype=bool)
                others[pinned] = False
                others_total = float(vec[others].sum())
                if target <= 1e-9:
                    vec[others] = 0.0
                elif others_total <= 1e-9:
                    vec[others] = target / (N_RESOURCES - 1)
                else:
                    vec[others] *= target / others_total
                vec[pinned] = true_r

            self._expected[i] = vec.astype(np.float32)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def expected(self, pid: int) -> np.ndarray:
        return self._expected[pid].copy()

    def uncertainty(self, pid: int) -> float:
        hand = float(self._expected[pid].sum())
        return float(min(self.hidden_mass[pid] / max(hand, 1.0), 1.0))

    def dev_deck_estimate(self, observer: int, state: "GameState") -> Tuple[np.ndarray, int]:
        """Believed composition of the remaining (face-down) dev deck.

        Remaining count is always public (`len(state.dev_deck)`). The
        believed composition starts from the known initial deck, subtracts
        every player's publicly-played cards, and subtracts the observer's
        own held/newly-bought cards (which the observer knows aren't in the
        deck). The result is clipped >= 0 and rescaled to sum to the true
        remaining count.
        """
        count = len(state.dev_deck)

        composition = np.array(_INITIAL_DEV_COMPOSITION, dtype=np.float32)
        for p in state.players:
            composition -= np.asarray(p.played_dev_cards, dtype=np.float32)

        observer_player = state.players[observer]
        composition -= np.asarray(observer_player.dev_cards, dtype=np.float32)
        composition -= np.asarray(observer_player.dev_cards_new, dtype=np.float32)

        composition = np.clip(composition, 0.0, None)
        total = float(composition.sum())
        if count == 0:
            composition = np.zeros(N_DEV_CARDS, dtype=np.float32)
        elif total <= 1e-9:
            composition = np.full(N_DEV_CARDS, count / N_DEV_CARDS, dtype=np.float32)
        else:
            composition = composition * (count / total)

        return composition.astype(np.float32), count
