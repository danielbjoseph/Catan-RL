"""
Mutable full game state.

Phase enum tracks what action is expected from the current player.
GameState holds all mutable state; clone() produces a deep copy for rollouts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from .actions import DevCard, Resource
from .board import BoardConfig
from .player_state import PlayerState
from .rules_profile import RulesProfile, STANDARD

WIN_VP = 10

# Dev card deck composition
_DEV_DECK: List[DevCard] = (
    [DevCard.KNIGHT] * 14
    + [DevCard.ROAD_BUILDING] * 2
    + [DevCard.YEAR_OF_PLENTY] * 2
    + [DevCard.MONOPOLY] * 2
    + [DevCard.VICTORY_POINT] * 5
)

# Bank starting counts
_BANK_START: List[int] = [19, 19, 19, 19, 19]  # indexed by Resource


class Phase(IntEnum):
    SETUP_SETTLEMENT_1 = 0   # forward order: place first settlement
    SETUP_ROAD_1       = 1   # place first road
    SETUP_SETTLEMENT_2 = 2   # reverse order: place second settlement
    SETUP_ROAD_2       = 3   # place second road
    ROLL               = 4   # must roll dice
    ROBBER             = 5   # must move robber (after 7 or knight)
    STEAL              = 6   # must choose steal target
    DISCARD            = 7   # one or more players must discard
    ROAD_BUILDING_1    = 8   # first road of road-building dev card
    ROAD_BUILDING_2    = 9   # second road of road-building dev card
    MAIN               = 10  # normal turn actions (build/trade/dev/end)
    GAME_OVER          = 11


@dataclass
class GameState:
    config: BoardConfig
    players: List[PlayerState]
    current_player: int = 0
    phase: Phase = Phase.SETUP_SETTLEMENT_1
    robber_hex: int = 0         # hex_id; set to desert_hex on init
    bank: List[int] = field(default_factory=lambda: list(_BANK_START))
    dev_deck: List[DevCard] = field(default_factory=list)
    dice: Optional[Tuple[int, int]] = None
    longest_road_holder: Optional[int] = None
    largest_army_holder: Optional[int] = None
    winner: Optional[int] = None
    turn_number: int = 0
    profile: RulesProfile = field(default_factory=lambda: STANDARD)

    # Pending state for sub-phases
    pending_steal_hex: Optional[int] = None
    discard_obligations: Dict[int, int] = field(default_factory=dict)  # player_id -> n to discard

    # Setup phase tracking: which player index (in setup order) is acting
    _setup_forward_idx: int = 0   # counts up during SETUP_*_1 phases
    _setup_backward_idx: int = 0  # counts down during SETUP_*_2 phases

    @classmethod
    def new_game(
        cls,
        config: BoardConfig,
        n_players: int = 4,
        seed: Optional[int] = None,
        profile: Optional[RulesProfile | str] = None,
    ) -> GameState:
        profile = RulesProfile.get(profile)
        rng = random.Random(seed)
        players = [PlayerState(i) for i in range(n_players)]

        if profile.dev_cards_enabled:
            dev_deck = list(_DEV_DECK)
            rng.shuffle(dev_deck)
        else:
            dev_deck = []

        state = cls(
            config=config,
            players=players,
            current_player=0,
            phase=Phase.SETUP_SETTLEMENT_1,
            robber_hex=config.desert_hex,
            bank=list(_BANK_START),
            dev_deck=dev_deck,
            profile=profile,
        )
        state._setup_forward_idx = 0
        state._setup_backward_idx = n_players - 1
        return state

    # ---------------------------------------------------------------------------
    # Convenience properties
    # ---------------------------------------------------------------------------

    @property
    def n_players(self) -> int:
        return len(self.players)

    @property
    def current(self) -> PlayerState:
        return self.players[self.current_player]

    @property
    def is_terminal(self) -> bool:
        return self.phase == Phase.GAME_OVER

    def all_road_edges(self) -> Dict[int, int]:
        """Returns {edge_id: player_id} for all placed roads."""
        result: Dict[int, int] = {}
        for p in self.players:
            for e in p.road_vertices:
                result[e] = p.player_id
        return result

    def all_settlement_vertices(self) -> Dict[int, int]:
        """Returns {vertex_id: player_id} for all settlements."""
        result: Dict[int, int] = {}
        for p in self.players:
            for v in p.settlement_vertices:
                result[v] = p.player_id
        return result

    def all_city_vertices(self) -> Dict[int, int]:
        """Returns {vertex_id: player_id} for all cities."""
        result: Dict[int, int] = {}
        for p in self.players:
            for v in p.city_vertices:
                result[v] = p.player_id
        return result

    def all_occupied_vertices(self) -> Dict[int, int]:
        """Returns {vertex_id: player_id} for all settlements + cities."""
        d = self.all_settlement_vertices()
        d.update(self.all_city_vertices())
        return d

    # ---------------------------------------------------------------------------
    # Bank helpers
    # ---------------------------------------------------------------------------

    def bank_give(self, resource: Resource, count: int):
        """Move resources from bank to a player (caller's responsibility to update player)."""
        self.bank[int(resource)] -= count

    def bank_take(self, resource: Resource, count: int):
        """Move resources from player back to bank."""
        self.bank[int(resource)] += count

    def bank_has(self, resource: Resource, count: int) -> bool:
        return self.bank[int(resource)] >= count

    # ---------------------------------------------------------------------------
    # Clone
    # ---------------------------------------------------------------------------

    def clone(self) -> GameState:
        s = GameState.__new__(GameState)
        s.config = self.config  # immutable, share reference
        s.players = [p.clone() for p in self.players]
        s.current_player = self.current_player
        s.phase = self.phase
        s.robber_hex = self.robber_hex
        s.bank = list(self.bank)
        s.dev_deck = list(self.dev_deck)
        s.dice = self.dice
        s.longest_road_holder = self.longest_road_holder
        s.largest_army_holder = self.largest_army_holder
        s.winner = self.winner
        s.turn_number = self.turn_number
        s.profile = self.profile  # immutable, share reference
        s.pending_steal_hex = self.pending_steal_hex
        s.discard_obligations = dict(self.discard_obligations)
        s._setup_forward_idx = self._setup_forward_idx
        s._setup_backward_idx = self._setup_backward_idx
        return s

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "players": [p.to_dict() for p in self.players],
            "current_player": self.current_player,
            "phase": int(self.phase),
            "robber_hex": self.robber_hex,
            "bank": self.bank,
            "dev_deck": [int(c) for c in self.dev_deck],
            "dice": list(self.dice) if self.dice else None,
            "longest_road_holder": self.longest_road_holder,
            "largest_army_holder": self.largest_army_holder,
            "winner": self.winner,
            "turn_number": self.turn_number,
            "profile": self.profile.to_dict(),
            "pending_steal_hex": self.pending_steal_hex,
            "discard_obligations": {str(k): v for k, v in self.discard_obligations.items()},
            "_setup_forward_idx": self._setup_forward_idx,
            "_setup_backward_idx": self._setup_backward_idx,
        }

    @classmethod
    def from_dict(cls, d: dict, config: BoardConfig) -> GameState:
        s = cls.__new__(cls)
        s.config = config
        s.players = [PlayerState.from_dict(p) for p in d["players"]]
        s.current_player = d["current_player"]
        s.phase = Phase(d["phase"])
        s.robber_hex = d["robber_hex"]
        s.bank = d["bank"]
        s.dev_deck = [DevCard(c) for c in d["dev_deck"]]
        s.dice = tuple(d["dice"]) if d["dice"] else None
        s.longest_road_holder = d["longest_road_holder"]
        s.largest_army_holder = d["largest_army_holder"]
        s.winner = d["winner"]
        s.turn_number = d["turn_number"]
        s.profile = RulesProfile.from_dict(d.get("profile"))
        s.pending_steal_hex = d["pending_steal_hex"]
        s.discard_obligations = {int(k): v for k, v in d["discard_obligations"].items()}
        s._setup_forward_idx = d["_setup_forward_idx"]
        s._setup_backward_idx = d["_setup_backward_idx"]
        return s
