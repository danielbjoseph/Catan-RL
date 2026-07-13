"""Per-player mutable state."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .actions import DevCard, Resource

# Build costs: {resource: count}
BUILD_COSTS: Dict[str, Dict[Resource, int]] = {
    "road":       {Resource.WOOD: 1, Resource.BRICK: 1},
    "settlement": {Resource.WOOD: 1, Resource.BRICK: 1, Resource.SHEEP: 1, Resource.WHEAT: 1},
    "city":       {Resource.WHEAT: 2, Resource.ORE: 3},
    "dev_card":   {Resource.SHEEP: 1, Resource.WHEAT: 1, Resource.ORE: 1},
}

MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4


@dataclass
class PlayerState:
    player_id: int
    resources: List[int] = field(default_factory=lambda: [0] * 5)   # indexed by Resource
    dev_cards: List[int] = field(default_factory=lambda: [0] * 5)    # indexed by DevCard, in hand
    dev_cards_new: List[int] = field(default_factory=lambda: [0] * 5) # bought this turn, unplayable
    played_dev_cards: List[int] = field(default_factory=lambda: [0] * 5)

    road_vertices: Set[int] = field(default_factory=set)   # edge IDs with roads
    settlement_vertices: Set[int] = field(default_factory=set)
    city_vertices: Set[int] = field(default_factory=set)

    roads_built: int = 0
    settlements_built: int = 0
    cities_built: int = 0

    has_played_dev_card: bool = False  # can only play one per turn
    army_size: int = 0                 # knights played total

    def clone(self) -> PlayerState:
        p = PlayerState(self.player_id)
        p.resources = list(self.resources)
        p.dev_cards = list(self.dev_cards)
        p.dev_cards_new = list(self.dev_cards_new)
        p.played_dev_cards = list(self.played_dev_cards)
        p.road_vertices = set(self.road_vertices)
        p.settlement_vertices = set(self.settlement_vertices)
        p.city_vertices = set(self.city_vertices)
        p.roads_built = self.roads_built
        p.settlements_built = self.settlements_built
        p.cities_built = self.cities_built
        p.has_played_dev_card = self.has_played_dev_card
        p.army_size = self.army_size
        return p

    # ---------------------------------------------------------------------------
    # Resource helpers
    # ---------------------------------------------------------------------------

    @property
    def total_resources(self) -> int:
        return sum(self.resources)

    def has_resources(self, cost: Dict[Resource, int]) -> bool:
        return all(self.resources[int(r)] >= n for r, n in cost.items())

    def spend(self, cost: Dict[Resource, int]):
        for r, n in cost.items():
            self.resources[int(r)] -= n

    def gain(self, resource: Resource, count: int = 1):
        self.resources[int(resource)] += count

    # ---------------------------------------------------------------------------
    # Build helpers
    # ---------------------------------------------------------------------------

    @property
    def roads_available(self) -> int:
        return MAX_ROADS - self.roads_built

    @property
    def settlements_available(self) -> int:
        return MAX_SETTLEMENTS - self.settlements_built

    @property
    def cities_available(self) -> int:
        return MAX_CITIES - self.cities_built

    def can_afford_road(self) -> bool:
        return self.roads_available > 0 and self.has_resources(BUILD_COSTS["road"])

    def can_afford_settlement(self) -> bool:
        return self.settlements_available > 0 and self.has_resources(BUILD_COSTS["settlement"])

    def can_afford_city(self) -> bool:
        return self.cities_available > 0 and self.has_resources(BUILD_COSTS["city"])

    def can_afford_dev_card(self) -> bool:
        return self.has_resources(BUILD_COSTS["dev_card"])

    # ---------------------------------------------------------------------------
    # Dev card helpers
    # ---------------------------------------------------------------------------

    def has_playable_dev_card(self, card: DevCard) -> bool:
        return self.dev_cards[int(card)] > 0 and not self.has_played_dev_card

    def receive_dev_card(self, card: DevCard):
        self.dev_cards_new[int(card)] += 1

    def end_turn_refresh_dev_cards(self):
        """Move newly bought cards into the playable hand."""
        for i in range(len(self.dev_cards)):
            self.dev_cards[i] += self.dev_cards_new[i]
            self.dev_cards_new[i] = 0
        self.has_played_dev_card = False

    def play_dev_card(self, card: DevCard):
        assert self.dev_cards[int(card)] > 0
        self.dev_cards[int(card)] -= 1
        self.played_dev_cards[int(card)] += 1
        self.has_played_dev_card = True

    # ---------------------------------------------------------------------------
    # VP
    # ---------------------------------------------------------------------------

    @property
    def public_vp(self) -> int:
        return self.settlements_built + 2 * self.cities_built

    @property
    def hidden_vp(self) -> int:
        return self.played_dev_cards[int(DevCard.VICTORY_POINT)]

    @property
    def total_vp(self) -> int:
        return self.public_vp + self.hidden_vp

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "resources": list(self.resources),
            "dev_cards": list(self.dev_cards),
            "dev_cards_new": list(self.dev_cards_new),
            "played_dev_cards": list(self.played_dev_cards),
            "road_vertices": sorted(self.road_vertices),
            "settlement_vertices": sorted(self.settlement_vertices),
            "city_vertices": sorted(self.city_vertices),
            "roads_built": self.roads_built,
            "settlements_built": self.settlements_built,
            "cities_built": self.cities_built,
            "has_played_dev_card": self.has_played_dev_card,
            "army_size": self.army_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PlayerState:
        p = cls(d["player_id"])
        p.resources = d["resources"]
        p.dev_cards = d["dev_cards"]
        p.dev_cards_new = d["dev_cards_new"]
        p.played_dev_cards = d["played_dev_cards"]
        p.road_vertices = set(d["road_vertices"])
        p.settlement_vertices = set(d["settlement_vertices"])
        p.city_vertices = set(d["city_vertices"])
        p.roads_built = d["roads_built"]
        p.settlements_built = d["settlements_built"]
        p.cities_built = d["cities_built"]
        p.has_played_dev_card = d["has_played_dev_card"]
        p.army_size = d["army_size"]
        return p
