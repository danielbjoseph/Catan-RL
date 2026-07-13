"""
Fixed 256-slot action catalog for Catan.

Every (action_type, parameter) combination has a permanent catalog index.
The policy network always outputs 256 logits; illegal slots are masked to -inf.

Catalog layout:
  0        ROLL_DICE
  1        END_TURN
  2-73     BUILD_ROAD(edge 0-71)
  74-127   BUILD_SETTLEMENT(vertex 0-53)
  128-181  BUILD_CITY(vertex 0-53)
  182-200  MOVE_ROBBER(hex 0-18)
  201-204  CHOOSE_STEAL_TARGET(player 0-3)
  205-224  MARITIME_TRADE(give, get) -- 20 ordered pairs where give != get
  225-229  DISCARD_RESOURCE(resource 0-4)
  230      BUY_DEV_CARD
  231      PLAY_KNIGHT
  232      PLAY_ROAD_BUILDING
  233-247  PLAY_YEAR_OF_PLENTY(res_a, res_b) -- 15 unordered pairs w/ repetition
  248-252  PLAY_MONOPOLY(resource 0-4)
  253      PLAY_VICTORY_POINT
  254-255  reserved
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

CATALOG_SIZE = 256


class Resource(IntEnum):
    WOOD = 0
    BRICK = 1
    SHEEP = 2
    WHEAT = 3
    ORE = 4


class DevCard(IntEnum):
    KNIGHT = 0
    ROAD_BUILDING = 1
    YEAR_OF_PLENTY = 2
    MONOPOLY = 3
    VICTORY_POINT = 4


class ActionType(IntEnum):
    ROLL_DICE = 0
    END_TURN = 1
    BUILD_ROAD = 2
    BUILD_SETTLEMENT = 3
    BUILD_CITY = 4
    MOVE_ROBBER = 5
    CHOOSE_STEAL_TARGET = 6
    MARITIME_TRADE = 7
    DISCARD_RESOURCE = 8
    BUY_DEV_CARD = 9
    PLAY_KNIGHT = 10
    PLAY_ROAD_BUILDING = 11
    PLAY_YEAR_OF_PLENTY = 12
    PLAY_MONOPOLY = 13
    PLAY_VICTORY_POINT = 14


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    # Parameters (None when not applicable)
    edge_id: Optional[int] = None        # BUILD_ROAD
    vertex_id: Optional[int] = None      # BUILD_SETTLEMENT, BUILD_CITY
    hex_id: Optional[int] = None         # MOVE_ROBBER
    player_id: Optional[int] = None      # CHOOSE_STEAL_TARGET
    resource: Optional[Resource] = None  # MARITIME_TRADE give, DISCARD, PLAY_MONOPOLY
    resource2: Optional[Resource] = None # MARITIME_TRADE get, PLAY_YEAR_OF_PLENTY second
    catalog_index: int = -1

    def __str__(self) -> str:
        t = self.action_type.name
        if self.action_type == ActionType.BUILD_ROAD:
            return f"{t}(edge={self.edge_id})"
        if self.action_type == ActionType.BUILD_SETTLEMENT:
            return f"{t}(vertex={self.vertex_id})"
        if self.action_type == ActionType.BUILD_CITY:
            return f"{t}(vertex={self.vertex_id})"
        if self.action_type == ActionType.MOVE_ROBBER:
            return f"{t}(hex={self.hex_id})"
        if self.action_type == ActionType.CHOOSE_STEAL_TARGET:
            return f"{t}(player={self.player_id})"
        if self.action_type == ActionType.MARITIME_TRADE:
            return f"{t}(give={self.resource.name}, get={self.resource2.name})"
        if self.action_type == ActionType.DISCARD_RESOURCE:
            return f"{t}(resource={self.resource.name})"
        if self.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            return f"{t}({self.resource.name}, {self.resource2.name})"
        if self.action_type == ActionType.PLAY_MONOPOLY:
            return f"{t}({self.resource.name})"
        return t


def _build_catalog() -> Tuple[list[Action], dict]:
    catalog: list[Action] = []
    index_map: dict = {}  # (action_type, params) -> catalog_index

    def add(action: Action):
        idx = len(catalog)
        obj = Action(
            action_type=action.action_type,
            edge_id=action.edge_id,
            vertex_id=action.vertex_id,
            hex_id=action.hex_id,
            player_id=action.player_id,
            resource=action.resource,
            resource2=action.resource2,
            catalog_index=idx,
        )
        catalog.append(obj)
        return obj

    # 0: ROLL_DICE
    add(Action(ActionType.ROLL_DICE))
    # 1: END_TURN
    add(Action(ActionType.END_TURN))
    # 2-73: BUILD_ROAD
    for e in range(72):
        add(Action(ActionType.BUILD_ROAD, edge_id=e))
    # 74-127: BUILD_SETTLEMENT
    for v in range(54):
        add(Action(ActionType.BUILD_SETTLEMENT, vertex_id=v))
    # 128-181: BUILD_CITY
    for v in range(54):
        add(Action(ActionType.BUILD_CITY, vertex_id=v))
    # 182-200: MOVE_ROBBER
    for h in range(19):
        add(Action(ActionType.MOVE_ROBBER, hex_id=h))
    # 201-204: CHOOSE_STEAL_TARGET
    for p in range(4):
        add(Action(ActionType.CHOOSE_STEAL_TARGET, player_id=p))
    # 205-224: MARITIME_TRADE (give != get)
    for give in Resource:
        for get in Resource:
            if give != get:
                add(Action(ActionType.MARITIME_TRADE, resource=give, resource2=get))
    # 225-229: DISCARD_RESOURCE
    for r in Resource:
        add(Action(ActionType.DISCARD_RESOURCE, resource=r))
    # 230: BUY_DEV_CARD
    add(Action(ActionType.BUY_DEV_CARD))
    # 231: PLAY_KNIGHT
    add(Action(ActionType.PLAY_KNIGHT))
    # 232: PLAY_ROAD_BUILDING
    add(Action(ActionType.PLAY_ROAD_BUILDING))
    # 233-247: PLAY_YEAR_OF_PLENTY (unordered pairs with repetition)
    for a in Resource:
        for b in Resource:
            if b >= a:
                add(Action(ActionType.PLAY_YEAR_OF_PLENTY, resource=a, resource2=b))
    # 248-252: PLAY_MONOPOLY
    for r in Resource:
        add(Action(ActionType.PLAY_MONOPOLY, resource=r))
    # 253: PLAY_VICTORY_POINT
    add(Action(ActionType.PLAY_VICTORY_POINT))
    # 254-255: padding (reserved, never legal)
    while len(catalog) < CATALOG_SIZE:
        add(Action(ActionType.ROLL_DICE))  # unreachable slots

    assert len(catalog) == CATALOG_SIZE, f"Catalog size mismatch: {len(catalog)}"
    return catalog, index_map


CATALOG, _INDEX_MAP = _build_catalog()

# Convenience accessors
ROLL_DICE    = CATALOG[0]
END_TURN     = CATALOG[1]


def road_action(edge_id: int) -> Action:
    assert 0 <= edge_id < 72
    return CATALOG[2 + edge_id]


def settlement_action(vertex_id: int) -> Action:
    assert 0 <= vertex_id < 54
    return CATALOG[74 + vertex_id]


def city_action(vertex_id: int) -> Action:
    assert 0 <= vertex_id < 54
    return CATALOG[128 + vertex_id]


def move_robber_action(hex_id: int) -> Action:
    assert 0 <= hex_id < 19
    return CATALOG[182 + hex_id]


def steal_action(player_id: int) -> Action:
    assert 0 <= player_id < 4
    return CATALOG[201 + player_id]


def maritime_trade_action(give: Resource, get: Resource) -> Action:
    assert give != get
    idx = 205
    for g in Resource:
        for r in Resource:
            if g != r:
                if g == give and r == get:
                    return CATALOG[idx]
                idx += 1
    raise ValueError(f"Invalid maritime trade: {give} -> {get}")


def discard_action(resource: Resource) -> Action:
    return CATALOG[225 + int(resource)]


BUY_DEV_CARD    = CATALOG[230]
PLAY_KNIGHT     = CATALOG[231]
PLAY_ROAD_BUILDING = CATALOG[232]


def year_of_plenty_action(res_a: Resource, res_b: Resource) -> Action:
    a, b = min(res_a, res_b), max(res_a, res_b)
    idx = 233
    for x in Resource:
        for y in Resource:
            if y >= x:
                if x == a and y == b:
                    return CATALOG[idx]
                idx += 1
    raise ValueError(f"Invalid year of plenty: {res_a}, {res_b}")


def monopoly_action(resource: Resource) -> Action:
    return CATALOG[248 + int(resource)]


PLAY_VICTORY_POINT = CATALOG[253]
