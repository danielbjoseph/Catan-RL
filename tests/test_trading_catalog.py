import pytest
from catan_rl.env.actions import (
    CATALOG, CATALOG_SIZE, CATALOG_SIZE_V1, CATALOG_VERSION,
    Action, ActionType, Resource,
    propose_trade_action, ACCEPT_TRADE, DECLINE_TRADE,
    maritime_trade_action, monopoly_action, PLAY_VICTORY_POINT,
)

def test_catalog_sizes():
    assert CATALOG_SIZE == 512
    assert CATALOG_SIZE_V1 == 256
    assert CATALOG_VERSION == 2
    assert len(CATALOG) == 512

def test_v1_prefix_frozen():
    # Spot-check every v1 segment boundary is untouched.
    assert CATALOG[0].action_type == ActionType.ROLL_DICE
    assert CATALOG[1].action_type == ActionType.END_TURN
    assert CATALOG[2].action_type == ActionType.BUILD_ROAD and CATALOG[2].edge_id == 0
    assert CATALOG[73].edge_id == 71
    assert CATALOG[74].vertex_id == 0 and CATALOG[74].action_type == ActionType.BUILD_SETTLEMENT
    assert CATALOG[181].action_type == ActionType.BUILD_CITY
    assert CATALOG[200].hex_id == 18
    assert CATALOG[204].player_id == 3
    assert maritime_trade_action(Resource.WOOD, Resource.BRICK).catalog_index == 205
    assert CATALOG[230].action_type == ActionType.BUY_DEV_CARD
    assert monopoly_action(Resource.ORE).catalog_index == 252
    assert PLAY_VICTORY_POINT.catalog_index == 253

def test_propose_trade_slots():
    a = propose_trade_action(Resource.WOOD, Resource.BRICK, give_n=1)
    assert a.catalog_index == 256
    assert a.action_type == ActionType.PROPOSE_TRADE
    assert a.resource == Resource.WOOD and a.resource2 == Resource.BRICK and a.give_n == 1
    b = propose_trade_action(Resource.WOOD, Resource.BRICK, give_n=2)
    assert b.catalog_index == 257
    last = propose_trade_action(Resource.ORE, Resource.WHEAT, give_n=2)
    assert last.catalog_index == 295
    with pytest.raises(AssertionError):
        propose_trade_action(Resource.WOOD, Resource.WOOD, give_n=1)
    with pytest.raises(AssertionError):
        propose_trade_action(Resource.WOOD, Resource.BRICK, give_n=3)

def test_accept_decline_slots():
    assert ACCEPT_TRADE.catalog_index == 296
    assert ACCEPT_TRADE.action_type == ActionType.ACCEPT_TRADE
    assert DECLINE_TRADE.catalog_index == 297
    assert DECLINE_TRADE.action_type == ActionType.DECLINE_TRADE

def test_action_str():
    a = propose_trade_action(Resource.SHEEP, Resource.ORE, give_n=2)
    assert str(a) == "PROPOSE_TRADE(give=2xSHEEP, get=ORE)"
    assert str(ACCEPT_TRADE) == "ACCEPT_TRADE"
    assert str(DECLINE_TRADE) == "DECLINE_TRADE"

def test_padding_never_real():
    for i in range(298, 512):
        assert CATALOG[i].action_type == ActionType.ROLL_DICE  # unreachable filler
