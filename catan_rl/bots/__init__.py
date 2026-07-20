"""Scripted bots: random, greedy, heuristic, and trade personalities."""

from .personalities import (
    PERSONALITIES,
    TradePersonality,
    make_personality_bot,
    resource_pips,
    trade_margin,
)

__all__ = [
    "PERSONALITIES",
    "TradePersonality",
    "make_personality_bot",
    "resource_pips",
    "trade_margin",
]
