"""Scripted bots: random, greedy, heuristic, and trade personalities."""

from typing import Callable

from . import greedy_bot, heuristic_bot, random_bot
from .personalities import (
    PERSONALITIES,
    TradePersonality,
    make_personality_bot,
    resource_pips,
    trade_margin,
)

BotFn = Callable[..., object]  # pick_action(state, rng) -> Action

_BASE_BOTS = {
    "random": random_bot.pick_action,
    "greedy": greedy_bot.pick_action,
    "heuristic": heuristic_bot.pick_action,
}


def resolve_bot(name: str) -> BotFn:
    """Resolve a bot/personality name to a pick_action(state, rng) -> Action
    callable.

    Accepts the base scripted bots ("random", "greedy", "heuristic") or any
    of the trade personality preset names in PERSONALITIES. Raises
    ValueError listing all valid names otherwise.
    """
    if name in _BASE_BOTS:
        return _BASE_BOTS[name]
    if name in PERSONALITIES:
        return make_personality_bot(PERSONALITIES[name])
    valid = sorted(_BASE_BOTS) + sorted(PERSONALITIES)
    raise ValueError(f"Unknown bot {name!r}; expected one of {valid}")


__all__ = [
    "PERSONALITIES",
    "TradePersonality",
    "make_personality_bot",
    "resource_pips",
    "trade_margin",
    "resolve_bot",
    "BotFn",
]
