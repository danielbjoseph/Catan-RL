"""
Environment throughput benchmark (spec Phase 4 target: >= 500 games/hour on CPU).

Runs complete random-agent games through the raw engine and reports
games/hour, mean turns, and plies/second. Exits 1 if below target.

Usage: python scripts/benchmark_throughput.py [--games 100] [--profile simplified_v1]
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from catan_rl.env.board import BoardConfig
from catan_rl.env.game_state import GameState
from catan_rl.env.rules import apply_action
from catan_rl.env.rules_profile import RulesProfile
from catan_rl.bots.random_bot import pick_action

TARGET_GAMES_PER_HOUR = 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--profile", default="simplified_v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=20000)
    args = parser.parse_args()

    profile = RulesProfile.get(args.profile)
    total_plies = 0
    total_turns = 0
    completed = 0

    start = time.perf_counter()
    for i in range(args.games):
        seed = args.seed + i
        rng = random.Random(seed)
        config = BoardConfig.standard(seed=seed)
        state = GameState.new_game(config, n_players=4, seed=seed, profile=profile)
        plies = 0
        while not state.is_terminal and plies < args.max_plies:
            apply_action(state, pick_action(state, rng), rng)
            plies += 1
        total_plies += plies
        total_turns += state.turn_number
        if state.is_terminal:
            completed += 1
    elapsed = time.perf_counter() - start

    games_per_hour = args.games / elapsed * 3600
    print(f"profile:        {profile.name}")
    print(f"games:          {args.games} ({completed} completed)")
    print(f"elapsed:        {elapsed:.2f}s")
    print(f"games/hour:     {games_per_hour:,.0f}")
    print(f"games/second:   {args.games / elapsed:.2f}")
    print(f"mean turns:     {total_turns / args.games:.1f}")
    print(f"plies/second:   {total_plies / elapsed:,.0f}")

    if games_per_hour < TARGET_GAMES_PER_HOUR:
        print(f"FAIL: below target of {TARGET_GAMES_PER_HOUR} games/hour", file=sys.stderr)
        sys.exit(1)
    print(f"OK: exceeds target of {TARGET_GAMES_PER_HOUR} games/hour")


if __name__ == "__main__":
    main()
