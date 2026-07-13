"""
Smoke test: run one complete game with random agents and print a summary.
Usage: python scripts/play_random_game.py [--seed N] [--quiet]
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
from catan_rl.env.validators import legal_actions
from catan_rl.env.scoring import compute_vp
from catan_rl.bots.random_bot import pick_action


def play_game(seed: int = 0, quiet: bool = False, max_turns: int = 5000) -> dict:
    rng = random.Random(seed)
    config = BoardConfig.standard(seed=seed)
    state = GameState.new_game(config, n_players=4, seed=seed)

    turn = 0
    start = time.perf_counter()

    while not state.is_terminal and turn < max_turns:
        action = pick_action(state, rng)
        if not quiet:
            print(f"  turn={turn:4d} p{state.current_player} phase={state.phase.name:22s} {action}")
        apply_action(state, action, rng)
        turn += 1

    elapsed = time.perf_counter() - start
    vps = [compute_vp(pid, state) for pid in range(4)]

    result = {
        "winner": state.winner,
        "turns": turn,
        "vps": vps,
        "elapsed_s": elapsed,
        "terminated": state.is_terminal,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--n", type=int, default=1, help="Number of games to run")
    args = parser.parse_args()

    total_turns = 0
    wins = [0, 0, 0, 0]
    start = time.perf_counter()

    for i in range(args.n):
        result = play_game(seed=args.seed + i, quiet=args.quiet or args.n > 1)
        total_turns += result["turns"]
        if result["winner"] is not None:
            wins[result["winner"]] += 1
        if args.n == 1 or (i + 1) % max(1, args.n // 10) == 0:
            print(f"Game {i+1}/{args.n}: winner=P{result['winner']} "
                  f"turns={result['turns']} vp={result['vps']} "
                  f"time={result['elapsed_s']:.2f}s")

    total_time = time.perf_counter() - start
    print(f"\n--- Summary ({args.n} games) ---")
    print(f"Total turns: {total_turns}  avg: {total_turns/args.n:.1f}")
    print(f"Win counts: {wins}")
    print(f"Total time: {total_time:.2f}s  ({args.n/total_time:.1f} games/s)")


if __name__ == "__main__":
    main()
