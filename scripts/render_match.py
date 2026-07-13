"""
Render a full match turn by turn as text.

Seats can be filled by a trained checkpoint and/or scripted bots.

Usage:
  python scripts/render_match.py --seed 42                       # 4 random bots
  python scripts/render_match.py --bots greedy,random,heuristic,random
  python scripts/render_match.py --ckpt runs/ppo_baseline/checkpoints/ckpt_000100.pt --seat 0
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from catan_rl.bots import greedy_bot, heuristic_bot, random_bot
from catan_rl.env.actions import CATALOG
from catan_rl.env.board import BoardConfig
from catan_rl.env.game_state import GameState
from catan_rl.env.rules import apply_action
from catan_rl.env.rules_profile import RulesProfile
from catan_rl.env.scoring import compute_vp

BOTS = {
    "random": random_bot.pick_action,
    "greedy": greedy_bot.pick_action,
    "heuristic": heuristic_bot.pick_action,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bots", default="random,random,random,random",
                        help="Comma list of 4 seat bots: random|greedy|heuristic")
    parser.add_argument("--ckpt", help="Checkpoint to control one seat")
    parser.add_argument("--seat", type=int, default=0, help="Seat for --ckpt policy")
    parser.add_argument("--profile", default="simplified_v1")
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--every", type=int, default=1,
                        help="Print a board summary every N turns")
    args = parser.parse_args()

    names = [b.strip() for b in args.bots.split(",")]
    if len(names) != 4:
        parser.error("--bots needs exactly 4 entries")
    actors = []
    for n in names:
        if n not in BOTS:
            parser.error(f"unknown bot {n!r}")
        actors.append((n, lambda s, r, f=BOTS[n]: f(s, r).catalog_index))

    if args.ckpt:
        from catan_rl.rl.checkpointing import load_policy
        from catan_rl.rl.evaluate import policy_action

        policy, meta = load_policy(args.ckpt)
        actors[args.seat] = (
            f"policy@it{meta['iteration']}",
            lambda s, r: policy_action(policy, s),
        )

    print("Seats: " + "  ".join(f"P{i}={name}" for i, (name, _) in enumerate(actors)))

    rng = random.Random(args.seed)
    config = BoardConfig.standard(seed=args.seed)
    state = GameState.new_game(config, n_players=4, seed=args.seed,
                               profile=RulesProfile.get(args.profile))

    last_turn = -1
    plies = 0
    while not state.is_terminal and state.turn_number < args.max_turns:
        idx = actors[state.current_player][1](state, rng)
        action = CATALOG[idx]
        print(f"  t={state.turn_number:4d} P{state.current_player} "
              f"{state.phase.name:20s} {action}")
        apply_action(state, action, rng)
        plies += 1
        if state.turn_number != last_turn and state.turn_number % args.every == 0:
            last_turn = state.turn_number
            vps = [compute_vp(p, state) for p in range(4)]
            print(f"--- turn {state.turn_number}: VP={vps} "
                  f"robber=hex{state.robber_hex} ---")

    vps = [compute_vp(p, state) for p in range(4)]
    print(f"\n=== GAME OVER after {state.turn_number} turns ({plies} plies) ===")
    print(f"VP: {vps}")
    if state.winner is not None:
        print(f"Winner: P{state.winner} ({actors[state.winner][0]})")
    else:
        print("No winner (turn limit reached)")


if __name__ == "__main__":
    main()
