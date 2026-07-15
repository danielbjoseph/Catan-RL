"""
Evaluate saved checkpoints against scripted bots and each other.

Usage:
  python scripts/evaluate_checkpoints.py --run runs/ppo_baseline
  python scripts/evaluate_checkpoints.py --run runs/ppo_baseline --games 50 --vs random,greedy,heuristic,prev
  python scripts/evaluate_checkpoints.py --ckpt runs/ppo_baseline/checkpoints/ckpt_000100.pt --vs greedy
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from catan_rl.bots import greedy_bot, heuristic_bot, random_bot
from catan_rl.rl.checkpointing import list_checkpoints, load_policy
from catan_rl.rl.evaluate import (
    evaluate_policy_vs_policy,
    evaluate_vs_bots,
    evaluate_vs_checkpoint,
)

BOTS = {
    "random": random_bot.pick_action,
    "greedy": greedy_bot.pick_action,
    "heuristic": heuristic_bot.pick_action,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Run directory (evaluates every checkpoint in it)")
    parser.add_argument("--ckpt", help="Single checkpoint path")
    parser.add_argument("--vs-ckpt", help="Second checkpoint to play --ckpt against "
                        "(cross-mode policy-vs-policy evaluation; requires --ckpt)")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--vs", default="random,greedy",
                        help="Comma list of: random,greedy,heuristic,prev")
    parser.add_argument("--profile", default="simplified_v1")
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.vs_ckpt and not args.ckpt:
        parser.error("--vs-ckpt requires --ckpt")

    if args.ckpt:
        ckpts = [Path(args.ckpt)]
    elif args.run:
        ckpts = list_checkpoints(Path(args.run) / "checkpoints")
        if not ckpts:
            ckpts = list_checkpoints(args.run)  # allow pointing at the ckpt dir itself
    else:
        parser.error("provide --run or --ckpt")
    if not ckpts:
        print("no checkpoints found")
        sys.exit(1)

    if args.vs_ckpt:
        policy_a, meta_a = load_policy(args.ckpt)
        policy_b, meta_b = load_policy(args.vs_ckpt)
        mode_a = meta_a.get("obs_mode", "self_play")
        mode_b = meta_b.get("obs_mode", "self_play")
        result = evaluate_policy_vs_policy(
            policy_a, mode_a, policy_b, mode_b, args.games,
            rules_profile=args.profile, seed=args.seed, max_turns=args.max_turns,
        )
        header = ["ckpt_a", "mode_a", "ckpt_b", "mode_b", "win_rate_a", "win_rate_b", "draws"]
        row = [
            Path(args.ckpt).name, mode_a, Path(args.vs_ckpt).name, mode_b,
            f"{result['win_rate_a']:.2f}", f"{result['win_rate_b']:.2f}",
            str(result["draws"]),
        ]
        print("  ".join(f"{h:>18s}" for h in header))
        print("  ".join(f"{c:>18s}" for c in row))
        return

    opponents = [o.strip() for o in args.vs.split(",") if o.strip()]
    kwargs = dict(rules_profile=args.profile, seed=args.seed, max_turns=args.max_turns)

    header = ["checkpoint"] + [f"vs_{o}" for o in opponents]
    print("  ".join(f"{h:>18s}" for h in header))

    prev_path = None
    for ckpt in ckpts:
        policy, meta = load_policy(ckpt)
        row = [ckpt.name]
        for opp in opponents:
            if opp == "prev":
                if prev_path is None:
                    row.append("-")
                    continue
                r = evaluate_vs_checkpoint(policy, prev_path, args.games, **kwargs)
                row.append(f"{r['win_rate']:.2f}")
            elif opp in BOTS:
                r = evaluate_vs_bots(policy, BOTS[opp], args.games, **kwargs)
                row.append(f"{r['win_rate']:.2f} (vp={r['mean_vp']:.1f})")
            else:
                parser.error(f"unknown opponent {opp!r}")
        print("  ".join(f"{c:>18s}" for c in row))
        prev_path = ckpt


if __name__ == "__main__":
    main()
