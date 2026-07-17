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

_DEFAULT_BELIEF_BLEND = 0.25
_DEFAULT_BELIEF_NOISE = 0.5


def eval_kwargs_from_meta(meta: dict, seed: int = 0) -> dict:
    """Derive the obs_mode/noise_cfg kwargs to evaluate a loaded checkpoint
    with, from that checkpoint's own saved metadata.

    A checkpoint trained in a non-self_play obs_mode has a different
    observation dimensionality baked into its policy weights, so evaluation
    must dispatch on the checkpoint's own stored ``obs_mode`` rather than
    silently defaulting to self_play (which crashes with a shape mismatch).
    For realistic mode, belief_blend/belief_noise are taken from the
    checkpoint's own stored training config when present, else the standard
    0.25/0.5 defaults.
    """
    obs_mode = meta.get("obs_mode", "self_play")
    noise_cfg = None
    if obs_mode == "realistic":
        cfg = meta.get("config") or {}
        noise_cfg = {
            "belief_blend": float(cfg.get("belief_blend", _DEFAULT_BELIEF_BLEND)),
            "belief_noise": float(cfg.get("belief_noise", _DEFAULT_BELIEF_NOISE)),
            "seed": seed,
        }
    return {"obs_mode": obs_mode, "noise_cfg": noise_cfg}


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
    parser.add_argument("--trace", type=int, default=None, metavar="N",
                        help="Trace every Nth eval game (--trace 1 traces every "
                             "game). Off by default. Written under --trace-dir.")
    parser.add_argument("--trace-dir", default=None,
                        help="Directory for traces (default: <run>/traces, or "
                             "<ckpt's parent>/traces when using --ckpt alone). "
                             "Each checkpoint/opponent pair gets its own subdir.")
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

    if args.trace_dir:
        base_trace_dir = Path(args.trace_dir)
    elif args.run:
        base_trace_dir = Path(args.run) / "traces"
    else:
        base_trace_dir = Path(args.ckpt).resolve().parent / "traces"

    if args.vs_ckpt:
        policy_a, meta_a = load_policy(args.ckpt)
        policy_b, meta_b = load_policy(args.vs_ckpt)
        kwargs_a = eval_kwargs_from_meta(meta_a, seed=args.seed)
        kwargs_b = eval_kwargs_from_meta(meta_b, seed=args.seed)
        mode_a = kwargs_a["obs_mode"]
        mode_b = kwargs_b["obs_mode"]
        trace_dir = base_trace_dir / f"{Path(args.ckpt).stem}_vs_{Path(args.vs_ckpt).stem}"
        result = evaluate_policy_vs_policy(
            policy_a, mode_a, policy_b, mode_b, args.games,
            rules_profile=args.profile, seed=args.seed, max_turns=args.max_turns,
            noise_cfg_a=kwargs_a["noise_cfg"], noise_cfg_b=kwargs_b["noise_cfg"],
            trace_dir=trace_dir, trace_every=args.trace,
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
        ckpt_kwargs = eval_kwargs_from_meta(meta, seed=args.seed)
        row = [ckpt.name]
        for opp in opponents:
            trace_dir = base_trace_dir / ckpt.stem / f"vs_{opp}"
            if opp == "prev":
                if prev_path is None:
                    row.append("-")
                    continue
                r = evaluate_vs_checkpoint(
                    policy, prev_path, args.games,
                    obs_mode=ckpt_kwargs["obs_mode"], noise_cfg=ckpt_kwargs["noise_cfg"],
                    trace_dir=trace_dir, trace_every=args.trace, **kwargs,
                )
                row.append(f"{r['win_rate']:.2f}")
            elif opp in BOTS:
                r = evaluate_vs_bots(
                    policy, BOTS[opp], args.games,
                    obs_mode=ckpt_kwargs["obs_mode"], noise_cfg=ckpt_kwargs["noise_cfg"],
                    trace_dir=trace_dir, trace_every=args.trace, **kwargs,
                )
                row.append(f"{r['win_rate']:.2f} (vp={r['mean_vp']:.1f})")
            else:
                parser.error(f"unknown opponent {opp!r}")
        print("  ".join(f"{c:>18s}" for c in row))
        prev_path = ckpt


if __name__ == "__main__":
    main()
