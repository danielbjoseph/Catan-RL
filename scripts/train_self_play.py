"""
Train the shared self-play PPO policy.

Usage:
  python scripts/train_self_play.py --config configs/ppo_baseline.yaml
  python scripts/train_self_play.py --config configs/ppo_baseline.yaml --iterations 100
  python scripts/train_self_play.py --config configs/ppo_baseline.yaml --resume
  python scripts/train_self_play.py --config configs/ppo_baseline.yaml --num-workers 4

Monitor with:  tensorboard --logdir runs/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from catan_rl.rl.self_play import SelfPlayTrainer, _load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo_baseline.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override the config's iteration count")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of workers for parallel collection")
    parser.add_argument("--run-dir", default=None,
                        help="Override runs/<experiment_name>")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in the run dir")
    parser.add_argument("--init-from", default=None,
                        help="Warm-start from a checkpoint at PATH, widening it "
                             "to the current config's obs_dim/action-catalog size "
                             "if smaller (optimizer starts fresh)")
    parser.add_argument("--trace", type=int, default=None,
                        help="Trace every Nth game of each iteration to "
                             "runs/<run_name>/traces/iter<k>_game<g>.json (default: off)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers

    trainer = SelfPlayTrainer(
        cfg, run_dir=args.run_dir, device=args.device, resume=args.resume,
        init_from=args.init_from,
    )
    if args.trace is not None:
        trainer.trace_every = args.trace
    print(f"config: {args.config}  run_dir: {trainer.run_dir}  device: {trainer.device}  num_workers: {trainer.cfg.get('num_workers')}")
    try:
        trainer.train(iterations=args.iterations)
    except KeyboardInterrupt:
        print("\ninterrupted — saving final checkpoint")
        trainer._checkpoint({})
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
