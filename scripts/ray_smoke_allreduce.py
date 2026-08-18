import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent))

from catan_rl.rl.logger import StructuredLogger


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    logger = StructuredLogger(run_id="phase-0-gloo", log_dir="runs/phase-0-gloo/logs")

    logger.log_event(
        "gloo_test_started",
        rank=rank,
        world_size=world_size,
    )

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:29500",
    )

    tensor = torch.tensor([float(rank + 1), float(rank + 2), float(rank + 3)])
    logger.log_event("tensor_before_allreduce", rank=rank, values=tensor.tolist())

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    logger.log_event("tensor_after_allreduce", rank=rank, values=tensor.tolist())

    expected = torch.tensor([2.0, 4.0, 6.0])
    if torch.allclose(tensor, expected):
        logger.log_event("gloo_test_passed", rank=rank)
        print(f"Rank {rank}: tensor before allreduce=[1.0, 2.0, 3.0]")
        print(f"Rank {rank}: allreduce result={tensor.tolist()}")
        print(f"Rank {rank}: ✓ all_reduce produced correct result")
    else:
        logger.log_event("gloo_test_failed", rank=rank, expected=expected.tolist(), got=tensor.tolist())
        print(f"Rank {rank}: ✗ all_reduce FAILED")
        sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
