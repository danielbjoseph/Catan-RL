import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

import ray
from catan_rl.rl.logger import StructuredLogger


@ray.remote
def remote_function():
    import os
    return os.getpid()


def main():
    logger = StructuredLogger(run_id="phase-0-ray", log_dir="runs/phase-0-ray/logs")

    main_pid = os.getpid()
    logger.log_event("ray_test_started", main_pid=main_pid)

    ray.init(address="127.0.0.1:6379")
    logger.log_event("ray_connected", address="127.0.0.1:6379")

    resources = ray.cluster_resources()
    logger.log_event(
        "ray_cluster_info",
        total_cpus=resources.get("CPU", 0),
        num_nodes=len(ray.nodes()),
    )

    remote_pid = ray.get(remote_function.remote())
    logger.log_event("remote_function_executed", remote_pid=remote_pid, different_from_main=remote_pid != main_pid)

    print(f"Main process PID: {main_pid}")
    print("✓ Connected to Ray cluster at 127.0.0.1:6379")
    print(f"Remote function executed on PID: {remote_pid}")

    if remote_pid != main_pid:
        print("✓ Remote function ran on a different process (proof of distribution)")
    else:
        print("✗ Remote function ran on main process (not distributed)")
        logger.log_event("ray_test_failed", reason="remote_function_not_distributed")
        sys.exit(1)

    print(f"✓ Cluster resources: {resources}")
    num_nodes = len(ray.nodes())
    print(f"✓ Cluster has {num_nodes} nodes")

    logger.log_event("ray_test_passed", num_nodes=num_nodes)
    print("\n✓ Phase 0 Ray smoke test PASSED")

    ray.shutdown()


if __name__ == "__main__":
    main()
