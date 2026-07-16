"""
Serve the trace-browsing dashboard.

Usage:
  python scripts/dashboard.py --port 8050 --runs-dir runs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from catan_rl.dashboard.app import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run subdirectories")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(args.runs_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
