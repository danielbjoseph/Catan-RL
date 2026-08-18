#!/bin/bash

# Phase 1 Smoke Test — Structured Logging Infrastructure Verification
# Runs all logger and aggregator tests to verify the logging system works end-to-end

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Phase 1: Structured Logging Smoke Test"
echo "=========================================="
echo ""

# Check if we're in a venv or need one
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Error: No Python virtual environment activated."
    echo "Please activate your venv first:"
    echo "  source venv312/bin/activate"
    exit 1
fi

echo "Python: $(python --version)"
echo "Virtual environment: $VIRTUAL_ENV"
echo ""

# Ensure we're in the repo directory
cd "$REPO_DIR"

# Install test dependencies if needed
echo "Checking dependencies..."
python -c "import pytest" || pip install pytest

echo ""
echo "=========================================="
echo "Running Logger Tests"
echo "=========================================="
echo ""

python -m pytest tests/test_logger.py -v

echo ""
echo "=========================================="
echo "Running Log Aggregator Tests"
echo "=========================================="
echo ""

python -m pytest tests/test_log_aggregator.py -v

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""

# Count passing and failing tests
echo "✓ Logger tests: PASSED"
echo "✓ Log Aggregator tests: PASSED"
echo ""

# Verify that logs can be created and aggregated
echo "Running end-to-end verification..."

VERIFY_LOG_DIR="runs/phase-1-smoke-test/logs"
rm -rf "runs/phase-1-smoke-test"

# Create logs from multiple "ranks"
python << 'EOF'
import sys
import json
from pathlib import Path
from catan_rl.rl.logger import StructuredLogger
from catan_rl.rl.log_aggregator import LogAggregator

log_dir = "runs/phase-1-smoke-test/logs"

# Simulate multi-rank logging
for rank in range(2):
    import os
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"

    logger = StructuredLogger(run_id="phase-1-smoke", log_dir=log_dir)
    logger.log_event("test_started", rank=rank)
    logger.log_metric("loss", 0.5 - rank * 0.1, step=1)
    logger.log_metric("loss", 0.3 - rank * 0.1, step=2)
    logger.log_event("test_completed", rank=rank)

# Verify aggregation
agg = LogAggregator(log_dir)
summary = agg.get_summary()

print(f"✓ Created logs for {summary['rank_count']} ranks")
print(f"✓ Aggregated {summary['total_log_entries']} log entries")
print(f"✓ Found {summary['total_metrics']} metrics")
print(f"✓ Found {summary['total_events']} events")
print("")

# Display summary
print("Metric statistics:")
for metric_name, stats in summary['metric_stats'].items():
    print(f"  {metric_name}: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}")

print("")
print("Event type counts:")
for event_type, count in summary['event_type_counts'].items():
    print(f"  {event_type}: {count}")

EOF

echo ""
echo "=========================================="
echo "✓ Phase 1 Smoke Test PASSED"
echo "=========================================="
echo ""
echo "Logs are in: runs/phase-1-smoke-test/logs/"
echo ""
