#!/usr/bin/env python3
"""CLI runner for BixBench benchmark evaluation.

Usage:
    python scripts/benchmarks/run_bixbench.py --mode zero-shot --limit 10
    python scripts/benchmarks/run_bixbench.py --mode agentic --limit 205
    python scripts/benchmarks/run_bixbench.py --resume bixbench_20260314_120000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root or scripts/benchmarks/
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BixBench evaluation for YOHAS 3.0")
    parser.add_argument(
        "--mode",
        choices=["zero-shot", "agentic"],
        default="agentic",
        help="Evaluation mode (default: agentic)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max instances to evaluate (default: all 205)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="MCTS iterations per instance in agentic mode (default: 5)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a previous run checkpoint ID",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from bixbench_adapter import run_bixbench

    results_path = run_bixbench(
        mode=args.mode,
        limit=args.limit,
        max_iterations=args.max_iterations,
        checkpoint_id=args.resume,
    )
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
