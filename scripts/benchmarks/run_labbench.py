#!/usr/bin/env python3
"""CLI runner for LAB-Bench benchmark evaluation.

Usage:
    python scripts/benchmarks/run_labbench.py --categories DbQA,SeqQA,LitQA2 --limit 100
    python scripts/benchmarks/run_labbench.py --categories all --limit 50
    python scripts/benchmarks/run_labbench.py --resume labbench_20260314_120000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PRIORITY_CATEGORIES = ["DbQA", "SeqQA", "LitQA2"]
ALL_CATEGORIES = [
    "LitQA2", "DbQA", "SuppQA", "FigQA",
    "TableQA", "ProtocolQA", "SeqQA", "CloningScenarios",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LAB-Bench evaluation for YOHAS 3.0")
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(PRIORITY_CATEGORIES),
        help=(
            f"Comma-separated categories or 'all' (default: {','.join(PRIORITY_CATEGORIES)}). "
            f"Available: {', '.join(ALL_CATEGORIES)}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max questions per category (default: all)",
    )
    parser.add_argument(
        "--mode",
        choices=["zero-shot", "agentic"],
        default="agentic",
        help="Evaluation mode (default: agentic)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="MCTS iterations per question in agentic mode (default: 3)",
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

    from labbench_adapter import run_labbench

    categories = ALL_CATEGORIES if args.categories == "all" else args.categories.split(",")

    results_path = run_labbench(
        categories=categories,
        limit=args.limit,
        mode=args.mode,
        max_iterations=args.max_iterations,
        checkpoint_id=args.resume,
    )
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
