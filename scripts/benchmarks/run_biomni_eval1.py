#!/usr/bin/env python3
"""CLI runner for Biomni-Eval1 benchmark evaluation.

Usage:
    python scripts/benchmarks/run_biomni_eval1.py --tasks all --limit 50
    python scripts/benchmarks/run_biomni_eval1.py --tasks gwas,drug_repurposing --limit 20
    python scripts/benchmarks/run_biomni_eval1.py --resume biomni_20260314_120000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ALL_TASKS = [
    "crispr_delivery",
    "gwas_causal_gene_gwas_catalog",
    "gwas_causal_gene_opentargets",
    "gwas_causal_gene_pharmaprojects",
    "gwas_variant_prioritization",
    "lab_bench_dbqa",
    "lab_bench_seqqa",
    "patient_gene_detection",
    "rare_disease_diagnosis",
    "screen_gene_retrieval",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Biomni-Eval1 evaluation for YOHAS 3.0")
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help=f"Comma-separated task names or 'all' (default: all). Available: {', '.join(ALL_TASKS)}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max instances per task (default: all)",
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

    from biomni_eval1_adapter import run_biomni_eval1

    tasks = None if args.tasks == "all" else args.tasks.split(",")

    results_path = run_biomni_eval1(
        tasks=tasks,
        limit=args.limit,
        mode=args.mode,
        max_iterations=args.max_iterations,
        checkpoint_id=args.resume,
    )
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
