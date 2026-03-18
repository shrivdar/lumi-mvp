#!/usr/bin/env python3
"""Batch research query runner — runs 20+ diverse biomedical queries through YOHAS.

Collects real trajectories via TrajectoryCollector with compute_reward wired,
saves results, KG snapshots, and strategy templates per query.

Usage:
    python scripts/run_diverse_queries.py                    # dry-run (no LLM)
    python scripts/run_diverse_queries.py --live             # live LLM calls
    python scripts/run_diverse_queries.py --resume           # skip completed queries
    python scripts/run_diverse_queries.py --limit 5          # run first 5 only
    python scripts/run_diverse_queries.py --max-trials 3     # multi-trial protocol
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Ensure backend is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diverse biomedical research queries
# ---------------------------------------------------------------------------

DIVERSE_QUERIES: list[dict[str, str]] = [
    {
        "id": "egfr_tki_resistance",
        "query": "What are the primary mechanisms of EGFR TKI resistance in non-small cell lung cancer, and what therapeutic strategies are being developed to overcome them?",
        "category": "drug_resistance",
    },
    {
        "id": "kras_g12c",
        "query": "What are the current strategies for targeting KRAS G12C mutations, and what combination approaches show promise for overcoming adaptive resistance?",
        "category": "target_discovery",
    },
    {
        "id": "microbiome_pd1",
        "query": "How does the gut microbiome influence PD-1/PD-L1 checkpoint immunotherapy response, and which bacterial taxa are associated with improved outcomes?",
        "category": "microbiome",
    },
    {
        "id": "pcsk9_beyond_ldl",
        "query": "What are the emerging roles of PCSK9 inhibition beyond LDL cholesterol reduction, including effects on inflammation, sepsis, and viral infection?",
        "category": "target_discovery",
    },
    {
        "id": "glp1_neurodegeneration",
        "query": "What is the evidence for GLP-1 receptor agonists as neuroprotective agents in Alzheimer's and Parkinson's disease, and what mechanisms underlie their effects?",
        "category": "drug_design",
    },
    {
        "id": "cd47_sirpa",
        "query": "What is the current understanding of the CD47-SIRPα checkpoint axis in cancer, and how do emerging therapies targeting this pathway compare to PD-1/PD-L1 inhibitors?",
        "category": "immunotherapy",
    },
    {
        "id": "crispr_base_editing",
        "query": "What are the current approaches to minimizing off-target effects in CRISPR base editing, and how do adenine and cytosine base editors compare in safety profiles?",
        "category": "gene_editing",
    },
    {
        "id": "bispecific_vs_cart",
        "query": "How do bispecific antibodies compare to CAR-T cell therapy in hematological malignancies in terms of efficacy, safety, and accessibility?",
        "category": "immunotherapy",
    },
    {
        "id": "tau_propagation",
        "query": "What are the molecular mechanisms of tau protein propagation in neurodegenerative tauopathies, and what therapeutic targets have been identified to block prion-like spreading?",
        "category": "protein_science",
    },
    {
        "id": "parp_resistance",
        "query": "What mechanisms drive PARP inhibitor resistance in BRCA-mutated cancers, and what combination strategies are being explored to restore sensitivity?",
        "category": "drug_resistance",
    },
    {
        "id": "il17a_autoimmune",
        "query": "What is the role of IL-17A in autoimmune diseases beyond psoriasis, and how do IL-17 pathway inhibitors compare across indications?",
        "category": "pathway_analysis",
    },
    {
        "id": "ferroptosis_cancer",
        "query": "How can ferroptosis be therapeutically exploited in cancer treatment, and what are the key regulators and biomarkers of ferroptotic cell death?",
        "category": "cancer_biology",
    },
    {
        "id": "adc_payload",
        "query": "What are the key considerations in antibody-drug conjugate payload optimization, and how do different linker-payload technologies affect therapeutic index?",
        "category": "drug_design",
    },
    {
        "id": "tigit_checkpoint",
        "query": "What is the current understanding of TIGIT checkpoint biology, and why have clinical trials of TIGIT-targeting antibodies shown mixed results?",
        "category": "immunotherapy",
    },
    {
        "id": "synthetic_lethality",
        "query": "How are synthetic lethality screening approaches being used to identify new cancer therapeutic targets, and what computational methods enhance hit discovery?",
        "category": "cancer_biology",
    },
    {
        "id": "rnai_delivery",
        "query": "What are the major delivery challenges for RNAi therapeutics, and how do lipid nanoparticles, GalNAc conjugates, and exosomes compare as delivery platforms?",
        "category": "gene_editing",
    },
    {
        "id": "protac_design",
        "query": "What are the key principles of PROTAC degrader design, and how do factors like linker length, E3 ligase choice, and target engagement affect degradation efficiency?",
        "category": "drug_design",
    },
    {
        "id": "epigenetic_aml",
        "query": "What is the current landscape of epigenetic therapy in acute myeloid leukemia, and how do IDH inhibitors, DOT1L inhibitors, and BET inhibitors compare?",
        "category": "cancer_biology",
    },
    {
        "id": "antibody_sirna",
        "query": "What are the emerging approaches for antibody-siRNA conjugates, and how do they overcome the delivery limitations of naked siRNA?",
        "category": "drug_design",
    },
    {
        "id": "sting_innate_immunity",
        "query": "How do STING agonists activate innate immune responses against cancer, and what are the challenges in translating preclinical efficacy to clinical benefit?",
        "category": "immunotherapy",
    },
]

# ---------------------------------------------------------------------------
# Results tracking
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "diverse_queries"
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.json"


def load_checkpoint() -> dict[str, dict]:
    """Load checkpoint of completed queries."""
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {}


def save_checkpoint(checkpoint: dict[str, dict]) -> None:
    """Save checkpoint after each query completes."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2, default=str))


# ---------------------------------------------------------------------------
# Query runner
# ---------------------------------------------------------------------------

async def run_query_dry(
    query_info: dict[str, str],
    *,
    max_trials: int = 1,
) -> dict:
    """Run a single query in dry-run mode (no LLM, simulated)."""
    from benchmarks.evaluator import BenchmarkEvaluator
    from benchmarks.models import BenchmarkInstance, BenchmarkSuite, RunMode
    from benchmarks.strategy_memory import StrategyMemory

    strategy_memory = StrategyMemory(
        storage_dir=RESULTS_DIR / "strategies",
    )

    instance = BenchmarkInstance(
        suite=BenchmarkSuite.BIOMNI_EVAL1,
        instance_id=query_info["id"],
        question=query_info["query"],
        ground_truth="",  # open-ended research query, no single ground truth
        category=query_info["category"],
    )

    evaluator = BenchmarkEvaluator(
        mode=RunMode.YOHAS_FULL,
        max_trials=max_trials,
        strategy_memory=strategy_memory,
        collect_trajectories=True,
    )

    start = time.monotonic()
    result = await evaluator.evaluate_instance(instance)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Extract strategy template
    strategy_memory.extract_template(
        name=query_info["id"],
        query=query_info["query"],
        result=result,
        score=result.score,
        tools_used=result.tools_used,
    )

    return {
        "query_id": query_info["id"],
        "category": query_info["category"],
        "predicted": result.predicted,
        "score": result.score,
        "tokens_used": result.tokens_used,
        "latency_ms": elapsed_ms,
        "turns": result.turns,
        "tools_used": result.tools_used,
        "trial_count": len(result.trial_results),
        "best_trial": result.best_trial,
        "status": result.status.value,
        "completed_at": datetime.now(UTC).isoformat(),
        "trajectories_collected": len(evaluator.trajectories),
    }


async def run_query_live(
    query_info: dict[str, str],
    *,
    max_trials: int = 1,
) -> dict:
    """Run a single query with live LLM calls through the full orchestrator."""
    from benchmarks.evaluator import BenchmarkEvaluator
    from benchmarks.models import BenchmarkInstance, BenchmarkSuite, RunMode
    from benchmarks.strategy_memory import StrategyMemory
    from core.llm import LLMClient

    strategy_memory = StrategyMemory(
        storage_dir=RESULTS_DIR / "strategies",
    )

    llm = LLMClient()

    instance = BenchmarkInstance(
        suite=BenchmarkSuite.BIOMNI_EVAL1,
        instance_id=query_info["id"],
        question=query_info["query"],
        ground_truth="",
        category=query_info["category"],
    )

    evaluator = BenchmarkEvaluator(
        mode=RunMode.YOHAS_FULL,
        llm=llm,
        max_trials=max_trials,
        strategy_memory=strategy_memory,
        collect_trajectories=True,
    )

    start = time.monotonic()
    result = await evaluator.evaluate_instance(instance)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Extract strategy template
    strategy_memory.extract_template(
        name=query_info["id"],
        query=query_info["query"],
        result=result,
        score=result.score,
        tools_used=result.tools_used,
    )

    # Save per-query result
    query_dir = RESULTS_DIR / query_info["id"]
    query_dir.mkdir(parents=True, exist_ok=True)
    result_path = query_dir / "result.json"
    result_path.write_text(result.model_dump_json(indent=2))

    return {
        "query_id": query_info["id"],
        "category": query_info["category"],
        "predicted": result.predicted,
        "score": result.score,
        "tokens_used": result.tokens_used,
        "latency_ms": elapsed_ms,
        "turns": result.turns,
        "tools_used": result.tools_used,
        "trial_count": len(result.trial_results),
        "best_trial": result.best_trial,
        "status": result.status.value,
        "completed_at": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOHAS Diverse Query Runner")
    p.add_argument("--live", action="store_true", help="Use live LLM calls")
    p.add_argument("--resume", action="store_true", help="Skip already-completed queries")
    p.add_argument("--limit", type=int, default=None, help="Limit number of queries to run")
    p.add_argument("--max-trials", type=int, default=1, help="Multi-trial count per query")
    p.add_argument("--query-id", type=str, default=None, help="Run a specific query by ID")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    import random
    random.seed(args.seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Filter queries
    queries = DIVERSE_QUERIES
    if args.query_id:
        queries = [q for q in queries if q["id"] == args.query_id]
        if not queries:
            logger.error("Query ID %s not found", args.query_id)
            return

    if args.limit:
        queries = queries[:args.limit]

    # Load checkpoint for resume
    checkpoint = load_checkpoint() if args.resume else {}

    runner = run_query_live if args.live else run_query_dry

    logger.info(
        "Starting batch run: %d queries, live=%s, max_trials=%d, resume=%s",
        len(queries),
        args.live,
        args.max_trials,
        args.resume,
    )

    completed = 0
    failed = 0

    for i, query_info in enumerate(queries, start=1):
        qid = query_info["id"]

        if args.resume and qid in checkpoint:
            logger.info("[%d/%d] Skipping %s (already completed)", i, len(queries), qid)
            completed += 1
            continue

        logger.info("[%d/%d] Running %s: %s", i, len(queries), qid, query_info["query"][:80])

        try:
            result = await runner(query_info, max_trials=args.max_trials)

            checkpoint[qid] = result
            save_checkpoint(checkpoint)

            status_str = "PASS" if result["score"] > 0 else "DONE"
            logger.info(
                "[%d/%d] %s %s — score=%.2f, tokens=%d, trials=%d, latency=%dms",
                i,
                len(queries),
                status_str,
                qid,
                result["score"],
                result["tokens_used"],
                result["trial_count"],
                result["latency_ms"],
            )
            completed += 1

        except Exception as exc:
            logger.error("[%d/%d] FAILED %s: %s", i, len(queries), qid, exc)
            checkpoint[qid] = {
                "query_id": qid,
                "status": "failed",
                "error": str(exc),
                "completed_at": datetime.now(UTC).isoformat(),
            }
            save_checkpoint(checkpoint)
            failed += 1

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH RUN COMPLETE")
    print("=" * 60)
    print(f"  Total queries: {len(queries)}")
    print(f"  Completed:     {completed}")
    print(f"  Failed:        {failed}")
    print(f"  Results dir:   {RESULTS_DIR}")
    print("=" * 60)

    # Category breakdown
    by_category: dict[str, list[float]] = {}
    for qid, data in checkpoint.items():
        if isinstance(data, dict) and "category" in data:
            cat = data["category"]
            by_category.setdefault(cat, []).append(data.get("score", 0.0))

    if by_category:
        print("\nCategory breakdown:")
        for cat, scores in sorted(by_category.items()):
            avg = sum(scores) / len(scores)
            print(f"  {cat:25s}  avg_score={avg:.2f}  n={len(scores)}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
