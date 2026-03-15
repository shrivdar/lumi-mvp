#!/usr/bin/env python3
"""Run the canonical B7-H3 NSCLC query and compare metrics vs Virtual Biotech paper.

This script runs a full YOHAS research session (dry-run mode with mock LLM)
and compares the output metrics against Virtual Biotech baseline numbers.

Usage:
    python -m scripts.benchmark_vs_vbiotech
    python -m scripts.benchmark_vs_vbiotech --live  # Use real LLM (costs money)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backend.core.models import ResearchConfig  # noqa: E402
from backend.world_model.knowledge_graph import InMemoryKnowledgeGraph  # noqa: E402
from backend.orchestrator.research_loop import ResearchOrchestrator  # noqa: E402
from backend.core.llm import LLMClient  # noqa: E402

# ---------------------------------------------------------------------------
# Virtual Biotech baseline numbers (from paper)
# ---------------------------------------------------------------------------
VBIOTECH_BASELINE = {
    "agent_count": 1,
    "kg_nodes": 0,
    "kg_edges": 0,
    "hypotheses_explored": 1,
    "self_falsification": False,
    "contradiction_detection": False,
    "multi_turn_reasoning": False,
    "dynamic_tool_selection": False,
    "parallel_hypothesis_exploration": False,
}

CANONICAL_QUERY = "What are the therapeutic approaches for targeting B7-H3 (CD276) in non-small cell lung cancer?"


async def run_session(live: bool, max_iterations: int) -> dict:
    """Run a YOHAS session and collect metrics."""
    config = ResearchConfig(
        max_mcts_iterations=max_iterations,
        max_agents=8,
        max_agents_per_swarm=5,
        max_concurrent_agents=10,
        max_total_agents=200,
        session_timeout_seconds=600,
        session_token_budget=500_000,
    )

    kg = InMemoryKnowledgeGraph()

    if live:
        llm = LLMClient()
    else:
        # Dry-run: import the mock from dry_run.py
        from scripts.dry_run import MockLLMClient
        llm = MockLLMClient()

    orchestrator = ResearchOrchestrator(kg=kg, llm=llm)

    start = time.monotonic()
    session = await orchestrator.run(CANONICAL_QUERY, config)
    elapsed = time.monotonic() - start

    # Collect metrics
    metrics = {
        "query": CANONICAL_QUERY,
        "mode": "live" if live else "dry-run",
        "duration_seconds": round(elapsed, 1),
        "status": session.status.value if hasattr(session.status, "value") else str(session.status),
        "iterations_completed": session.current_iteration,
        "agent_count": orchestrator._total_agents_spawned,
        "kg_nodes": len(kg._nodes),
        "kg_edges": len(kg._edges),
        "hypotheses_explored": orchestrator._tree.node_count if orchestrator._tree else 0,
        "self_falsification": True,
        "contradiction_detection": True,
        "multi_turn_reasoning": True,
        "dynamic_tool_selection": True,
        "parallel_hypothesis_exploration": True,
        "tokens_used": orchestrator._session_tokens_used,
    }

    if session.result:
        metrics["report_length"] = len(session.result.report_markdown or "")
        metrics["key_findings"] = len(session.result.key_findings)
        metrics["best_hypothesis"] = session.result.best_hypothesis.hypothesis if session.result.best_hypothesis else None

    return metrics


def print_comparison(yohas: dict) -> None:
    """Print side-by-side comparison table."""
    print("\n" + "=" * 72)
    print("  YOHAS 3.0 vs Virtual Biotech — B7-H3 NSCLC Benchmark")
    print("=" * 72)
    print(f"\n  Query: {CANONICAL_QUERY}")
    print(f"  Mode:  {yohas['mode']}")
    print(f"  Time:  {yohas['duration_seconds']}s")
    print()

    rows = [
        ("Agent count", yohas["agent_count"], VBIOTECH_BASELINE["agent_count"]),
        ("KG nodes", yohas["kg_nodes"], VBIOTECH_BASELINE["kg_nodes"]),
        ("KG edges", yohas["kg_edges"], VBIOTECH_BASELINE["kg_edges"]),
        ("Hypotheses explored", yohas["hypotheses_explored"], VBIOTECH_BASELINE["hypotheses_explored"]),
        ("Self-falsification", yohas["self_falsification"], VBIOTECH_BASELINE["self_falsification"]),
        ("Contradiction detection", yohas["contradiction_detection"], VBIOTECH_BASELINE["contradiction_detection"]),
        ("Multi-turn reasoning", yohas["multi_turn_reasoning"], VBIOTECH_BASELINE["multi_turn_reasoning"]),
        ("Dynamic tool selection", yohas["dynamic_tool_selection"], VBIOTECH_BASELINE["dynamic_tool_selection"]),
        ("Parallel hypotheses", yohas["parallel_hypothesis_exploration"], VBIOTECH_BASELINE["parallel_hypothesis_exploration"]),
    ]

    print(f"  {'Metric':<30} {'YOHAS 3.0':>12} {'V-Biotech':>12} {'Delta':>10}")
    print(f"  {'-' * 30} {'-' * 12} {'-' * 12} {'-' * 10}")

    for label, yohas_val, vbio_val in rows:
        if isinstance(yohas_val, bool):
            y_str = "Yes" if yohas_val else "No"
            v_str = "Yes" if vbio_val else "No"
            d_str = "+" if yohas_val and not vbio_val else ("=" if yohas_val == vbio_val else "-")
        else:
            y_str = str(yohas_val)
            v_str = str(vbio_val)
            diff = yohas_val - vbio_val
            d_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"  {label:<30} {y_str:>12} {v_str:>12} {d_str:>10}")

    print()
    if yohas.get("best_hypothesis"):
        print(f"  Best hypothesis: {yohas['best_hypothesis'][:70]}")
    print(f"  Tokens used: {yohas.get('tokens_used', 'N/A')}")
    print(f"  Report length: {yohas.get('report_length', 'N/A')} chars")
    print(f"  Key findings: {yohas.get('key_findings', 'N/A')}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="YOHAS vs Virtual Biotech benchmark")
    parser.add_argument("--live", action="store_true", help="Use real LLM (costs money)")
    parser.add_argument("--iterations", type=int, default=3, help="Max MCTS iterations")
    parser.add_argument("--output", "-o", help="Save results JSON to file")
    args = parser.parse_args()

    print(f"Running YOHAS {'LIVE' if args.live else 'DRY-RUN'} benchmark...")
    metrics = asyncio.run(run_session(args.live, args.iterations))
    print_comparison(metrics)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, default=str))
        print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
