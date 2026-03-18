#!/usr/bin/env python3
"""End-to-end LIVE integration test — B7-H3 NSCLC query through the full orchestrator.

This is the FIRST real test of the YOHAS 3.0 platform with:
- Real Claude API (Opus 4.6) for LLM calls
- Real PubMed, Semantic Scholar, UniProt, KEGG, ChEMBL, ClinicalTrials.gov
- Real MCTS hypothesis tree exploration
- Real agent swarms with multi-turn investigation loops
- Real self-falsification on every edge
- Real report generation

Budget cap: session_token_budget=500,000 (~$5-10)
Expected runtime: 5-20 minutes depending on iteration depth

Usage:
    ANTHROPIC_API_KEY=sk-... python scripts/run_b7h3_live.py
    ANTHROPIC_API_KEY=sk-... python scripts/run_b7h3_live.py --iterations 3 --budget 200000
    ANTHROPIC_API_KEY=sk-... python scripts/run_b7h3_live.py --dry-check
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import gc
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the backend package is importable
# ---------------------------------------------------------------------------
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------
from agents.factory import create_agent, create_agent_from_spec  # noqa: E402
from core.llm import LLMClient  # noqa: E402
from core.models import (  # noqa: E402
    AgentResult,
    ResearchConfig,
    SessionStatus,
)
from integrations.registry import IntegrationsRegistry  # noqa: E402
from integrations.tool_catalog import get_catalog  # noqa: E402
from orchestrator.research_loop import ResearchOrchestrator  # noqa: E402
from report.generator import generate_report, generate_report_v2  # noqa: E402
from world_model.knowledge_graph import InMemoryKnowledgeGraph  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUERY = "Role of B7-H3 (CD276) in non-small cell lung cancer immune evasion and therapeutic targeting"

DEFAULT_CONFIG = dict(
    max_hypothesis_depth=2,
    max_mcts_iterations=3,
    max_agents=6,
    max_agents_per_swarm=2,
    max_concurrent_agents=2,
    confidence_threshold=0.75,
    session_token_budget=500_000,
    agent_token_budget=200_000,
    max_llm_calls_per_agent=15,
    enable_falsification=True,
    enable_hitl=False,  # No human in the loop for automated test
)

# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------


class LiveMetrics:
    """Collects real-time metrics during the live run."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.events: list[dict[str, Any]] = []
        self.agent_results: list[dict[str, Any]] = []
        self.kg_snapshots: list[dict[str, Any]] = []
        self.iteration_metrics: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def record_event(self, event_type: str, **data: Any) -> None:
        self.events.append({
            "timestamp": time.monotonic() - self.start_time,
            "event": event_type,
            **data,
        })

    def record_agent_result(self, result: AgentResult) -> None:
        self.agent_results.append({
            "agent_id": result.agent_id,
            "agent_type": str(result.agent_type),
            "success": result.success,
            "nodes_added": len(result.nodes_added),
            "edges_added": len(result.edges_added),
            "falsification_results": len(result.falsification_results),
            "falsified_count": sum(1 for f in result.falsification_results if f.falsified),
            "llm_calls": result.llm_calls,
            "llm_tokens_used": result.llm_tokens_used,
            "duration_ms": result.duration_ms,
            "turns": len(result.turns),
            "errors": result.errors,
        })

    def record_kg_snapshot(self, kg: InMemoryKnowledgeGraph, iteration: int) -> None:
        self.kg_snapshots.append({
            "iteration": iteration,
            "timestamp": time.monotonic() - self.start_time,
            "nodes": kg.node_count(),
            "edges": kg.edge_count(),
        })

    def summary(self) -> dict[str, Any]:
        duration = self.end_time - self.start_time
        total_tokens = sum(r["llm_tokens_used"] for r in self.agent_results)
        total_llm_calls = sum(r["llm_calls"] for r in self.agent_results)
        total_nodes = sum(r["nodes_added"] for r in self.agent_results)
        total_edges = sum(r["edges_added"] for r in self.agent_results)
        total_falsified = sum(r["falsified_count"] for r in self.agent_results)
        total_fals_attempts = sum(r["falsification_results"] for r in self.agent_results)
        failed_agents = sum(1 for r in self.agent_results if not r["success"])

        return {
            "duration_seconds": round(duration, 1),
            "total_agents_spawned": len(self.agent_results),
            "failed_agents": failed_agents,
            "total_llm_calls": total_llm_calls,
            "total_tokens": total_tokens,
            "total_nodes_added": total_nodes,
            "total_edges_added": total_edges,
            "falsification_attempts": total_fals_attempts,
            "edges_falsified": total_falsified,
            "kg_final_snapshot": self.kg_snapshots[-1] if self.kg_snapshots else {},
            "error_count": len(self.errors),
        }


# ---------------------------------------------------------------------------
# Event listener for real-time logging
# ---------------------------------------------------------------------------

_metrics = LiveMetrics()


def _on_kg_event(event_type: str, data: dict[str, Any]) -> None:
    """Callback attached to the KG for real-time event logging."""
    _metrics.record_event(f"kg:{event_type}", **data)
    if event_type == "node_created":
        name = data.get("name", "?")
        ntype = data.get("type", "?")
        print(f"    KG + node [{ntype}] {name}")
    elif event_type == "edge_created":
        relation = data.get("relation", "?")
        print(f"    KG + edge [{relation}]")
    elif event_type == "contradiction_detected":
        print("    KG ! contradiction detected")


# ---------------------------------------------------------------------------
# Pre-flight check
# ---------------------------------------------------------------------------

async def preflight_check() -> bool:
    """Quick validation that the LLM + key integrations are reachable."""
    print("\n  Pre-flight checks:")
    all_ok = True

    # 1. LLM connectivity
    try:
        llm = LLMClient()
        response = await llm.query(
            "Respond with exactly: READY",
            system_prompt="You are a test probe. Respond with exactly the word READY.",
            max_tokens=10,
        )
        ok = "READY" in response.text.upper()
        print(f"    {'OK' if ok else 'FAIL'} LLM connectivity (Claude API)")
        if not ok:
            all_ok = False
    except Exception as exc:
        print(f"    FAIL LLM connectivity: {exc}")
        all_ok = False

    # 2. PubMed (quick search)
    try:
        from integrations.pubmed import PubMedTool
        tool = PubMedTool()
        result = await tool.execute(query="B7-H3 NSCLC", max_results=1)
        has_results = bool(result.get("articles") or result.get("results"))
        print(f"    {'OK' if has_results else 'WARN'} PubMed API")
    except Exception as exc:
        print(f"    WARN PubMed API: {exc} (non-fatal)")

    # 3. Semantic Scholar
    try:
        from integrations.semantic_scholar import SemanticScholarTool
        tool = SemanticScholarTool()
        result = await tool.execute(query="B7-H3 immune checkpoint", max_results=1)
        has_results = bool(result.get("papers") or result.get("results"))
        print(f"    {'OK' if has_results else 'WARN'} Semantic Scholar API")
    except Exception as exc:
        print(f"    WARN Semantic Scholar: {exc} (non-fatal)")

    return all_ok


# ---------------------------------------------------------------------------
# Main live run
# ---------------------------------------------------------------------------

async def run_live(args: argparse.Namespace) -> int:
    """Execute the full B7-H3 NSCLC live research session."""
    global _metrics

    print("=" * 72)
    print("  YOHAS 3.0 — LIVE END-TO-END INTEGRATION TEST")
    print("  First Real Run")
    print("=" * 72)
    print(f"\n  Query: {QUERY}")
    print(f"  Date:  {datetime.now(timezone.utc).isoformat()}")

    # --- Pre-flight ---
    if not args.skip_preflight:
        ok = await preflight_check()
        if not ok:
            print("\n  Pre-flight FAILED. Fix issues above or use --skip-preflight to bypass.")
            return 1
    print()

    # --- Build config ---
    config = ResearchConfig(
        max_hypothesis_depth=args.depth,
        max_mcts_iterations=args.iterations,
        max_agents=DEFAULT_CONFIG["max_agents"],
        max_agents_per_swarm=args.agents_per_swarm,
        max_concurrent_agents=DEFAULT_CONFIG["max_concurrent_agents"],
        confidence_threshold=DEFAULT_CONFIG["confidence_threshold"],
        session_token_budget=args.budget,
        agent_token_budget=DEFAULT_CONFIG["agent_token_budget"],
        max_llm_calls_per_agent=DEFAULT_CONFIG["max_llm_calls_per_agent"],
        enable_falsification=True,
        enable_hitl=False,
    )

    print("  Config:")
    print(f"    MCTS iterations:    {config.max_mcts_iterations}")
    print(f"    Hypothesis depth:   {config.max_hypothesis_depth}")
    print(f"    Agents per swarm:   {config.max_agents_per_swarm}")
    print(f"    Session token cap:  {config.session_token_budget:,}")
    print(f"    Agent token cap:    {config.agent_token_budget:,}")
    print()

    # --- Initialize components ---
    print("  Initializing components...")

    llm = LLMClient()
    from core.config import settings
    print(f"    LLMClient ready (model: {settings.llm_model})")

    kg = InMemoryKnowledgeGraph(graph_id="b7h3-live-test")
    kg.add_listener(_on_kg_event)
    print(f"    KnowledgeGraph ready (id: {kg.graph_id})")

    # Bootstrap real integrations (no Redis for this test — caching disabled)
    integrations = IntegrationsRegistry(redis=None)
    await integrations.bootstrap()
    tool_instances = {t.name: t for t in integrations.list_instances()}
    print(f"    Tools bootstrapped: {list(tool_instances.keys())}")

    # Tool catalog for swarm composer
    catalog = get_catalog()
    print(f"    Tool catalog: {len(catalog)} entries")

    # Prepare output directory early for checkpointing
    output_dir = Path(__file__).resolve().parent.parent / "outputs" / "live_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"b7h3_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint callback — saves KG + metrics after each MCTS iteration
    async def _checkpoint(session_id: str, iteration: int, orch: Any) -> None:
        try:
            # Save hypothesis tree state
            tree_dict = orch.tree.to_dict() if hasattr(orch, "tree") and orch.tree else {}
            ckpt = {
                "session_id": session_id,
                "iteration": iteration,
                "timestamp": time.monotonic() - _metrics.start_time,
                "kg_nodes": kg.node_count(),
                "kg_edges": kg.edge_count(),
                "agents_completed": len(_metrics.agent_results),
                "hypothesis_tree": tree_dict,
            }
            # Save incremental checkpoint
            with open(run_dir / "checkpoint.json", "w") as f:
                json.dump(ckpt, f, indent=2, default=str)
            # Save KG snapshot (overwrite each iteration)
            with open(run_dir / "knowledge_graph.json", "w") as f:
                json.dump(kg.to_json(), f, indent=2, default=str)
            # Save agent results so far
            with open(run_dir / "agent_results.json", "w") as f:
                json.dump(_metrics.agent_results, f, indent=2, default=str)
            print(f"    [checkpoint] iter={iteration} nodes={kg.node_count()} edges={kg.edge_count()}")
            gc.collect()
        except Exception as exc:
            print(f"    [checkpoint] WARN: {exc}")

    # Build orchestrator
    orchestrator = ResearchOrchestrator(
        llm=llm,
        kg=kg,
        agent_factory=create_agent,
        spec_factory=create_agent_from_spec,
        tool_entries=catalog,
        tool_instances=tool_instances,
        checkpoint_callback=_checkpoint,
    )
    print("    Orchestrator ready")
    print(f"    Output dir: {run_dir}")
    print()

    # Emergency checkpoint on unexpected exit (e.g. OOM, unhandled signal)
    orch = orchestrator  # alias for atexit closure

    def _save_checkpoint_on_exit() -> None:
        try:
            ckpt = {
                "emergency": True,
                "timestamp": time.monotonic() - _metrics.start_time if _metrics.start_time else 0,
                "kg_nodes": kg.node_count(),
                "kg_edges": kg.edge_count(),
                "agents_completed": len(_metrics.agent_results),
                "hypothesis_tree": orch.tree.to_dict() if orch and hasattr(orch, "tree") and orch.tree else None,
            }
            with open(run_dir / "emergency_checkpoint.json", "w") as f:
                json.dump(ckpt, f, indent=2, default=str)
            with open(run_dir / "knowledge_graph.json", "w") as f:
                json.dump(kg.to_json(), f, indent=2, default=str)
            with open(run_dir / "agent_results.json", "w") as f:
                json.dump(_metrics.agent_results, f, indent=2, default=str)
            print("  [emergency checkpoint saved]")
        except Exception:
            pass  # best-effort on exit

    atexit.register(_save_checkpoint_on_exit)

    # --- Run ---
    print("=" * 72)
    print("  STARTING RESEARCH SESSION")
    print("=" * 72 + "\n")

    _metrics = LiveMetrics()
    _metrics.start_time = time.monotonic()

    # Set up graceful shutdown on Ctrl+C
    _interrupted = False

    def _handle_sigint(sig: int, frame: Any) -> None:
        nonlocal _interrupted
        if _interrupted:
            print("\n\n  Force quit.")
            sys.exit(1)
        _interrupted = True
        print("\n\n  Graceful shutdown requested (Ctrl+C again to force)...")
        print("  Waiting for current agents to finish...\n")

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)

    session = None
    try:
        session = await orchestrator.run(QUERY, config)
    except KeyboardInterrupt:
        print("\n  Session interrupted by user.")
    except Exception as exc:
        _metrics.errors.append(str(exc))
        print(f"\n  SESSION ERROR: {exc}")
        traceback.print_exc()
    finally:
        _metrics.end_time = time.monotonic()
        signal.signal(signal.SIGINT, original_handler)

    # --- Collect final metrics ---
    # Record all agent results from the orchestrator
    for result in orchestrator._all_results:
        _metrics.record_agent_result(result)
    _metrics.record_kg_snapshot(kg, session.current_iteration if session else 0)

    # --- Print results ---
    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)

    if session:
        print(f"\n  Status:        {session.status}")
        print(f"  Iterations:    {session.current_iteration}")
        print(f"  KG nodes:      {kg.node_count()}")
        print(f"  KG edges:      {kg.edge_count()}")
        print(f"  Hypotheses:    {session.total_hypotheses}")

        if session.result:
            result = session.result
            print(f"  LLM calls:     {result.total_llm_calls}")
            print(f"  Total tokens:  {result.total_tokens:,}")
            print(f"  Duration:      {result.total_duration_ms / 1000:.1f}s")

            if result.best_hypothesis:
                print("\n  Best hypothesis:")
                print(f"    {result.best_hypothesis.hypothesis}")
                print(f"    Confidence: {result.best_hypothesis.confidence:.2f}")
                print(f"    Visits: {result.best_hypothesis.visit_count}")
                print(f"    Info gain: {result.best_hypothesis.avg_info_gain:.3f}")

            if result.key_findings:
                print(f"\n  Top findings ({len(result.key_findings)}):")
                for i, edge in enumerate(result.key_findings[:10], 1):
                    src = kg.get_node(edge.source_id)
                    tgt = kg.get_node(edge.target_id)
                    src_name = src.name if src else edge.source_id[:12]
                    tgt_name = tgt.name if tgt else edge.target_id[:12]
                    print(f"    {i}. {src_name} --[{edge.relation}]--> {tgt_name} "
                          f"(conf={edge.confidence.overall:.2f})")

            if result.contradictions:
                print(f"\n  Contradictions found: {len(result.contradictions)}")

            if result.screening:
                print(f"\n  Biosecurity: {result.screening.tier}")

    # --- Metrics summary ---
    summary = _metrics.summary()
    print("\n  Metrics summary:")
    print(f"    Duration:          {summary['duration_seconds']}s")
    print(f"    Agents spawned:    {summary['total_agents_spawned']}")
    print(f"    Failed agents:     {summary['failed_agents']}")
    print(f"    LLM calls:         {summary['total_llm_calls']}")
    print(f"    Tokens used:       {summary['total_tokens']:,}")
    print(f"    Nodes added:       {summary['total_nodes_added']}")
    print(f"    Edges added:       {summary['total_edges_added']}")
    print(f"    Falsification:     {summary['edges_falsified']}/{summary['falsification_attempts']}")
    print(f"    Errors:            {summary['error_count']}")

    # --- Save outputs (run_dir created earlier for checkpointing) ---

    # 1. Metrics JSON
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {metrics_path}")

    # 2. KG snapshot
    kg_path = run_dir / "knowledge_graph.json"
    with open(kg_path, "w") as f:
        json.dump(kg.to_json(), f, indent=2, default=str)
    print(f"  Saved: {kg_path}")

    # 3. Agent trajectories
    trajectories_path = run_dir / "agent_results.json"
    with open(trajectories_path, "w") as f:
        json.dump(_metrics.agent_results, f, indent=2, default=str)
    print(f"  Saved: {trajectories_path}")

    # 4. Event log
    events_path = run_dir / "events.json"
    with open(events_path, "w") as f:
        json.dump(_metrics.events, f, indent=2, default=str)
    print(f"  Saved: {events_path}")

    # 5. Report (V1 + V2)
    if session and session.result:
        try:
            report_v1 = await generate_report(session, session.result, kg, llm)
            report_v1_path = run_dir / "report_v1.md"
            with open(report_v1_path, "w") as f:
                f.write(report_v1)
            print(f"  Saved: {report_v1_path}")
        except Exception as exc:
            print(f"  WARN: Report V1 generation failed: {exc}")

        try:
            report_v2 = await generate_report_v2(session, session.result, kg, llm)
            report_v2_path = run_dir / "report_v2.md"
            with open(report_v2_path, "w") as f:
                f.write(report_v2)
            print(f"  Saved: {report_v2_path}")
        except Exception as exc:
            print(f"  WARN: Report V2 generation failed: {exc}")

    # 6. Full session JSON
    if session:
        session_path = run_dir / "session.json"
        with open(session_path, "w") as f:
            json.dump(session.model_dump(mode="json"), f, indent=2, default=str)
        print(f"  Saved: {session_path}")

    # --- Validation ---
    print("\n" + "=" * 72)
    print("  VALIDATION")
    print("=" * 72)

    checks: dict[str, bool] = {}
    if session:
        checks["session_completed"] = session.status == SessionStatus.COMPLETED
        checks["kg_has_nodes"] = kg.node_count() > 0
        checks["kg_has_edges"] = kg.edge_count() > 0
        checks["agents_ran"] = len(_metrics.agent_results) > 0
        checks["some_agents_succeeded"] = any(r["success"] for r in _metrics.agent_results)
        checks["has_result"] = session.result is not None
        checks["has_hypothesis"] = (
            session.result is not None
            and session.result.best_hypothesis is not None
            and bool(session.result.best_hypothesis.hypothesis)
        )
        checks["falsification_ran"] = summary["falsification_attempts"] > 0
        checks["under_token_budget"] = summary["total_tokens"] <= args.budget
    else:
        checks["session_created"] = False

    all_pass = True
    for name, passed in checks.items():
        status = "OK" if passed else "FAIL"
        print(f"    {status} {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  ALL CHECKS PASSED — First real end-to-end run successful!")
    else:
        failed = [n for n, p in checks.items() if not p]
        print(f"  {len(failed)} CHECK(S) FAILED: {', '.join(failed)}")

    print(f"\n  Output directory: {run_dir}")
    print("=" * 72 + "\n")

    # Clean up integrations
    await integrations.close_all()

    return 0 if all_pass else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOHAS 3.0 — Live B7-H3 NSCLC end-to-end integration test",
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_CONFIG["max_mcts_iterations"],
        help=f"Max MCTS iterations (default: {DEFAULT_CONFIG['max_mcts_iterations']})",
    )
    parser.add_argument(
        "--budget", type=int, default=DEFAULT_CONFIG["session_token_budget"],
        help=f"Session token budget (default: {DEFAULT_CONFIG['session_token_budget']:,})",
    )
    parser.add_argument(
        "--depth", type=int, default=DEFAULT_CONFIG["max_hypothesis_depth"],
        help=f"Max hypothesis tree depth (default: {DEFAULT_CONFIG['max_hypothesis_depth']})",
    )
    parser.add_argument(
        "--agents-per-swarm", type=int, default=DEFAULT_CONFIG["max_agents_per_swarm"],
        help=f"Agents per swarm (default: {DEFAULT_CONFIG['max_agents_per_swarm']})",
    )
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip pre-flight connectivity checks",
    )
    parser.add_argument(
        "--dry-check", action="store_true",
        help="Run only the pre-flight checks, then exit",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Custom output directory for results",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    # Validate API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        return 1

    if args.dry_check:
        ok = await preflight_check()
        return 0 if ok else 1

    return await run_live(args)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
