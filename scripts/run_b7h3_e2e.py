#!/usr/bin/env python3
"""End-to-end live test: B7-H3 NSCLC ADC target evaluation.

Runs the FULL orchestrator pipeline with real Claude API calls and real native tools.
No mocks. This is the first E2E validation of the YOHAS system.

Usage:
    # From repo root (backend must be on PYTHONPATH):
    cd backend && ANTHROPIC_API_KEY=sk-... python -m scripts.run_b7h3_e2e

    # Or with .env loaded:
    cd backend && python -m scripts.run_b7h3_e2e

Requirements:
    - ANTHROPIC_API_KEY set (via env or ../.env)
    - No Docker required for this script (uses in-memory KG, no DB persistence)
    - Redis optional (falls back to in-memory rate limiting)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — ensure backend is importable
# ---------------------------------------------------------------------------
_backend = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_backend))

# Load .env if it exists (for ANTHROPIC_API_KEY)
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set. Export it or add to .env")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Imports (after path + env setup)
# ---------------------------------------------------------------------------
from core.config import settings  # noqa: E402
from core.llm import LLMClient  # noqa: E402
from core.models import ResearchConfig  # noqa: E402
from core.tool_registry import InMemoryToolRegistry  # noqa: E402
from integrations.registry import IntegrationsRegistry  # noqa: E402
from orchestrator.research_loop import ResearchOrchestrator  # noqa: E402
from world_model.knowledge_graph import InMemoryKnowledgeGraph  # noqa: E402


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
B7H3_QUERY = (
    "Investigate B7-H3 (CD276) as an antibody-drug conjugate target in non-small "
    "cell lung cancer. Evaluate expression specificity, immune evasion mechanisms, "
    "existing clinical evidence, and propose an ADC strategy with candidate payload "
    "and linker recommendations."
)

# ---------------------------------------------------------------------------
# Research config — tuned for first E2E (conservative to avoid blowup)
# ---------------------------------------------------------------------------
E2E_CONFIG = ResearchConfig(
    max_mcts_iterations=3,          # 3 MCTS iterations (enough to prove the loop)
    max_agents_per_swarm=4,         # 3 agents + 1 critic per hypothesis
    max_concurrent_agents=3,        # don't overwhelm the API
    max_total_agents=15,            # hard cap
    agent_token_budget=50_000,      # 50K tokens per agent
    session_token_budget=500_000,   # 500K total session budget
    session_timeout_seconds=1800,   # 30 min timeout
    max_hypothesis_depth=2,
    max_hypothesis_breadth=4,
    code_first=False,               # tool-calling mode for first run
)


async def main() -> None:
    print("=" * 70)
    print("YOHAS 3.0 — B7-H3 E2E Live Test")
    print("=" * 70)
    print(f"Query: {B7H3_QUERY[:80]}...")
    print(f"Model: {settings.llm_model}")
    print(f"Fast model: {settings.llm_fast_model}")
    print(f"Max MCTS iterations: {E2E_CONFIG.max_mcts_iterations}")
    print(f"Max agents per swarm: {E2E_CONFIG.max_agents_per_swarm}")
    print(f"Session token budget: {E2E_CONFIG.session_token_budget:,}")
    print("=" * 70)

    # 1. Initialize core components
    print("\n[1/4] Initializing LLM client...")
    llm = LLMClient()

    print("[2/4] Creating knowledge graph...")
    kg = InMemoryKnowledgeGraph()

    # 3. Set up native tools
    print("[3/4] Registering native tools...")
    tool_registry = InMemoryToolRegistry()
    integrations = IntegrationsRegistry(tool_registry=tool_registry)
    await integrations.bootstrap()

    # Also register MCP/container tool catalog entries (metadata only, no live servers)
    catalog_count = integrations.register_catalog_tools()

    tool_entries = tool_registry.list_tools()
    tool_instances = {t.name: t for t in integrations.list_instances()}
    native_names = list(tool_instances.keys())
    print(f"  Native tools ({len(native_names)}): {native_names}")
    print(f"  Catalog tools (MCP metadata): {catalog_count}")
    print(f"  Total tool entries: {len(tool_entries)}")

    # 4. Create orchestrator
    print("[4/4] Creating orchestrator...")
    orchestrator = ResearchOrchestrator(
        llm=llm,
        kg=kg,
        tool_entries=tool_entries,
        tool_instances=tool_instances,
    )

    # Run!
    print(f"\n{'─' * 70}")
    print("Starting research session...")
    print(f"{'─' * 70}\n")

    start = time.monotonic()
    try:
        session = await orchestrator.run(B7H3_QUERY, E2E_CONFIG)
    except Exception as exc:
        elapsed = time.monotonic() - start
        print(f"\n{'!' * 70}")
        print(f"RESEARCH SESSION FAILED after {elapsed:.0f}s")
        print(f"Error: {exc}")
        print(f"{'!' * 70}")
        traceback.print_exc()

        # Still print whatever we got
        print("\n--- Partial KG State ---")
        print(f"Nodes: {kg.node_count()}")
        print(f"Edges: {kg.edge_count()}")
        if kg.node_count() > 0:
            for node in list(kg._nodes.values())[:10]:
                print(f"  [{node.type}] {node.name}")
        if kg.edge_count() > 0:
            for edge in list(kg._edges.values())[:10]:
                print(f"  {edge.source_id} --[{edge.relation}]--> {edge.target_id} (conf: {edge.confidence.overall:.2f})")

        print(f"\nLLM usage: {llm.total_input_tokens:,} in / {llm.total_output_tokens:,} out")
        sys.exit(1)

    elapsed = time.monotonic() - start

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"RESEARCH SESSION COMPLETED in {elapsed:.0f}s")
    print(f"{'=' * 70}")

    print(f"\nSession ID: {session.id}")
    print(f"Status: {session.status}")

    # KG stats
    print(f"\n--- Knowledge Graph ---")
    print(f"Nodes: {kg.node_count()}")
    print(f"Edges: {kg.edge_count()}")

    if kg.node_count() > 0:
        print("\nTop nodes:")
        for node in list(kg._nodes.values())[:20]:
            print(f"  [{node.type}] {node.name}")

    if kg.edge_count() > 0:
        print("\nTop edges (by confidence):")
        edges = sorted(kg._edges.values(), key=lambda e: e.confidence.overall, reverse=True)
        for edge in edges[:15]:
            src = kg.get_node(edge.source_id)
            tgt = kg.get_node(edge.target_id)
            src_name = src.name if src else edge.source_id[:8]
            tgt_name = tgt.name if tgt else edge.target_id[:8]
            print(f"  {src_name} --[{edge.relation}]--> {tgt_name} (conf: {edge.confidence.overall:.2f})")

    # Result summary
    if session.result:
        result = session.result
        print(f"\n--- Research Result ---")
        print(f"Best hypothesis: {result.best_hypothesis.hypothesis[:120]}")
        print(f"Key findings (edges): {len(result.key_findings)}")
        for i, edge in enumerate(result.key_findings[:5], 1):
            src = kg.get_node(edge.source_id)
            tgt = kg.get_node(edge.target_id)
            src_name = src.name if src else edge.source_id[:8]
            tgt_name = tgt.name if tgt else edge.target_id[:8]
            print(f"  {i}. {src_name} --[{edge.relation}]--> {tgt_name} (conf: {edge.confidence.overall:.2f})")

        if result.hypothesis_ranking:
            print(f"\nHypothesis rankings: {len(result.hypothesis_ranking)}")
            for h in result.hypothesis_ranking[:5]:
                status = h.status if hasattr(h, 'status') else '?'
                conf = f"{h.score:.2f}" if hasattr(h, 'score') and h.score else '?'
                print(f"  - [{status}] {h.hypothesis[:100]}")

        if result.recommended_experiments:
            print(f"\nRecommended experiments: {len(result.recommended_experiments)}")
            for exp in result.recommended_experiments[:3]:
                print(f"  - {exp[:120]}")

        if result.report_markdown:
            doc_preview = result.report_markdown[:1000]
            print(f"\n--- Report (first 1000 chars) ---")
            print(doc_preview)

            # Save full document
            doc_path = Path(__file__).parent.parent / "data" / "b7h3_report.md"
            doc_path.write_text(result.report_markdown)
            print(f"\nFull report saved to: {doc_path}")

    # LLM usage
    print(f"\n--- LLM Usage ---")
    print(f"Total input tokens: {llm.total_input_tokens:,}")
    print(f"Total output tokens: {llm.total_output_tokens:,}")
    print(f"Total calls: {llm.call_count}")
    for model, stats in llm._per_model_usage.items():
        print(f"  {model}: {stats['calls']} calls, {stats['input_tokens']:,} in / {stats['output_tokens']:,} out")

    # Save KG snapshot
    kg_path = Path(__file__).parent.parent / "data" / "b7h3_kg_snapshot.json"
    kg_data = kg.to_cytoscape()
    kg_path.write_text(json.dumps(kg_data, indent=2, default=str))
    print(f"\nKG snapshot saved to: {kg_path}")

    # Success criteria check
    print(f"\n{'=' * 70}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'=' * 70}")
    checks = {
        "Orchestrator didn't crash": session.status.value != "FAILED",
        "KG has 15+ nodes": kg.node_count() >= 15,
        "KG has 20+ edges": kg.edge_count() >= 20,
        "Report generated": bool(session.result and session.result.report_markdown),
        "Key findings extracted": bool(session.result and session.result.key_findings),
    }
    all_passed = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {check}")

    print(f"\n{'OVERALL: PASS' if all_passed else 'OVERALL: FAIL'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
