#!/usr/bin/env python3
"""Live integration test for the multi-turn agent loop.

Uses a REAL LLM call (Claude) with mock tools to validate:
- XML tag parsing (<think>, <tool>, <execute>, <answer>)
- Tool execution and observation accumulation
- Answer extraction and KG node/edge creation
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the backend package is importable
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, os.path.abspath(BACKEND_DIR))

# Set API key before importing anything that reads settings
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set")
    sys.exit(1)

os.environ["ANTHROPIC_API_KEY"] = API_KEY

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------
from core.interfaces import BaseTool
from core.llm import LLMClient
from core.models import AgentTask, AgentType
from agents.templates import LITERATURE_ANALYST_TEMPLATE
from agents.literature_analyst import LiteratureAnalystAgent
from world_model.knowledge_graph import InMemoryKnowledgeGraph


# ═══════════════════════════════════════════════════════════════════════════════
# Mock tools that return realistic fake results
# ═══════════════════════════════════════════════════════════════════════════════

class MockPubMedTool(BaseTool):
    """Returns realistic fake PubMed results."""

    tool_id = "pubmed"
    name = "pubmed"
    description = (
        "Search PubMed for biomedical literature. "
        "Args: {\"query\": \"search terms\", \"max_results\": 5}"
    )
    rate_limit = 10.0
    cache_ttl = 3600

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        query = kwargs.get("query", kwargs.get("action", ""))
        print(f"  📚 PubMed search: {str(query)[:80]}")
        return {
            "results": [
                {
                    "pmid": "PMID:25416956",
                    "title": "BRCA1 and BRCA2: different roles in a common pathway of genome protection",
                    "abstract": (
                        "BRCA1 and BRCA2 are tumor suppressor genes that play critical roles in DNA "
                        "damage repair through homologous recombination. Mutations in BRCA1 significantly "
                        "increase the risk of breast and ovarian cancer. BRCA1 functions as an E3 "
                        "ubiquitin ligase and is involved in cell cycle checkpoint control, chromatin "
                        "remodeling, and transcriptional regulation. Loss of BRCA1 function leads to "
                        "genomic instability and accumulation of DNA damage."
                    ),
                    "authors": ["Roy R", "Chun J", "Powell SN"],
                    "journal": "Nature Reviews Cancer",
                    "year": 2012,
                    "doi": "10.1038/nrc3181",
                    "citation_count": 1250,
                    "mesh_terms": ["BRCA1", "Breast Cancer", "DNA Repair", "Homologous Recombination"],
                },
                {
                    "pmid": "PMID:20215531",
                    "title": "Triple-negative breast cancer: molecular subtypes and new targets for therapy",
                    "abstract": (
                        "Triple-negative breast cancers (TNBCs) lack expression of estrogen receptor, "
                        "progesterone receptor, and HER2. Many TNBCs harbor BRCA1 mutations and show "
                        "defects in homologous recombination DNA repair. PARP inhibitors such as olaparib "
                        "exploit this vulnerability through synthetic lethality. Clinical trials show "
                        "significant benefit of PARP inhibitors in BRCA1-mutated breast cancers."
                    ),
                    "authors": ["Foulkes WD", "Smith IE", "Reis-Filho JS"],
                    "journal": "New England Journal of Medicine",
                    "year": 2010,
                    "doi": "10.1056/NEJMra1001389",
                    "citation_count": 3800,
                    "mesh_terms": ["BRCA1", "Triple Negative Breast Cancer", "PARP Inhibitors"],
                },
                {
                    "pmid": "PMID:29617664",
                    "title": "BRCA1 interacts with TP53 to regulate cell cycle arrest and apoptosis",
                    "abstract": (
                        "BRCA1 physically interacts with TP53 and enhances p53-dependent transcription "
                        "of genes involved in cell cycle arrest and apoptosis. This interaction is "
                        "critical for the tumor suppressive function of both proteins. Disruption of "
                        "the BRCA1-p53 axis leads to uncontrolled cell proliferation and resistance "
                        "to DNA-damaging chemotherapy agents."
                    ),
                    "authors": ["Zhang H", "Somasundaram K", "Peng Y"],
                    "journal": "Molecular Cell",
                    "year": 2018,
                    "doi": "10.1016/j.molcel.2018.02",
                    "citation_count": 450,
                    "mesh_terms": ["BRCA1", "TP53", "Apoptosis", "Cell Cycle"],
                },
            ]
        }


class MockSemanticScholarTool(BaseTool):
    """Returns realistic fake Semantic Scholar results."""

    tool_id = "semantic_scholar"
    name = "semantic_scholar"
    description = (
        "Search Semantic Scholar for academic papers with citation data. "
        "Args: {\"query\": \"search terms\", \"max_results\": 5}"
    )
    rate_limit = 10.0
    cache_ttl = 3600

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        query = kwargs.get("query", kwargs.get("action", ""))
        print(f"  🔬 Semantic Scholar search: {str(query)[:80]}")
        return {
            "results": [
                {
                    "paper_id": "s2-abc123",
                    "title": "Olaparib for BRCA1/2-mutated breast cancer: clinical outcomes and resistance mechanisms",
                    "abstract": (
                        "Olaparib, a PARP inhibitor, has shown significant clinical benefit in patients "
                        "with BRCA1/2-mutated breast cancer. In the OlympiAD trial, olaparib improved "
                        "progression-free survival compared to standard chemotherapy. However, resistance "
                        "mechanisms include BRCA1 reversion mutations, upregulation of drug efflux pumps, "
                        "and restoration of homologous recombination through secondary mutations."
                    ),
                    "year": 2023,
                    "citation_count": 89,
                    "doi": "10.1200/JCO.2023.1234",
                    "venue": "Journal of Clinical Oncology",
                    "authors": [{"name": "Robson M"}, {"name": "Im SA"}],
                },
                {
                    "paper_id": "s2-def456",
                    "title": "BRCA1 in the DNA damage response and at telomeres",
                    "abstract": (
                        "BRCA1 plays essential roles in the DNA damage response (DDR) and telomere "
                        "maintenance. BRCA1 is recruited to DNA double-strand breaks where it promotes "
                        "homologous recombination repair. At telomeres, BRCA1 prevents aberrant "
                        "processing and maintains genomic stability. Loss of BRCA1 leads to telomere "
                        "dysfunction and chromosomal instability, contributing to tumorigenesis."
                    ),
                    "year": 2022,
                    "citation_count": 156,
                    "doi": "10.1038/s41568-022-0456",
                    "venue": "Nature Reviews Cancer",
                    "authors": [{"name": "Deng CX"}],
                },
            ]
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Main test
# ═══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    print("=" * 70)
    print("  MULTI-TURN AGENT LOOP — LIVE LLM TEST")
    print("=" * 70)

    # 1. Create real LLM client
    llm = LLMClient()
    print(f"\n✅ LLMClient created (model: {llm._client.api_key[:12]}...)")

    # 2. Create in-memory KG
    kg = InMemoryKnowledgeGraph(graph_id="live-test-kg")
    print(f"✅ InMemoryKnowledgeGraph created (id: {kg.graph_id})")

    # 3. Create mock tools
    tools: dict[str, BaseTool] = {
        "pubmed": MockPubMedTool(),
        "semantic_scholar": MockSemanticScholarTool(),
    }
    print(f"✅ Mock tools created: {list(tools.keys())}")

    # 4. Create agent
    agent = LiteratureAnalystAgent(
        template=LITERATURE_ANALYST_TEMPLATE,
        llm=llm,
        kg=kg,
        tools=tools,
    )
    print(f"✅ LiteratureAnalystAgent created (id: {agent.agent_id[:12]}...)")

    # 5. Create task
    task = AgentTask(
        research_id="live-test-001",
        agent_type=AgentType.LITERATURE_ANALYST,
        agent_id=agent.agent_id,
        hypothesis_branch="h-brca1-breast-cancer",
        instruction="What is the role of BRCA1 in breast cancer?",
        context={"disease": "breast cancer", "gene": "BRCA1"},
    )
    print(f"✅ Task created: {task.instruction}")

    # 6. Execute
    print("\n" + "─" * 70)
    print("  EXECUTING MULTI-TURN INVESTIGATION")
    print("─" * 70 + "\n")

    start = time.monotonic()
    result = await agent.execute(task)
    elapsed = time.monotonic() - start

    # 7. Print results
    print("\n" + "─" * 70)
    print("  RESULTS")
    print("─" * 70)

    print(f"\n⏱  Duration: {elapsed:.1f}s")
    print(f"✅ Success: {result.success}")
    print(f"📊 LLM calls: {result.llm_calls}")
    print(f"🔢 Tokens used: {result.llm_tokens_used}")
    print(f"📝 Turns: {len(result.turns)}")

    # Print each turn
    print("\n── Turn Details ──")
    for turn in result.turns:
        action_preview = turn.parsed_action[:120].replace("\n", " ")
        exec_preview = (turn.execution_result[:120].replace("\n", " ")) if turn.execution_result else ""
        print(f"  Turn {turn.turn_number} [{turn.turn_type}]:")
        print(f"    Action: {action_preview}...")
        if exec_preview:
            print(f"    Result: {exec_preview}...")
        if turn.error:
            print(f"    ❌ Error: {turn.error[:100]}")
        print()

    # Print KG results
    print(f"── Knowledge Graph ──")
    print(f"  Nodes added: {len(result.nodes_added)}")
    for node in result.nodes_added:
        print(f"    • [{node.type}] {node.name} (conf={node.confidence:.2f})")
    print(f"  Edges added: {len(result.edges_added)}")
    for edge in result.edges_added:
        src = kg.get_node(edge.source_id)
        tgt = kg.get_node(edge.target_id)
        src_name = src.name if src else edge.source_id
        tgt_name = tgt.name if tgt else edge.target_id
        print(f"    • {src_name} --[{edge.relation}]--> {tgt_name} (conf={edge.confidence.overall:.2f})")

    print(f"\n── Summary ──")
    print(f"  {result.summary[:500]}")

    if result.errors:
        print(f"\n── Errors ──")
        for err in result.errors:
            print(f"  ❌ {err}")

    # 8. Validation
    print("\n" + "─" * 70)
    print("  VALIDATION")
    print("─" * 70)

    checks = {
        "success": result.success,
        "has_turns": len(result.turns) > 0,
        "has_nodes": len(result.nodes_added) > 0,
        "has_edges": len(result.edges_added) > 0,
        "has_summary": len(result.summary) > 10,
        "llm_calls_made": result.llm_calls > 0,
        "no_errors": len(result.errors) == 0,
    }

    all_pass = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {passed}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("🎉 ALL CHECKS PASSED — Multi-turn agent loop works with real LLM!")
    else:
        print("⚠️  SOME CHECKS FAILED — see details above")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
