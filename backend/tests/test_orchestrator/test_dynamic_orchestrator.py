"""Tests for dynamic orchestrator — spec composition, benchmark mode, token budget integration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.models import (
    AgentConstraints,
    AgentResult,
    AgentType,
    HypothesisNode,
    ResearchConfig,
    ToolRegistryEntry,
    ToolSourceType,
)
from orchestrator.swarm_composer import SwarmComposer
from orchestrator.token_budget import TokenBudgetManager
from tests.test_agents.conftest import MockLLMClient
from world_model.knowledge_graph import InMemoryKnowledgeGraph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tool_entries() -> list[ToolRegistryEntry]:
    return [
        ToolRegistryEntry(
            name="pubmed", description="PubMed search",
            source_type=ToolSourceType.NATIVE, category="literature_search",
        ),
        ToolRegistryEntry(
            name="semantic_scholar", description="Semantic Scholar search",
            source_type=ToolSourceType.NATIVE, category="literature_search",
        ),
        ToolRegistryEntry(
            name="uniprot", description="UniProt protein lookup",
            source_type=ToolSourceType.NATIVE, category="protein_analysis",
        ),
        ToolRegistryEntry(
            name="kegg", description="KEGG pathway analysis",
            source_type=ToolSourceType.NATIVE, category="pathway_analysis",
        ),
        ToolRegistryEntry(
            name="chembl", description="ChEMBL drug data",
            source_type=ToolSourceType.NATIVE, category="drug_discovery",
        ),
    ]


@pytest.fixture()
def sample_hypothesis() -> HypothesisNode:
    return HypothesisNode(
        id="h-test-1",
        hypothesis="B7-H3 overexpression drives NSCLC immune evasion via PD-L1 pathway crosstalk",
        rationale="B7-H3 is a known immune checkpoint and may interact with PD-L1 signaling",
        depth=1,
    )


@pytest.fixture()
def tool_entries() -> list[ToolRegistryEntry]:
    return _make_tool_entries()


# ---------------------------------------------------------------------------
# SwarmComposer.compose_swarm_specs
# ---------------------------------------------------------------------------


class TestComposeSwarmSpecs:
    @pytest.mark.asyncio
    async def test_compose_specs_from_llm(self, sample_hypothesis, tool_entries):
        """LLM returns valid agent specs."""
        llm_response = json.dumps([
            {
                "role": "Literature searcher for B7-H3",
                "instructions": "Search PubMed for B7-H3 in NSCLC. Focus on immune checkpoint interactions.",
                "tools": ["pubmed", "semantic_scholar"],
                "agent_type_hint": "literature_analyst",
            },
            {
                "role": "Protein analyst for B7-H3 structure",
                "instructions": "Analyze B7-H3 protein structure and binding domains via UniProt.",
                "tools": ["uniprot"],
                "agent_type_hint": "protein_engineer",
            },
        ])
        llm = MockLLMClient(responses=[llm_response])
        composer = SwarmComposer(
            llm=llm, tool_registry_entries=tool_entries, session_id="test",
        )
        config = ResearchConfig(max_agents_per_swarm=5)

        specs = await composer.compose_swarm_specs("B7-H3 NSCLC", sample_hypothesis, config)

        assert len(specs) >= 2  # at least 2 from LLM
        # Scientific critic is always auto-added
        has_critic = any(s.agent_type_hint == AgentType.SCIENTIFIC_CRITIC for s in specs)
        assert has_critic

        # Check spec structure
        lit_spec = next(s for s in specs if s.agent_type_hint == AgentType.LITERATURE_ANALYST)
        assert "B7-H3" in lit_spec.role
        assert "pubmed" in lit_spec.tools
        assert lit_spec.hypothesis_branch == sample_hypothesis.id

    @pytest.mark.asyncio
    async def test_compose_specs_fallback_on_llm_failure(self, sample_hypothesis, tool_entries):
        """Falls back to heuristic spec generation when LLM fails."""
        llm = MockLLMClient(responses=[
            "invalid json!@#",  # compose_swarm_specs fails — triggers heuristic fallback
        ])
        composer = SwarmComposer(
            llm=llm, tool_registry_entries=tool_entries, session_id="test",
        )
        config = ResearchConfig(max_agents_per_swarm=15)

        specs = await composer.compose_swarm_specs("test query", sample_hypothesis, config)

        assert len(specs) >= 2
        has_critic = any(s.agent_type_hint == AgentType.SCIENTIFIC_CRITIC for s in specs)
        assert has_critic

    @pytest.mark.asyncio
    async def test_compose_specs_with_constraints(self, sample_hypothesis, tool_entries):
        """Pre-allocated constraints are applied to specs."""
        llm_response = json.dumps([
            {
                "role": "Lit searcher",
                "instructions": "Search papers",
                "tools": ["pubmed"],
                "agent_type_hint": "literature_analyst",
            },
        ])
        llm = MockLLMClient(responses=[llm_response])
        composer = SwarmComposer(
            llm=llm, tool_registry_entries=tool_entries, session_id="test",
        )

        constraints = [
            AgentConstraints(token_budget=25_000, max_turns=10),
            AgentConstraints(token_budget=30_000, max_turns=12),
        ]
        config = ResearchConfig(max_agents_per_swarm=5)

        specs = await composer.compose_swarm_specs(
            "test", sample_hypothesis, config,
            agent_constraints=constraints,
        )

        # First spec should get the first constraint
        assert specs[0].constraints.token_budget == 25_000
        assert specs[0].constraints.max_turns == 10

    @pytest.mark.asyncio
    async def test_compose_specs_always_has_critic(self, sample_hypothesis, tool_entries):
        """Even if LLM returns no critic, one is auto-added."""
        llm_response = json.dumps([
            {
                "role": "Lit searcher",
                "instructions": "Search papers",
                "tools": ["pubmed"],
                "agent_type_hint": "literature_analyst",
            },
        ])
        llm = MockLLMClient(responses=[llm_response])
        composer = SwarmComposer(
            llm=llm, tool_registry_entries=tool_entries, session_id="test",
        )
        config = ResearchConfig(max_agents_per_swarm=5)

        specs = await composer.compose_swarm_specs("test", sample_hypothesis, config)

        critic_specs = [s for s in specs if s.agent_type_hint == AgentType.SCIENTIFIC_CRITIC]
        assert len(critic_specs) == 1
        assert "falsif" in critic_specs[0].instructions.lower() or "critic" in critic_specs[0].role.lower()

    @pytest.mark.asyncio
    async def test_compose_specs_caps_at_max(self, sample_hypothesis, tool_entries):
        """Specs are capped at max_agents_per_swarm."""
        many_agents = json.dumps([
            {"role": f"Agent {i}", "instructions": f"Do task {i}", "tools": ["pubmed"]}
            for i in range(20)
        ])
        llm = MockLLMClient(responses=[many_agents])
        composer = SwarmComposer(
            llm=llm, tool_registry_entries=tool_entries, session_id="test",
        )
        config = ResearchConfig(max_agents_per_swarm=4)

        specs = await composer.compose_swarm_specs("test", sample_hypothesis, config)
        assert len(specs) <= 4


# ---------------------------------------------------------------------------
# Token Budget + Swarm integration
# ---------------------------------------------------------------------------


class TestTokenBudgetSwarmIntegration:
    def test_budget_flows_to_agent_constraints(self):
        mgr = TokenBudgetManager(session_budget=1_000_000)
        mgr.allocate_hypothesis_budget("h1", active_hypothesis_count=2)
        config = ResearchConfig(
            agent_token_budget=50_000,
            max_llm_calls_per_agent=20,
        )

        constraints = mgr.allocate_for_swarm("h1", agent_count=5, config=config)
        assert len(constraints) == 5
        for c in constraints:
            assert c.token_budget <= config.agent_token_budget
            assert c.token_budget > 0

    def test_budget_exhaustion_blocks_further_allocation(self):
        mgr = TokenBudgetManager(session_budget=10_000)
        mgr.record_usage("h1", "a1", 10_000)
        assert mgr.is_exhausted()

        # Allocation still works but with minimal budget
        constraints = mgr.allocate_for_swarm("h1", agent_count=3, config=ResearchConfig())
        for c in constraints:
            # Should get minimum viable budget
            assert c.token_budget >= 0


# ---------------------------------------------------------------------------
# Benchmark mode (unit-level)
# ---------------------------------------------------------------------------


class TestBenchmarkDecomposition:
    @pytest.mark.asyncio
    async def test_decompose_question(self):
        """Orchestrator decomposes a benchmark question into a hypothesis."""
        from orchestrator.research_loop import ResearchOrchestrator

        llm = MockLLMClient(responses=[
            json.dumps({
                "hypothesis": "B7-H3 promotes NSCLC tumor growth via immune checkpoint signaling",
                "rationale": "B7-H3 is a co-stimulatory molecule that may be involved in tumor immunity",
            }),
        ])

        kg = InMemoryKnowledgeGraph(graph_id="bench-test")
        orch = ResearchOrchestrator(llm=llm, kg=kg)
        orch._session = MagicMock()
        orch._session.id = "bench-session"

        result = await orch._decompose_benchmark_question(
            "What is the role of B7-H3 in NSCLC?"
        )

        assert "B7-H3" in result["hypothesis"]
        assert result["rationale"]

    @pytest.mark.asyncio
    async def test_decompose_question_fallback(self):
        """Falls back gracefully when LLM fails."""
        from orchestrator.research_loop import ResearchOrchestrator

        llm = MockLLMClient(responses=["not json"])
        kg = InMemoryKnowledgeGraph(graph_id="bench-test")
        orch = ResearchOrchestrator(llm=llm, kg=kg)
        orch._session = MagicMock()
        orch._session.id = "bench-session"

        result = await orch._decompose_benchmark_question("What is X?")
        assert "hypothesis" in result
        assert "What is X?" in result["hypothesis"]


class TestSynthesizeBenchmarkAnswer:
    @pytest.mark.asyncio
    async def test_synthesize_answer(self):
        from orchestrator.research_loop import ResearchOrchestrator

        llm = MockLLMClient(responses=[
            "B7-H3 is an immune checkpoint that promotes tumor immune evasion in NSCLC.",
        ])
        kg = InMemoryKnowledgeGraph(graph_id="bench-test")
        orch = ResearchOrchestrator(llm=llm, kg=kg)
        orch._session = MagicMock()
        orch._session.id = "bench-session"

        results = [
            AgentResult(
                task_id="t1", agent_id="a1", agent_type=AgentType.LITERATURE_ANALYST,
                success=True, summary="B7-H3 is overexpressed in NSCLC tumors",
            ),
            AgentResult(
                task_id="t2", agent_id="a2", agent_type=AgentType.PROTEIN_ENGINEER,
                success=True, summary="B7-H3 protein has IgV-IgC domain structure",
            ),
        ]

        answer = await orch._synthesize_benchmark_answer(
            "What is B7-H3's role in NSCLC?", results,
        )
        assert "B7-H3" in answer

    @pytest.mark.asyncio
    async def test_synthesize_answer_fallback(self):
        """When LLM synthesis fails, falls back to concatenated summaries."""
        from orchestrator.research_loop import ResearchOrchestrator

        # Create an LLM that raises on query to trigger fallback
        llm = MagicMock()
        llm.query = AsyncMock(side_effect=Exception("LLM unavailable"))
        kg = InMemoryKnowledgeGraph(graph_id="bench-test")
        orch = ResearchOrchestrator(llm=llm, kg=kg)
        orch._session = MagicMock()
        orch._session.id = "bench-session"

        results = [
            AgentResult(
                task_id="t1", agent_id="a1", agent_type=AgentType.LITERATURE_ANALYST,
                success=True, summary="Summary A",
            ),
            AgentResult(
                task_id="t2", agent_id="a2", agent_type=AgentType.PROTEIN_ENGINEER,
                success=True, summary="Summary B",
            ),
        ]

        answer = await orch._synthesize_benchmark_answer("question?", results)
        assert "Summary A" in answer
        assert "Summary B" in answer


# ---------------------------------------------------------------------------
# ResearchConfig scale constants
# ---------------------------------------------------------------------------


class TestResearchConfigDefaults:
    def test_scale_constants(self):
        config = ResearchConfig()
        assert config.max_concurrent_agents == 100
        assert config.max_total_agents == 10_000
        assert config.max_hypothesis_breadth == 50
        assert config.max_hypothesis_depth == 5

    def test_agent_token_budget(self):
        config = ResearchConfig()
        assert config.agent_token_budget == 200_000
        assert config.session_token_budget == 10_000_000
