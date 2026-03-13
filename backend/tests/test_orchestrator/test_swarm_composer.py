"""Tests for the swarm composer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.models import (
    AgentType,
    HypothesisNode,
    HypothesisStatus,
    ResearchConfig,
    ToolRegistryEntry,
    ToolSourceType,
)
from orchestrator.swarm_composer import SwarmComposer


@pytest.fixture()
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.query = AsyncMock(return_value='["literature_analyst", "drug_hunter"]')
    llm.parse_json = MagicMock(return_value=["literature_analyst", "drug_hunter"])
    llm.token_summary = {"calls": 1, "total_tokens": 100}
    return llm


@pytest.fixture()
def tool_entries() -> list[ToolRegistryEntry]:
    return [
        ToolRegistryEntry(
            name="pubmed", description="PubMed",
            source_type=ToolSourceType.NATIVE, category="literature",
        ),
        ToolRegistryEntry(
            name="semantic_scholar", description="S2",
            source_type=ToolSourceType.NATIVE, category="literature",
        ),
        ToolRegistryEntry(
            name="chembl", description="ChEMBL",
            source_type=ToolSourceType.NATIVE, category="drug",
        ),
        ToolRegistryEntry(
            name="clinicaltrials", description="CT.gov",
            source_type=ToolSourceType.NATIVE, category="clinical",
        ),
    ]


@pytest.fixture()
def composer(mock_llm: MagicMock, tool_entries: list[ToolRegistryEntry]) -> SwarmComposer:
    return SwarmComposer(
        llm=mock_llm,
        tool_registry_entries=tool_entries,
        session_id="test-session",
    )


@pytest.fixture()
def hypothesis() -> HypothesisNode:
    return HypothesisNode(
        hypothesis="B7-H3 overexpression in NSCLC promotes immune evasion",
        rationale="B7-H3 is a checkpoint molecule",
        status=HypothesisStatus.EXPLORING,
    )


@pytest.fixture()
def config() -> ResearchConfig:
    return ResearchConfig(max_agents_per_swarm=5)


class TestSwarmComposition:
    @pytest.mark.asyncio
    async def test_compose_swarm_includes_critic(
        self, composer: SwarmComposer, hypothesis: HypothesisNode, config: ResearchConfig,
    ) -> None:
        agents = await composer.compose_swarm("B7-H3 NSCLC", hypothesis, config)
        assert AgentType.SCIENTIFIC_CRITIC in agents

    @pytest.mark.asyncio
    async def test_compose_swarm_respects_max(
        self, composer: SwarmComposer, hypothesis: HypothesisNode,
    ) -> None:
        config = ResearchConfig(max_agents_per_swarm=2)
        agents = await composer.compose_swarm("B7-H3 NSCLC", hypothesis, config)
        assert len(agents) <= 2

    @pytest.mark.asyncio
    async def test_compose_swarm_llm_failure_falls_back(
        self, composer: SwarmComposer, hypothesis: HypothesisNode, config: ResearchConfig,
    ) -> None:
        composer.llm.query = AsyncMock(side_effect=Exception("LLM down"))
        agents = await composer.compose_swarm("drug discovery for cancer", hypothesis, config)
        # Should still return agents via fallback
        assert len(agents) > 0
        assert AgentType.SCIENTIFIC_CRITIC in agents

    @pytest.mark.asyncio
    async def test_compose_swarm_with_custom_agent_types(
        self, composer: SwarmComposer, hypothesis: HypothesisNode,
    ) -> None:
        config = ResearchConfig(
            agent_types=[AgentType.LITERATURE_ANALYST, AgentType.SCIENTIFIC_CRITIC],
            max_agents_per_swarm=5,
        )
        agents = await composer.compose_swarm("test", hypothesis, config)
        assert AgentType.SCIENTIFIC_CRITIC in agents


class TestTaskGeneration:
    @pytest.mark.asyncio
    async def test_generate_tasks(
        self, composer: SwarmComposer, hypothesis: HypothesisNode,
    ) -> None:
        composer.llm.query = AsyncMock(return_value='{"literature_analyst": "Search PubMed for B7-H3 studies"}')

        agent_types = [AgentType.LITERATURE_ANALYST, AgentType.SCIENTIFIC_CRITIC]
        tasks = await composer.generate_tasks("B7-H3 NSCLC", hypothesis, agent_types, "research-1")

        assert len(tasks) == 2
        for task in tasks:
            assert task.research_id == "research-1"
            assert task.hypothesis_branch == hypothesis.id
            assert task.instruction  # non-empty

    @pytest.mark.asyncio
    async def test_generate_tasks_fallback(
        self, composer: SwarmComposer, hypothesis: HypothesisNode,
    ) -> None:
        composer.llm.query = AsyncMock(side_effect=Exception("LLM down"))

        agent_types = [AgentType.LITERATURE_ANALYST]
        tasks = await composer.generate_tasks("test query", hypothesis, agent_types, "research-1")

        assert len(tasks) == 1
        assert tasks[0].instruction  # default instruction generated


class TestToolSelection:
    def test_select_tools(self, composer: SwarmComposer) -> None:
        entries = composer.select_tools_for_agent(
            AgentType.LITERATURE_ANALYST, ["pubmed", "semantic_scholar"],
        )
        assert len(entries) == 2
        assert entries[0].name == "pubmed"

    def test_select_tools_missing(self, composer: SwarmComposer) -> None:
        entries = composer.select_tools_for_agent(
            AgentType.PROTEIN_ENGINEER, ["uniprot", "esm"],
        )
        assert len(entries) == 0  # not in our test entries


class TestFallbackSelection:
    def test_fallback_drug_query(self, composer: SwarmComposer) -> None:
        hypothesis = HypothesisNode(hypothesis="Drug targets for NSCLC")
        selected = composer._fallback_selection(
            "drug discovery for NSCLC", hypothesis, list(AgentType),
        )
        assert AgentType.DRUG_HUNTER in selected
        assert AgentType.LITERATURE_ANALYST in selected

    def test_fallback_protein_query(self, composer: SwarmComposer) -> None:
        hypothesis = HypothesisNode(hypothesis="Protein structure of B7-H3")
        selected = composer._fallback_selection(
            "protein binding domain analysis", hypothesis, list(AgentType),
        )
        assert AgentType.PROTEIN_ENGINEER in selected

    def test_fallback_minimum_agents(self, composer: SwarmComposer) -> None:
        hypothesis = HypothesisNode(hypothesis="Unknown query")
        selected = composer._fallback_selection(
            "xyz", hypothesis, list(AgentType),
        )
        assert len(selected) >= 2


class TestEvents:
    @pytest.mark.asyncio
    async def test_composition_emits_events(
        self, composer: SwarmComposer, hypothesis: HypothesisNode, config: ResearchConfig,
    ) -> None:
        await composer.compose_swarm("test", hypothesis, config)
        events = composer.drain_events()
        assert any(e.event_type == "swarm_composed" for e in events)
