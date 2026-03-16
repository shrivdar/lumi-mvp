"""Tests for ToolRetriever — LLM-based dynamic tool selection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.tool_retriever import ToolRetriever
from core.llm import LLMResponse
from core.models import ToolRegistryEntry, ToolSourceType


def _r(text: str) -> LLMResponse:
    """Wrap a string in LLMResponse for mock LLM return values."""
    return LLMResponse(text=text, call_tokens=100)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entries() -> list[ToolRegistryEntry]:
    """Create a realistic set of tool entries."""
    tools = [
        ("pubmed", "Search PubMed for biomedical publications", "literature"),
        ("semantic_scholar", "Search Semantic Scholar for papers", "literature"),
        ("uniprot", "Protein sequences and annotations from UniProt", "protein"),
        ("esm", "Protein structure prediction via ESM", "protein"),
        ("kegg", "Pathway lookup via KEGG", "pathway"),
        ("reactome", "Pathway analysis via Reactome", "pathway"),
        ("mygene", "Gene annotation via MyGene", "genomics"),
        ("chembl", "Drug compound search via ChEMBL", "drug"),
        ("clinicaltrials", "Clinical trial search", "clinical"),
        ("slack", "Slack HITL notifications", "communication"),
    ]
    return [
        ToolRegistryEntry(
            name=name, description=desc,
            source_type=ToolSourceType.NATIVE, category=cat,
        )
        for name, desc, cat in tools
    ]


@pytest.fixture()
def tool_entries() -> list[ToolRegistryEntry]:
    return _make_entries()


@pytest.fixture()
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.query = AsyncMock(return_value=_r('["pubmed", "semantic_scholar"]'))
    return llm


@pytest.fixture()
def retriever(mock_llm: MagicMock, tool_entries: list[ToolRegistryEntry]) -> ToolRetriever:
    return ToolRetriever(llm=mock_llm, tool_entries=tool_entries)


# ---------------------------------------------------------------------------
# Tests: LLM-based selection
# ---------------------------------------------------------------------------


class TestToolRetrieverLLM:
    @pytest.mark.asyncio
    async def test_select_tools_for_literature_task(self, retriever: ToolRetriever) -> None:
        """LLM selects pubmed + semantic_scholar for a literature task."""
        retriever.llm.query = AsyncMock(return_value=_r('["pubmed", "semantic_scholar"]'))

        tools = await retriever.select_tools(
            task="Find papers about BRCA1 mutations in breast cancer",
            hypothesis="BRCA1 loss-of-function drives breast cancer",
            top_k=3,
        )

        assert "pubmed" in tools
        assert "semantic_scholar" in tools

    @pytest.mark.asyncio
    async def test_select_tools_for_protein_task(self, retriever: ToolRetriever) -> None:
        """LLM selects uniprot + esm for a protein task."""
        retriever.llm.query = AsyncMock(return_value=_r('["uniprot", "esm", "pubmed"]'))

        tools = await retriever.select_tools(
            task="Analyze the structure of B7-H3 protein and predict binding sites",
            top_k=3,
        )

        assert "uniprot" in tools
        assert "esm" in tools

    @pytest.mark.asyncio
    async def test_select_tools_for_drug_task(self, retriever: ToolRetriever) -> None:
        """LLM selects chembl + clinicaltrials for a drug discovery task."""
        retriever.llm.query = AsyncMock(return_value=_r('["chembl", "clinicaltrials", "pubmed"]'))

        tools = await retriever.select_tools(
            task="Find drugs that target B7-H3 and their clinical trial status",
            top_k=3,
        )

        assert "chembl" in tools
        assert "clinicaltrials" in tools

    @pytest.mark.asyncio
    async def test_select_respects_top_k(self, retriever: ToolRetriever) -> None:
        """Should return at most top_k tools."""
        retriever.llm.query = AsyncMock(
            return_value=_r('["pubmed", "semantic_scholar", "chembl", "clinicaltrials", "kegg"]')
        )

        tools = await retriever.select_tools(task="broad search", top_k=2)

        assert len(tools) <= 2

    @pytest.mark.asyncio
    async def test_select_filters_unknown_tools(self, retriever: ToolRetriever) -> None:
        """LLM response with unknown tool names should be filtered out."""
        retriever.llm.query = AsyncMock(
            return_value=_r('["pubmed", "fake_tool", "nonexistent"]')
        )

        tools = await retriever.select_tools(task="test", top_k=3)

        assert "pubmed" in tools
        assert "fake_tool" not in tools
        assert "nonexistent" not in tools

    @pytest.mark.asyncio
    async def test_select_returns_all_when_fewer_than_top_k(self, mock_llm: MagicMock) -> None:
        """When fewer tools exist than top_k, return all of them."""
        entries = [
            ToolRegistryEntry(
                name="pubmed", description="PubMed",
                source_type=ToolSourceType.NATIVE, category="literature",
            ),
            ToolRegistryEntry(
                name="semantic_scholar", description="S2",
                source_type=ToolSourceType.NATIVE, category="literature",
            ),
        ]
        retriever = ToolRetriever(llm=mock_llm, tool_entries=entries)

        tools = await retriever.select_tools(task="any task", top_k=5)

        # Should return all 2 without calling LLM
        assert len(tools) == 2


# ---------------------------------------------------------------------------
# Tests: Heuristic fallback
# ---------------------------------------------------------------------------


class TestToolRetrieverHeuristic:
    @pytest.mark.asyncio
    async def test_heuristic_fallback_on_llm_failure(self, retriever: ToolRetriever) -> None:
        """When LLM fails, heuristic should still return relevant tools."""
        retriever.llm.query = AsyncMock(side_effect=Exception("LLM unavailable"))

        tools = await retriever.select_tools(
            task="Search for publications about protein structure",
            hypothesis="Novel protein fold discovered",
            top_k=3,
        )

        assert len(tools) > 0
        assert len(tools) <= 3

    @pytest.mark.asyncio
    async def test_heuristic_matches_protein_keywords(self, retriever: ToolRetriever) -> None:
        retriever.llm.query = AsyncMock(side_effect=Exception("fail"))

        tools = await retriever.select_tools(
            task="Analyze protein sequence and predict structure using ESM",
            top_k=3,
        )

        assert "uniprot" in tools or "esm" in tools

    @pytest.mark.asyncio
    async def test_heuristic_matches_drug_keywords(self, retriever: ToolRetriever) -> None:
        retriever.llm.query = AsyncMock(side_effect=Exception("fail"))

        tools = await retriever.select_tools(
            task="Find drug compounds and inhibitors with IC50 data",
            top_k=3,
        )

        assert "chembl" in tools

    @pytest.mark.asyncio
    async def test_heuristic_matches_clinical_keywords(self, retriever: ToolRetriever) -> None:
        retriever.llm.query = AsyncMock(side_effect=Exception("fail"))

        tools = await retriever.select_tools(
            task="Search for clinical trial data about patient efficacy",
            top_k=3,
        )

        assert "clinicaltrials" in tools

    @pytest.mark.asyncio
    async def test_heuristic_pads_with_defaults(self, retriever: ToolRetriever) -> None:
        """When no keywords match, heuristic should still return tools."""
        retriever.llm.query = AsyncMock(side_effect=Exception("fail"))

        tools = await retriever.select_tools(
            task="do something completely unrelated xyz",
            top_k=2,
        )

        assert len(tools) == 2


# ---------------------------------------------------------------------------
# Tests: Disabled tools
# ---------------------------------------------------------------------------


class TestToolRetrieverDisabledTools:
    @pytest.mark.asyncio
    async def test_disabled_tools_excluded(self, mock_llm: MagicMock) -> None:
        entries = [
            ToolRegistryEntry(
                name="pubmed", description="PubMed",
                source_type=ToolSourceType.NATIVE, category="literature",
                enabled=True,
            ),
            ToolRegistryEntry(
                name="chembl", description="ChEMBL",
                source_type=ToolSourceType.NATIVE, category="drug",
                enabled=False,
            ),
        ]
        mock_llm.query = AsyncMock(return_value=_r('["pubmed", "chembl"]'))
        retriever = ToolRetriever(llm=mock_llm, tool_entries=entries)

        tools = await retriever.select_tools(task="Find drugs", top_k=3)

        assert "chembl" not in tools
