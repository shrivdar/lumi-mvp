"""Test fixtures for agent tests — mock LLMClient, mock tools, sample tasks."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.models import (
    AgentTask,
    AgentType,
    EdgeConfidence,
    EdgeRelationType,
    EvidenceSource,
    EvidenceSourceType,
    KGEdge,
    KGNode,
    NodeType,
)
from world_model.knowledge_graph import InMemoryKnowledgeGraph

# ---------------------------------------------------------------------------
# Mock LLMClient
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Mock LLM client that returns configurable responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or []
        self._call_index = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    async def query(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        kg_context: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        research_id: str = "",
        agent_id: str = "",
    ) -> str:
        self.call_count += 1
        self.total_input_tokens += len(prompt) // 4
        if self._call_index < len(self._responses):
            response = self._responses[self._call_index]
            self._call_index += 1
        else:
            response = '{"error": "no more mock responses"}'
        self.total_output_tokens += len(response) // 4
        return response

    @staticmethod
    def parse_json(text: str) -> Any:
        """Extract JSON from text."""
        import re

        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())
        for i, ch in enumerate(text):
            if ch in ("{", "["):
                try:
                    return json.loads(text[i:])
                except json.JSONDecodeError:
                    continue
        raise ValueError("No valid JSON found")

    @property
    def token_summary(self) -> dict[str, int]:
        return {
            "calls": self.call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }


# ---------------------------------------------------------------------------
# Mock Tools
# ---------------------------------------------------------------------------


def make_mock_tool(name: str, responses: dict[str, Any] | None = None) -> MagicMock:
    """Create a mock BaseTool with configurable responses."""
    tool = MagicMock()
    tool.tool_id = name
    tool.name = name
    tool.description = f"Mock {name} tool"

    default_response = responses or {"results": []}

    async def mock_execute(**kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "search")
        if isinstance(default_response, dict) and action in default_response:
            return default_response[action]
        return default_response

    tool.execute = AsyncMock(side_effect=mock_execute)
    return tool


@pytest.fixture()
def mock_llm() -> MockLLMClient:
    """Mock LLM client with empty responses."""
    return MockLLMClient()


@pytest.fixture()
def mock_pubmed_tool() -> MagicMock:
    """Mock PubMed tool with sample results."""
    return make_mock_tool("pubmed", {
        "results": [
            {
                "pmid": "PMID:12345678",
                "title": "BRCA1 mutations in breast cancer",
                "abstract": "BRCA1 gene mutations are associated with increased risk of breast cancer.",
                "authors": ["Smith J", "Doe A"],
                "journal": "Nature Genetics",
                "doi": "10.1038/ng1234",
                "mesh_terms": ["BRCA1", "Breast Cancer"],
            },
            {
                "pmid": "PMID:87654321",
                "title": "TP53 tumor suppression mechanisms",
                "abstract": "TP53 plays a critical role in cell cycle regulation and apoptosis.",
                "authors": ["Jones B"],
                "journal": "Cell",
                "doi": "10.1016/cell.5678",
            },
        ]
    })


@pytest.fixture()
def mock_semantic_scholar_tool() -> MagicMock:
    """Mock Semantic Scholar tool."""
    return make_mock_tool("semantic_scholar", {
        "results": [
            {
                "paper_id": "abc123",
                "title": "Novel B7-H3 targeting in NSCLC",
                "abstract": "B7-H3 is overexpressed in non-small cell lung cancer.",
                "year": 2024,
                "citation_count": 15,
                "doi": "10.1234/novel",
            }
        ]
    })


@pytest.fixture()
def mock_tools(mock_pubmed_tool: MagicMock, mock_semantic_scholar_tool: MagicMock) -> dict[str, MagicMock]:
    """All mock tools as a dict."""
    return {
        "pubmed": mock_pubmed_tool,
        "semantic_scholar": mock_semantic_scholar_tool,
        "uniprot": make_mock_tool("uniprot", {
            "results": [
                {
                    "accession": "P38398",
                    "protein_name": "BRCA1",
                    "function": "E3 ubiquitin-protein ligase",
                    "sequence": "MDLSALREVE" * 10,
                    "organism": "Homo sapiens",
                    "gene_names": ["BRCA1"],
                }
            ]
        }),
        "kegg": make_mock_tool("kegg", {
            "results": [
                {"id": "hsa05224", "name": "Breast cancer pathway", "description": "Breast cancer signaling"},
            ]
        }),
        "reactome": make_mock_tool("reactome", {
            "results": [
                {"id": "R-HSA-109581", "name": "PI3K/AKT signaling", "description": "PI3K pathway"},
            ]
        }),
        "mygene": make_mock_tool("mygene", {
            "results": [
                {
                    "entrezgene": 672,
                    "symbol": "BRCA1",
                    "name": "BRCA1 DNA repair associated",
                    "summary": "Breast cancer susceptibility gene",
                    "type_of_gene": "protein-coding",
                    "ensembl": {"gene": "ENSG00000012048"},
                }
            ]
        }),
        "chembl": make_mock_tool("chembl", {
            "results": [
                {
                    "molecule_chembl_id": "CHEMBL83",
                    "pref_name": "Tamoxifen",
                    "max_phase": 4,
                    "molecule_type": "Small molecule",
                    "indication_class": "Breast cancer",
                    "first_approval": 1977,
                }
            ]
        }),
        "clinicaltrials": make_mock_tool("clinicaltrials", {
            "results": [
                {
                    "nct_id": "NCT00001234",
                    "title": "Phase III Trial of Tamoxifen in Breast Cancer",
                    "brief_summary": "Evaluating tamoxifen efficacy",
                    "phase": "Phase 3",
                    "status": "Completed",
                    "enrollment": 500,
                    "conditions": ["Breast Cancer"],
                    "interventions": ["Tamoxifen"],
                }
            ]
        }),
        "esm": make_mock_tool("esm", {
            "embeddings": [[0.1, 0.2, 0.3]],
        }),
    }


# ---------------------------------------------------------------------------
# Sample task
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_task() -> AgentTask:
    """A sample research task."""
    return AgentTask(
        task_id="task-001",
        research_id="research-001",
        agent_type=AgentType.LITERATURE_ANALYST,
        agent_id="agent-lit-001",
        hypothesis_branch="h-brca-breast-cancer",
        instruction="Investigate the role of BRCA1 in breast cancer development and treatment resistance.",
        context={"disease": "breast cancer", "gene": "BRCA1"},
    )


@pytest.fixture()
def agent_kg() -> InMemoryKnowledgeGraph:
    """Fresh KG for agent tests."""
    return InMemoryKnowledgeGraph(graph_id="agent-test-kg")


@pytest.fixture()
def seeded_kg(agent_kg: InMemoryKnowledgeGraph) -> InMemoryKnowledgeGraph:
    """KG pre-seeded with some nodes and edges for critic/designer testing."""
    nodes = [
        KGNode(
            id="n-brca1", type=NodeType.GENE, name="BRCA1",
            description="Breast cancer gene 1", confidence=0.9,
            created_by="agent-lit-1", hypothesis_branch="h-main",
        ),
        KGNode(
            id="n-tp53", type=NodeType.GENE, name="TP53",
            description="Tumor protein p53", confidence=0.95,
            created_by="agent-lit-1", hypothesis_branch="h-main",
        ),
        KGNode(
            id="n-bc", type=NodeType.DISEASE, name="Breast Cancer",
            confidence=0.99, created_by="agent-lit-1", hypothesis_branch="h-main",
        ),
    ]
    for n in nodes:
        agent_kg.add_node(n)

    edges = [
        KGEdge(
            id="e-brca1-bc", source_id="n-brca1", target_id="n-bc",
            relation=EdgeRelationType.ASSOCIATED_WITH,
            confidence=EdgeConfidence(overall=0.4, evidence_quality=0.3, evidence_count=1),
            evidence=[
                EvidenceSource(
                    source_type=EvidenceSourceType.PUBMED,
                    source_id="PMID:11111111",
                    title="Weak BRCA1 association",
                    claim="BRCA1 may be associated with breast cancer",
                    quality_score=0.3,
                )
            ],
            created_by="agent-lit-1", hypothesis_branch="h-main",
        ),
        KGEdge(
            id="e-tp53-bc", source_id="n-tp53", target_id="n-bc",
            relation=EdgeRelationType.ASSOCIATED_WITH,
            confidence=EdgeConfidence(overall=0.9, evidence_quality=0.9, evidence_count=5),
            created_by="agent-lit-1", hypothesis_branch="h-main",
        ),
    ]
    for e in edges:
        agent_kg.add_edge(e)

    return agent_kg
