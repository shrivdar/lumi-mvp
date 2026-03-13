"""Root test configuration and shared fixtures."""

from __future__ import annotations

import os

# Ensure tests don't accidentally use real API keys
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("NCBI_API_KEY", "")
os.environ.setdefault("S2_API_KEY", "")
os.environ.setdefault("HF_API_TOKEN", "")
os.environ.setdefault("SLACK_BOT_TOKEN", "")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///test.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

import pytest

from core.models import (
    EdgeConfidence,
    EdgeRelationType,
    EvidenceSource,
    EvidenceSourceType,
    KGEdge,
    KGNode,
    NodeType,
)
from world_model.knowledge_graph import InMemoryKnowledgeGraph


@pytest.fixture()
def kg() -> InMemoryKnowledgeGraph:
    """Fresh in-memory knowledge graph."""
    return InMemoryKnowledgeGraph(graph_id="test-graph")


@pytest.fixture()
def sample_nodes() -> list[KGNode]:
    """A handful of biomedical nodes for testing."""
    return [
        KGNode(
            id="n-brca1",
            type=NodeType.GENE,
            name="BRCA1",
            aliases=["BRCA1/BRCA2-containing complex subunit 1"],
            description="Breast cancer type 1 susceptibility protein",
            external_ids={"ncbi_gene": "672"},
            confidence=0.95,
            created_by="agent-lit-1",
            hypothesis_branch="h-breast-cancer",
            sources=[
                EvidenceSource(
                    source_type=EvidenceSourceType.PUBMED,
                    source_id="PMID:12345678",
                    doi="10.1234/example",
                    quality_score=0.9,
                    confidence=0.95,
                )
            ],
        ),
        KGNode(
            id="n-tp53",
            type=NodeType.GENE,
            name="TP53",
            aliases=["p53", "tumor protein p53"],
            description="Tumor suppressor protein p53",
            external_ids={"ncbi_gene": "7157"},
            confidence=0.98,
            created_by="agent-lit-1",
            hypothesis_branch="h-breast-cancer",
        ),
        KGNode(
            id="n-brca",
            type=NodeType.DISEASE,
            name="Breast Cancer",
            aliases=["breast carcinoma"],
            description="Malignant neoplasm of the breast",
            external_ids={"mesh": "D001943"},
            confidence=0.99,
            created_by="agent-lit-1",
            hypothesis_branch="h-breast-cancer",
        ),
        KGNode(
            id="n-pi3k",
            type=NodeType.PATHWAY,
            name="PI3K/AKT Signaling",
            description="Phosphoinositide 3-kinase signaling pathway",
            external_ids={"reactome": "R-HSA-109581"},
            confidence=0.9,
            created_by="agent-pathway-1",
            hypothesis_branch="h-breast-cancer",
        ),
        KGNode(
            id="n-tamoxifen",
            type=NodeType.DRUG,
            name="Tamoxifen",
            description="Selective estrogen receptor modulator",
            external_ids={"chembl": "CHEMBL83"},
            confidence=0.95,
            created_by="agent-drug-1",
            hypothesis_branch="h-breast-cancer",
        ),
    ]


@pytest.fixture()
def sample_edges() -> list[KGEdge]:
    """Edges connecting the sample nodes."""
    return [
        KGEdge(
            id="e-brca1-brca",
            source_id="n-brca1",
            target_id="n-brca",
            relation=EdgeRelationType.ASSOCIATED_WITH,
            confidence=EdgeConfidence(
                overall=0.92,
                evidence_quality=0.9,
                evidence_count=5,
                replication_count=3,
            ),
            created_by="agent-lit-1",
            hypothesis_branch="h-breast-cancer",
            evidence=[
                EvidenceSource(
                    source_type=EvidenceSourceType.PUBMED,
                    source_id="PMID:11111111",
                    quality_score=0.9,
                )
            ],
        ),
        KGEdge(
            id="e-tp53-brca",
            source_id="n-tp53",
            target_id="n-brca",
            relation=EdgeRelationType.ASSOCIATED_WITH,
            confidence=EdgeConfidence(overall=0.88, evidence_quality=0.85, evidence_count=3),
            created_by="agent-lit-1",
            hypothesis_branch="h-breast-cancer",
        ),
        KGEdge(
            id="e-brca1-pi3k",
            source_id="n-brca1",
            target_id="n-pi3k",
            relation=EdgeRelationType.PARTICIPATES_IN,
            confidence=EdgeConfidence(overall=0.75, evidence_quality=0.7, evidence_count=2),
            created_by="agent-pathway-1",
            hypothesis_branch="h-breast-cancer",
        ),
        KGEdge(
            id="e-tamoxifen-brca",
            source_id="n-tamoxifen",
            target_id="n-brca",
            relation=EdgeRelationType.TREATS,
            confidence=EdgeConfidence(overall=0.85, evidence_quality=0.9, evidence_count=10),
            created_by="agent-drug-1",
            hypothesis_branch="h-breast-cancer",
        ),
    ]


@pytest.fixture()
def populated_kg(
    kg: InMemoryKnowledgeGraph,
    sample_nodes: list[KGNode],
    sample_edges: list[KGEdge],
) -> InMemoryKnowledgeGraph:
    """KG pre-loaded with sample nodes and edges."""
    for node in sample_nodes:
        kg.add_node(node)
    for edge in sample_edges:
        kg.add_edge(edge)
    return kg
