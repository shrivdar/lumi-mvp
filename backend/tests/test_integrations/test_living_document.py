"""Tests for the Living Document — auto-updated markdown from KG events."""

from __future__ import annotations

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
from integrations.living_document import LivingDocument
from world_model.knowledge_graph import InMemoryKnowledgeGraph


@pytest.fixture()
def doc() -> LivingDocument:
    return LivingDocument(session_id="test-session", title="Test Report")


class TestLivingDocumentAttach:
    def test_attach_and_detach(self, kg: InMemoryKnowledgeGraph, doc: LivingDocument) -> None:
        doc.attach(kg)
        assert len(kg._listeners) == 1
        doc.detach()
        assert len(kg._listeners) == 0

    def test_detach_without_attach(self, doc: LivingDocument) -> None:
        doc.detach()  # should not raise


class TestLivingDocumentUpdatesOnKGEvents:
    def test_node_created_triggers_update(self, kg: InMemoryKnowledgeGraph, doc: LivingDocument) -> None:
        doc.attach(kg)
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="BRCA1", created_by="agent-1"))

        assert doc.version_count == 1
        content = doc.current_content
        assert "BRCA1" not in content  # nodes don't show directly in summary
        assert "Entities discovered:** 1" in content

    def test_edge_created_triggers_update(
        self, kg: InMemoryKnowledgeGraph, doc: LivingDocument
    ) -> None:
        doc.attach(kg)
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="BRCA1", created_by="a1"))
        kg.add_node(KGNode(id="n2", type=NodeType.DISEASE, name="Cancer", created_by="a1"))
        kg.add_edge(KGEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            relation=EdgeRelationType.ASSOCIATED_WITH,
            confidence=EdgeConfidence(overall=0.85, evidence_quality=0.8, evidence_count=3),
            created_by="a1",
            hypothesis_branch="h-main",
        ))

        content = doc.current_content
        assert "BRCA1" in content
        assert "Cancer" in content
        assert "ASSOCIATED_WITH" in content
        assert "h-main" in content

    def test_edge_falsified_updates_document(
        self, kg: InMemoryKnowledgeGraph, doc: LivingDocument
    ) -> None:
        doc.attach(kg)
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="TP53", created_by="a1"))
        kg.add_node(KGNode(id="n2", type=NodeType.DISEASE, name="Cancer", created_by="a1"))
        kg.add_edge(KGEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            relation=EdgeRelationType.ACTIVATES,
            confidence=EdgeConfidence(overall=0.9, evidence_quality=0.9, evidence_count=5),
            created_by="a1",
            hypothesis_branch="h-test",
        ))

        version_before = doc.version_count
        kg.mark_edge_falsified("e1", evidence=[
            EvidenceSource(
                source_type=EvidenceSourceType.PUBMED,
                source_id="PMID:99999999",
                quality_score=0.95,
                agent_id="critic-1",
            )
        ])

        assert doc.version_count > version_before
        assert "e1" in doc._falsified

    def test_low_confidence_edge_flagged_uncertain(
        self, kg: InMemoryKnowledgeGraph, doc: LivingDocument
    ) -> None:
        doc.attach(kg)
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="GeneA", created_by="a1"))
        kg.add_node(KGNode(id="n2", type=NodeType.GENE, name="GeneB", created_by="a1"))
        kg.add_edge(KGEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            relation=EdgeRelationType.INTERACTS_WITH,
            confidence=EdgeConfidence(overall=0.3, evidence_quality=0.2, evidence_count=1),
            created_by="a1",
        ))

        assert "e1" in doc._uncertainties
        content = doc.current_content
        assert "needs more evidence" in content


class TestLivingDocumentRender:
    def test_empty_document_renders(self, doc: LivingDocument) -> None:
        content = doc.render()
        assert "# Test Report" in content
        assert "Executive Summary" in content
        assert "Hypotheses" in content
        assert "Evidence Map" in content
        assert "Key Findings" in content
        assert "Contradictions" in content
        assert "Uncertainties" in content

    def test_render_includes_session_id(self, doc: LivingDocument) -> None:
        content = doc.render()
        assert "test-session" in content

    def test_full_document_with_populated_kg(
        self, populated_kg: InMemoryKnowledgeGraph
    ) -> None:
        doc = LivingDocument(session_id="pop-session", title="Populated Report")
        doc.attach(populated_kg)

        # Manually fire events so the doc picks up existing nodes/edges
        for node in populated_kg._nodes.values():
            doc._on_kg_event("node_created", {"node_id": node.id})
        for edge in populated_kg._edges.values():
            doc._on_kg_event("edge_created", {
                "edge_id": edge.id,
                "hypothesis_branch": edge.hypothesis_branch,
                "is_contradiction": edge.is_contradiction,
            })

        content = doc.current_content
        assert "BRCA1" in content
        assert "Breast Cancer" in content
        assert "Tamoxifen" in content


class TestVersionHistory:
    def test_versions_are_tracked(self, kg: InMemoryKnowledgeGraph, doc: LivingDocument) -> None:
        doc.attach(kg)
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="BRCA1", created_by="a1"))
        kg.add_node(KGNode(id="n2", type=NodeType.GENE, name="TP53", created_by="a1"))

        assert doc.version_count >= 2
        v1 = doc.get_version(1)
        assert v1 is not None
        assert v1.trigger_event == "node_created"

    def test_diff_is_computed(self, kg: InMemoryKnowledgeGraph, doc: LivingDocument) -> None:
        doc.attach(kg)
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="BRCA1", created_by="a1"))
        kg.add_node(KGNode(id="n2", type=NodeType.GENE, name="TP53", created_by="a1"))

        # Second version should have a diff
        v2 = doc.get_version(2)
        assert v2 is not None
        assert v2.diff != ""
        assert "---" in v2.diff or "+++" in v2.diff

    def test_get_version_out_of_range(self, doc: LivingDocument) -> None:
        assert doc.get_version(0) is None
        assert doc.get_version(999) is None

    def test_get_latest_version_empty(self, doc: LivingDocument) -> None:
        assert doc.get_latest_version() is None

    def test_version_history_metadata(self, kg: InMemoryKnowledgeGraph, doc: LivingDocument) -> None:
        doc.attach(kg)
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="BRCA1", created_by="a1"))

        history = doc.get_version_history()
        assert len(history) >= 1
        assert "version" in history[0]
        assert "timestamp" in history[0]
        assert "trigger_event" in history[0]

    def test_get_diff_by_version(self, kg: InMemoryKnowledgeGraph, doc: LivingDocument) -> None:
        doc.attach(kg)
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="BRCA1", created_by="a1"))
        kg.add_node(KGNode(id="n2", type=NodeType.GENE, name="TP53", created_by="a1"))

        diff = doc.get_diff(2)
        assert diff  # non-empty for second version
        assert doc.get_diff(999) == ""  # out of range
