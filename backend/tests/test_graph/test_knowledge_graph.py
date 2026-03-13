"""Tests for the in-memory knowledge graph engine."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from core.exceptions import GraphError
from core.models import (
    EdgeRelationType,
    EvidenceSource,
    EvidenceSourceType,
    KGEdge,
    KGNode,
    NodeType,
)
from world_model.knowledge_graph import InMemoryKnowledgeGraph

# ═══════════════════════════════════════════════════════════════════════════
# CRUD — Nodes
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeCRUD:
    def test_add_and_get_node(self, kg: InMemoryKnowledgeGraph) -> None:
        node = KGNode(id="n1", type=NodeType.GENE, name="BRCA1", created_by="agent-1")
        node_id = kg.add_node(node)

        assert node_id == "n1"
        assert kg.node_count() == 1

        retrieved = kg.get_node("n1")
        assert retrieved is not None
        assert retrieved.name == "BRCA1"
        assert retrieved.type == NodeType.GENE

    def test_get_node_by_name(self, kg: InMemoryKnowledgeGraph) -> None:
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="BRCA1", created_by="a1"))
        kg.add_node(KGNode(id="n2", type=NodeType.DISEASE, name="Cancer", created_by="a1"))

        assert kg.get_node_by_name("brca1") is not None
        assert kg.get_node_by_name("BRCA1") is not None
        assert kg.get_node_by_name("brca1", NodeType.GENE) is not None
        assert kg.get_node_by_name("brca1", NodeType.DISEASE) is None
        assert kg.get_node_by_name("nonexistent") is None

    def test_get_node_by_alias(self, kg: InMemoryKnowledgeGraph) -> None:
        node = KGNode(id="n1", type=NodeType.GENE, name="TP53", aliases=["p53", "tumor protein p53"], created_by="a1")
        kg.add_node(node)

        assert kg.get_node_by_name("p53") is not None
        assert kg.get_node_by_name("tumor protein p53") is not None

    def test_dedup_merges_existing_node(self, kg: InMemoryKnowledgeGraph) -> None:
        node1 = KGNode(
            id="n1",
            type=NodeType.GENE,
            name="BRCA1",
            external_ids={"ncbi": "672"},
            confidence=0.8,
            created_by="a1",
        )
        node2 = KGNode(
            id="n2",
            type=NodeType.GENE,
            name="BRCA1",
            aliases=["BRCA"],
            external_ids={"uniprot": "P38398"},
            confidence=0.9,
            created_by="a2",
        )

        id1 = kg.add_node(node1)
        id2 = kg.add_node(node2)

        assert id1 == id2 == "n1"
        assert kg.node_count() == 1

        merged = kg.get_node("n1")
        assert merged is not None
        assert "BRCA" in merged.aliases
        assert merged.external_ids.get("uniprot") == "P38398"
        assert merged.external_ids.get("ncbi") == "672"
        assert merged.confidence == 0.9  # max

    def test_update_node(self, kg: InMemoryKnowledgeGraph) -> None:
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="BRCA1", created_by="a1"))
        kg.update_node("n1", {"description": "Updated description", "confidence": 0.99})

        node = kg.get_node("n1")
        assert node is not None
        assert node.description == "Updated description"
        assert node.confidence == 0.99

    def test_update_nonexistent_node_raises(self, kg: InMemoryKnowledgeGraph) -> None:
        with pytest.raises(GraphError, match="not found"):
            kg.update_node("nonexistent", {"name": "x"})

    def test_get_nonexistent_node_returns_none(self, kg: InMemoryKnowledgeGraph) -> None:
        assert kg.get_node("nonexistent") is None


# ═══════════════════════════════════════════════════════════════════════════
# CRUD — Edges
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCRUD:
    def test_add_and_get_edge(self, kg: InMemoryKnowledgeGraph) -> None:
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="A", created_by="a1"))
        kg.add_node(KGNode(id="n2", type=NodeType.DISEASE, name="B", created_by="a1"))

        edge = KGEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            relation=EdgeRelationType.ASSOCIATED_WITH,
            created_by="a1",
        )
        eid = kg.add_edge(edge)

        assert eid == "e1"
        assert kg.edge_count() == 1

        retrieved = kg.get_edge("e1")
        assert retrieved is not None
        assert retrieved.relation == EdgeRelationType.ASSOCIATED_WITH

    def test_add_edge_missing_source_raises(self, kg: InMemoryKnowledgeGraph) -> None:
        kg.add_node(KGNode(id="n2", type=NodeType.GENE, name="B", created_by="a1"))
        edge = KGEdge(id="e1", source_id="missing", target_id="n2", relation=EdgeRelationType.INHIBITS, created_by="a1")

        with pytest.raises(GraphError, match="Source node"):
            kg.add_edge(edge)

    def test_add_edge_missing_target_raises(self, kg: InMemoryKnowledgeGraph) -> None:
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="A", created_by="a1"))
        edge = KGEdge(id="e1", source_id="n1", target_id="missing", relation=EdgeRelationType.INHIBITS, created_by="a1")

        with pytest.raises(GraphError, match="Target node"):
            kg.add_edge(edge)

    def test_get_edges_from_to_between(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        from_brca1 = populated_kg.get_edges_from("n-brca1")
        assert len(from_brca1) == 2  # BRCA1 → Cancer, BRCA1 → PI3K

        to_brca = populated_kg.get_edges_to("n-brca")
        assert len(to_brca) == 3  # BRCA1→Cancer, TP53→Cancer, Tamoxifen→Cancer

        between = populated_kg.get_edges_between("n-brca1", "n-brca")
        assert len(between) == 1

    def test_update_edge_confidence(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        evidence = EvidenceSource(
            source_type=EvidenceSourceType.PUBMED,
            source_id="PMID:99999999",
            quality_score=0.95,
            agent_id="agent-lit-2",
        )
        populated_kg.update_edge_confidence("e-brca1-brca", 0.98, evidence)

        edge = populated_kg.get_edge("e-brca1-brca")
        assert edge is not None
        assert edge.confidence.overall == 0.98
        assert len(edge.evidence) == 2  # original + new

    def test_mark_edge_falsified(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        evidence = [
            EvidenceSource(
                source_type=EvidenceSourceType.PUBMED,
                source_id="PMID:counter",
                agent_id="agent-critic",
            )
        ]
        populated_kg.mark_edge_falsified("e-brca1-pi3k", evidence)

        edge = populated_kg.get_edge("e-brca1-pi3k")
        assert edge is not None
        assert edge.falsified is True
        assert edge.confidence.overall < 0.3  # heavily penalized
        assert edge.falsification_evidence is not None

    def test_get_nonexistent_edge_returns_none(self, kg: InMemoryKnowledgeGraph) -> None:
        assert kg.get_edge("nonexistent") is None


# ═══════════════════════════════════════════════════════════════════════════
# Contradiction detection
# ═══════════════════════════════════════════════════════════════════════════


class TestContradictions:
    def test_contradiction_detected_on_add(self, kg: InMemoryKnowledgeGraph) -> None:
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="A", created_by="a1"))
        kg.add_node(KGNode(id="n2", type=NodeType.GENE, name="B", created_by="a1"))

        # A activates B
        kg.add_edge(KGEdge(
            id="e1", source_id="n1", target_id="n2",
            relation=EdgeRelationType.ACTIVATES, created_by="a1",
        ))

        # A inhibits B (contradiction!)
        kg.add_edge(KGEdge(
            id="e2", source_id="n1", target_id="n2",
            relation=EdgeRelationType.INHIBITS, created_by="a2",
        ))

        e1 = kg.get_edge("e1")
        e2 = kg.get_edge("e2")
        assert e1 is not None and e2 is not None
        assert e2.is_contradiction is True
        assert e1.is_contradiction is True
        assert "e1" in e2.contradicted_by
        assert "e2" in e1.contradicted_by

    def test_no_false_contradiction(self, kg: InMemoryKnowledgeGraph) -> None:
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="A", created_by="a1"))
        kg.add_node(KGNode(id="n2", type=NodeType.GENE, name="B", created_by="a1"))

        kg.add_edge(KGEdge(
            id="e1", source_id="n1", target_id="n2",
            relation=EdgeRelationType.ACTIVATES, created_by="a1",
        ))
        kg.add_edge(KGEdge(
            id="e2", source_id="n1", target_id="n2",
            relation=EdgeRelationType.UPREGULATES, created_by="a1",
        ))

        e1 = kg.get_edge("e1")
        e2 = kg.get_edge("e2")
        assert e1 is not None and not e1.is_contradiction
        assert e2 is not None and not e2.is_contradiction

    def test_get_contradictions_query(self, kg: InMemoryKnowledgeGraph) -> None:
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="A", created_by="a1"))
        kg.add_node(KGNode(id="n2", type=NodeType.GENE, name="B", created_by="a1"))

        kg.add_edge(KGEdge(
            id="e1", source_id="n1", target_id="n2",
            relation=EdgeRelationType.UPREGULATES, created_by="a1",
        ))

        probe = KGEdge(
            id="e-probe", source_id="n1", target_id="n2",
            relation=EdgeRelationType.DOWNREGULATES, created_by="a1",
        )
        contradictions = kg.get_contradictions(probe)
        assert len(contradictions) == 1
        assert contradictions[0].id == "e1"


# ═══════════════════════════════════════════════════════════════════════════
# Graph queries
# ═══════════════════════════════════════════════════════════════════════════


class TestGraphQueries:
    def test_get_subgraph(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        sub = populated_kg.get_subgraph("n-brca1", hops=1)
        node_ids = {n["id"] for n in sub["nodes"]}
        assert "n-brca1" in node_ids
        assert "n-brca" in node_ids  # 1 hop via edge
        assert "n-pi3k" in node_ids  # 1 hop via edge

    def test_get_subgraph_2_hops(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        sub = populated_kg.get_subgraph("n-tamoxifen", hops=2)
        node_ids = {n["id"] for n in sub["nodes"]}
        # tamoxifen → cancer (1 hop) → BRCA1, TP53 (2 hops)
        assert "n-tamoxifen" in node_ids
        assert "n-brca" in node_ids
        assert "n-brca1" in node_ids
        assert "n-tp53" in node_ids

    def test_shortest_path(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        path = populated_kg.shortest_path("n-brca1", "n-brca")
        assert path is not None
        assert path == ["n-brca1", "n-brca"]

    def test_shortest_path_multi_hop(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        # tamoxifen → cancer ← BRCA1 → PI3K
        path = populated_kg.shortest_path("n-tamoxifen", "n-pi3k")
        assert path is not None
        assert path[0] == "n-tamoxifen"
        assert path[-1] == "n-pi3k"
        assert len(path) == 4  # tamoxifen → cancer → BRCA1 → PI3K

    def test_shortest_path_no_path(self, kg: InMemoryKnowledgeGraph) -> None:
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="A", created_by="a1"))
        kg.add_node(KGNode(id="n2", type=NodeType.GENE, name="B", created_by="a1"))
        assert kg.shortest_path("n1", "n2") is None

    def test_shortest_path_same_node(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        assert populated_kg.shortest_path("n-brca1", "n-brca1") == ["n-brca1"]

    def test_get_upstream_downstream(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        upstream = populated_kg.get_upstream("n-brca", depth=1)
        upstream_ids = {n["id"] for n in upstream["nodes"]}
        assert "n-brca1" in upstream_ids
        assert "n-tp53" in upstream_ids
        assert "n-tamoxifen" in upstream_ids

        downstream = populated_kg.get_downstream("n-brca1", depth=1)
        downstream_ids = {n["id"] for n in downstream["nodes"]}
        assert "n-brca" in downstream_ids
        assert "n-pi3k" in downstream_ids

    def test_get_recent_edges(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        recent = populated_kg.get_recent_edges(n=2)
        assert len(recent) == 2

    def test_get_edges_by_hypothesis(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        edges = populated_kg.get_edges_by_hypothesis("h-breast-cancer")
        assert len(edges) == 4

    def test_get_weakest_edges(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        weakest = populated_kg.get_weakest_edges(n=2)
        assert len(weakest) == 2
        assert weakest[0].confidence.overall <= weakest[1].confidence.overall

    def test_get_orphan_nodes(self, kg: InMemoryKnowledgeGraph) -> None:
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="A", created_by="a1"))
        kg.add_node(KGNode(id="n2", type=NodeType.GENE, name="B", created_by="a1"))
        kg.add_node(KGNode(id="n3", type=NodeType.GENE, name="C", created_by="a1"))
        kg.add_edge(KGEdge(
            id="e1", source_id="n1", target_id="n2",
            relation=EdgeRelationType.INTERACTS_WITH, created_by="a1",
        ))

        orphans = kg.get_orphan_nodes()
        assert len(orphans) == 1
        assert orphans[0].id == "n3"


# ═══════════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════════


class TestStats:
    def test_counts(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        assert populated_kg.node_count() == 5
        assert populated_kg.edge_count() == 4

    def test_avg_confidence(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        avg = populated_kg.avg_confidence()
        assert 0.0 < avg < 1.0

    def test_avg_confidence_empty(self, kg: InMemoryKnowledgeGraph) -> None:
        assert kg.avg_confidence() == 0.0

    def test_edges_added_since(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        old_time = datetime.now(UTC) - timedelta(hours=1)
        count = populated_kg.edges_added_since(old_time)
        assert count == 4

        future_time = datetime.now(UTC) + timedelta(hours=1)
        assert populated_kg.edges_added_since(future_time) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════════════


class TestSerialization:
    def test_to_cytoscape(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        cyto = populated_kg.to_cytoscape()

        assert "elements" in cyto
        assert "metadata" in cyto
        assert cyto["metadata"]["node_count"] == 5
        assert cyto["metadata"]["edge_count"] == 4
        assert "branch_colors" in cyto["metadata"]

        nodes = [e for e in cyto["elements"] if e["group"] == "nodes"]
        edges = [e for e in cyto["elements"] if e["group"] == "edges"]
        assert len(nodes) == 5
        assert len(edges) == 4

        # Check node has visualization metadata
        node_data = nodes[0]["data"]
        assert "cluster_id" in node_data
        assert "visual_weight" in node_data
        assert "type_color" in node_data
        assert "branch_color" in node_data
        assert "animation_state" in node_data
        assert "created_at" in node_data

        # Check edge has animation hints
        edge_data = edges[0]["data"]
        assert "flow_speed" in edge_data
        assert "direction" in edge_data
        assert "pulse" in edge_data
        assert "branch_color" in edge_data
        assert "confidence_overall" in edge_data
        assert "evidence_count" in edge_data
        assert "replication_count" in edge_data
        assert "falsification_attempts" in edge_data

    def test_to_json(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        data = populated_kg.to_json()
        assert data["graph_id"] == "test-graph"
        assert len(data["nodes"]) == 5
        assert len(data["edges"]) == 4
        assert "avg_confidence" in data["metadata"]

    def test_to_markdown_summary(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        md = populated_kg.to_markdown_summary()
        assert "Knowledge Graph Summary" in md
        assert "Nodes:" in md
        assert "Edges:" in md
        assert "GENE" in md  # type breakdown

    def test_json_round_trip(self, populated_kg: InMemoryKnowledgeGraph) -> None:
        data = populated_kg.to_json()

        new_kg = InMemoryKnowledgeGraph()
        new_kg.load_from_json(data)

        assert new_kg.node_count() == populated_kg.node_count()
        assert new_kg.edge_count() == populated_kg.edge_count()
        assert new_kg.graph_id == "test-graph"

        # Verify a node survived the round trip
        brca1 = new_kg.get_node("n-brca1")
        assert brca1 is not None
        assert brca1.name == "BRCA1"

        # Verify an edge survived the round trip
        edge = new_kg.get_edge("e-brca1-brca")
        assert edge is not None
        assert edge.confidence.overall == 0.92


# ═══════════════════════════════════════════════════════════════════════════
# Thread safety
# ═══════════════════════════════════════════════════════════════════════════


class TestThreadSafety:
    def test_concurrent_node_writes(self, kg: InMemoryKnowledgeGraph) -> None:
        """Verify no data corruption under concurrent writes."""
        errors: list[Exception] = []

        def add_nodes(prefix: str, count: int) -> None:
            try:
                for i in range(count):
                    kg.add_node(KGNode(
                        id=f"{prefix}-{i}",
                        type=NodeType.GENE,
                        name=f"{prefix}_gene_{i}",
                        created_by="agent-test",
                    ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_nodes, args=(f"t{t}", 50)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert kg.node_count() == 200

    def test_concurrent_mixed_operations(self, kg: InMemoryKnowledgeGraph) -> None:
        """Mix reads and writes concurrently."""
        # Pre-populate
        for i in range(20):
            kg.add_node(KGNode(id=f"n-{i}", type=NodeType.GENE, name=f"Gene{i}", created_by="a1"))
        for i in range(19):
            kg.add_edge(KGEdge(
                id=f"e-{i}",
                source_id=f"n-{i}",
                target_id=f"n-{i+1}",
                relation=EdgeRelationType.INTERACTS_WITH,
                created_by="a1",
            ))

        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(50):
                    kg.node_count()
                    kg.get_node("n-0")
                    kg.get_edges_from("n-0")
                    kg.avg_confidence()
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(20, 50):
                    kg.add_node(KGNode(id=f"n-w-{i}", type=NodeType.PROTEIN, name=f"Protein{i}", created_by="a2"))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ═══════════════════════════════════════════════════════════════════════════
# Event system
# ═══════════════════════════════════════════════════════════════════════════


class TestEventSystem:
    def test_node_created_event(self, kg: InMemoryKnowledgeGraph) -> None:
        events: list[tuple[str, dict]] = []
        kg.add_listener(lambda t, d: events.append((t, d)))

        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="A", created_by="a1"))

        assert len(events) == 1
        assert events[0][0] == "node_created"
        assert events[0][1]["node_id"] == "n1"

    def test_edge_created_event(self, kg: InMemoryKnowledgeGraph) -> None:
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="A", created_by="a1"))
        kg.add_node(KGNode(id="n2", type=NodeType.GENE, name="B", created_by="a1"))

        events: list[tuple[str, dict]] = []
        kg.add_listener(lambda t, d: events.append((t, d)))

        kg.add_edge(KGEdge(
            id="e1", source_id="n1", target_id="n2",
            relation=EdgeRelationType.INHIBITS, created_by="a1",
        ))

        assert len(events) == 1
        assert events[0][0] == "edge_created"

    def test_listener_error_does_not_crash(self, kg: InMemoryKnowledgeGraph) -> None:
        def bad_listener(t: str, d: dict) -> None:
            raise RuntimeError("boom")

        kg.add_listener(bad_listener)

        # Should not raise
        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="A", created_by="a1"))
        assert kg.node_count() == 1

    def test_remove_listener(self, kg: InMemoryKnowledgeGraph) -> None:
        events: list[tuple[str, dict]] = []
        def cb(t: str, d: dict) -> None:
            events.append((t, d))

        kg.add_listener(cb)
        kg.remove_listener(cb)

        kg.add_node(KGNode(id="n1", type=NodeType.GENE, name="A", created_by="a1"))
        assert len(events) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Batch operations
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchOps:
    def test_batch_add_nodes(self, kg: InMemoryKnowledgeGraph) -> None:
        nodes = [
            KGNode(id=f"n{i}", type=NodeType.GENE, name=f"Gene{i}", created_by="a1")
            for i in range(10)
        ]
        ids = kg.batch_add_nodes(nodes)
        assert len(ids) == 10
        assert kg.node_count() == 10

    def test_batch_add_edges(self, kg: InMemoryKnowledgeGraph) -> None:
        for i in range(5):
            kg.add_node(KGNode(id=f"n{i}", type=NodeType.GENE, name=f"Gene{i}", created_by="a1"))

        edges = [
            KGEdge(
                id=f"e{i}", source_id=f"n{i}", target_id=f"n{i+1}",
                relation=EdgeRelationType.INTERACTS_WITH, created_by="a1",
            )
            for i in range(4)
        ]
        ids = kg.batch_add_edges(edges)
        assert len(ids) == 4
        assert kg.edge_count() == 4
