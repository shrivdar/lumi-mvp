"""In-memory knowledge graph engine with thread-safe writes, event emission, and rich serialization.

Implements the ``KnowledgeGraph`` protocol defined in ``core.interfaces``.

Design goals (from task requirements):
- LAB-Bench granularity: every edge carries DOI/PMID, extraction method,
  confidence decomposition, replication count, and falsification history.
- Rich Cytoscape serialization: temporal metadata, auto-clustering, visual
  weight, edge animation hints, hypothesis-branch coloring.
- High write throughput: thread-safe via ``threading.RLock``, batched writes,
  and event emission for real-time frontend updates.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from core.audit import AuditLogger
from core.constants import KG_MAX_EDGES, KG_MAX_NODES
from core.exceptions import GraphError
from core.models import (
    EdgeRelationType,
    EvidenceSource,
    KGEdge,
    KGNode,
    NodeType,
)

_audit = AuditLogger("knowledge_graph")

# ---------------------------------------------------------------------------
# Hypothesis-branch color palette
# ---------------------------------------------------------------------------
_BRANCH_COLORS: list[str] = [
    "#6366f1",  # indigo
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#ef4444",  # red
    "#8b5cf6",  # violet
    "#06b6d4",  # cyan
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#f97316",  # orange
    "#3b82f6",  # blue
]

# Node-type → default color
_NODE_TYPE_COLORS: dict[str, str] = {
    NodeType.PROTEIN: "#3b82f6",
    NodeType.GENE: "#10b981",
    NodeType.DISEASE: "#ef4444",
    NodeType.PATHWAY: "#8b5cf6",
    NodeType.DRUG: "#f59e0b",
    NodeType.CELL_TYPE: "#06b6d4",
    NodeType.TISSUE: "#ec4899",
    NodeType.CLINICAL_TRIAL: "#14b8a6",
    NodeType.PUBLICATION: "#6b7280",
}

# Contradiction relation pairs
_CONTRADICTION_PAIRS: dict[EdgeRelationType, EdgeRelationType] = {
    EdgeRelationType.INHIBITS: EdgeRelationType.ACTIVATES,
    EdgeRelationType.ACTIVATES: EdgeRelationType.INHIBITS,
    EdgeRelationType.UPREGULATES: EdgeRelationType.DOWNREGULATES,
    EdgeRelationType.DOWNREGULATES: EdgeRelationType.UPREGULATES,
    EdgeRelationType.EVIDENCE_FOR: EdgeRelationType.EVIDENCE_AGAINST,
    EdgeRelationType.EVIDENCE_AGAINST: EdgeRelationType.EVIDENCE_FOR,
    EdgeRelationType.SYNERGIZES_WITH: EdgeRelationType.ANTAGONIZES,
    EdgeRelationType.ANTAGONIZES: EdgeRelationType.SYNERGIZES_WITH,
}


EventCallback = Callable[[str, dict[str, Any]], None]


class InMemoryKnowledgeGraph:
    """Thread-safe in-memory knowledge graph with rich provenance tracking.

    Satisfies the ``KnowledgeGraph`` protocol from ``core.interfaces``.
    """

    def __init__(self, graph_id: str = "") -> None:
        self.graph_id = graph_id

        # Primary stores
        self._nodes: dict[str, KGNode] = {}
        self._edges: dict[str, KGEdge] = {}

        # Indices for fast lookup
        self._name_index: dict[str, str] = {}  # lowercase(name) → node_id
        self._type_index: dict[NodeType, set[str]] = defaultdict(set)  # type → {node_ids}
        self._outgoing: dict[str, list[str]] = defaultdict(list)  # source_id → [edge_ids]
        self._incoming: dict[str, list[str]] = defaultdict(list)  # target_id → [edge_ids]
        self._hypothesis_edges: dict[str, list[str]] = defaultdict(list)  # branch → [edge_ids]
        self._edge_pair_index: dict[tuple[str, str], list[str]] = defaultdict(list)  # (src, tgt) → [edge_ids]

        # Branch color assignment
        self._branch_colors: dict[str, str] = {}
        self._next_color_idx = 0

        # Event listeners (for WebSocket / real-time updates)
        self._listeners: list[EventCallback] = []

        # Write batching
        self._write_batch: deque[dict[str, Any]] = deque()
        self._batch_lock = threading.Lock()

        # Thread safety for all graph mutations
        self._lock = threading.RLock()

    # ── Event system ──────────────────────────────────────────────────────

    def add_listener(self, callback: EventCallback) -> None:
        self._listeners.append(callback)

    def remove_listener(self, callback: EventCallback) -> None:
        self._listeners = [cb for cb in self._listeners if cb is not callback]

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        for cb in self._listeners:
            try:
                cb(event_type, data)
            except Exception:
                pass  # never let listener errors crash the graph

    # ── Branch coloring ───────────────────────────────────────────────────

    def _get_branch_color(self, branch_id: str | None) -> str:
        if not branch_id:
            return "#94a3b8"  # slate for unassigned
        if branch_id not in self._branch_colors:
            self._branch_colors[branch_id] = _BRANCH_COLORS[self._next_color_idx % len(_BRANCH_COLORS)]
            self._next_color_idx += 1
        return self._branch_colors[branch_id]

    # ═══════════════════════════════════════════════════════════════════════
    # CRUD
    # ═══════════════════════════════════════════════════════════════════════

    def add_node(self, node: KGNode) -> str:
        with self._lock:
            if len(self._nodes) >= KG_MAX_NODES:
                raise GraphError(
                    f"Max node limit ({KG_MAX_NODES}) reached",
                    error_code="KG_MAX_NODES",
                )

            # Dedup: if a node with same name+type exists, return existing ID
            existing = self._find_existing_node(node.name, node.type)
            if existing:
                self._merge_node(existing, node)
                return existing

            self._nodes[node.id] = node
            self._name_index[node.name.lower()] = node.id
            for alias in node.aliases:
                self._name_index[alias.lower()] = node.id
            self._type_index[node.type].add(node.id)

        _audit.kg_mutation(
            "add_node",
            agent_id=node.created_by,
            hypothesis_branch=node.hypothesis_branch or "",
            node_id=node.id,
            node_type=node.type.value,
            node_name=node.name,
        )

        self._emit("node_created", {
            "node_id": node.id,
            "type": node.type.value,
            "name": node.name,
            "created_by": node.created_by,
            "hypothesis_branch": node.hypothesis_branch,
        })

        return node.id

    def add_edge(self, edge: KGEdge) -> str:
        with self._lock:
            if len(self._edges) >= KG_MAX_EDGES:
                raise GraphError(
                    f"Max edge limit ({KG_MAX_EDGES}) reached",
                    error_code="KG_MAX_EDGES",
                )

            if edge.source_id not in self._nodes:
                raise GraphError(
                    f"Source node {edge.source_id} not found",
                    error_code="KG_MISSING_NODE",
                )
            if edge.target_id not in self._nodes:
                raise GraphError(
                    f"Target node {edge.target_id} not found",
                    error_code="KG_MISSING_NODE",
                )

            # Check for contradictions
            contradictions = self._detect_contradictions(edge)
            if contradictions:
                edge.is_contradiction = True
                edge.contradicted_by = [c.id for c in contradictions]
                for c in contradictions:
                    c.is_contradiction = True
                    if edge.id not in c.contradicted_by:
                        c.contradicted_by.append(edge.id)

            self._edges[edge.id] = edge
            self._outgoing[edge.source_id].append(edge.id)
            self._incoming[edge.target_id].append(edge.id)
            self._edge_pair_index[(edge.source_id, edge.target_id)].append(edge.id)
            if edge.hypothesis_branch:
                self._hypothesis_edges[edge.hypothesis_branch].append(edge.id)

        _audit.kg_mutation(
            "add_edge",
            agent_id=edge.created_by,
            hypothesis_branch=edge.hypothesis_branch or "",
            edge_id=edge.id,
            relation=edge.relation.value,
            source_id=edge.source_id,
            target_id=edge.target_id,
            is_contradiction=edge.is_contradiction,
        )

        self._emit("edge_created", {
            "edge_id": edge.id,
            "relation": edge.relation.value,
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "created_by": edge.created_by,
            "hypothesis_branch": edge.hypothesis_branch,
            "is_contradiction": edge.is_contradiction,
        })

        return edge.id

    def get_node(self, node_id: str) -> KGNode | None:
        return self._nodes.get(node_id)

    def get_node_by_name(self, name: str, type: NodeType | None = None) -> KGNode | None:
        node_id = self._name_index.get(name.lower())
        if node_id is None:
            return None
        node = self._nodes.get(node_id)
        if node and type is not None and node.type != type:
            return None
        return node

    def get_edge(self, edge_id: str) -> KGEdge | None:
        return self._edges.get(edge_id)

    def get_edges_from(self, source_id: str) -> list[KGEdge]:
        return [self._edges[eid] for eid in self._outgoing.get(source_id, []) if eid in self._edges]

    def get_edges_to(self, target_id: str) -> list[KGEdge]:
        return [self._edges[eid] for eid in self._incoming.get(target_id, []) if eid in self._edges]

    def get_edges_between(self, source_id: str, target_id: str) -> list[KGEdge]:
        return [self._edges[eid] for eid in self._edge_pair_index.get((source_id, target_id), []) if eid in self._edges]

    def update_node(self, node_id: str, updates: dict[str, Any]) -> None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                raise GraphError(f"Node {node_id} not found", error_code="KG_NOT_FOUND")

            for key, value in updates.items():
                if hasattr(node, key) and key not in ("id", "created_at"):
                    setattr(node, key, value)
            node.updated_at = datetime.now(UTC)

        self._emit("node_updated", {"node_id": node_id, "updates": list(updates.keys())})

    def update_edge_confidence(self, edge_id: str, new_confidence: float, evidence: EvidenceSource) -> None:
        with self._lock:
            edge = self._edges.get(edge_id)
            if edge is None:
                raise GraphError(f"Edge {edge_id} not found", error_code="KG_NOT_FOUND")

            edge.evidence.append(evidence)
            edge.confidence.evidence_count = len(edge.evidence)
            edge.confidence.overall = new_confidence
            edge.confidence.last_evaluated = datetime.now(UTC)
            edge.updated_at = datetime.now(UTC)

        _audit.kg_mutation(
            "update_edge_confidence",
            agent_id=evidence.agent_id,
            hypothesis_branch=edge.hypothesis_branch or "",
            edge_id=edge_id,
            new_confidence=new_confidence,
        )

        self._emit("edge_confidence_updated", {
            "edge_id": edge_id,
            "new_confidence": new_confidence,
        })

    def mark_edge_falsified(self, edge_id: str, evidence: list[EvidenceSource]) -> None:
        with self._lock:
            edge = self._edges.get(edge_id)
            if edge is None:
                raise GraphError(f"Edge {edge_id} not found", error_code="KG_NOT_FOUND")

            edge.falsified = True
            edge.falsification_evidence = evidence
            edge.confidence.falsification_attempts += 1
            edge.confidence.overall = max(0.0, edge.confidence.overall * 0.3)
            edge.confidence.last_evaluated = datetime.now(UTC)
            edge.updated_at = datetime.now(UTC)

        agent_id = evidence[0].agent_id if evidence else ""
        _audit.falsification(
            agent_id=agent_id,
            edge_id=edge_id,
            result="falsified",
            evidence_count=len(evidence),
        )

        self._emit("edge_falsified", {"edge_id": edge_id})

    # ═══════════════════════════════════════════════════════════════════════
    # Queries
    # ═══════════════════════════════════════════════════════════════════════

    def get_subgraph(self, center_id: str, hops: int = 2) -> dict[str, Any]:
        visited_nodes: set[str] = set()
        visited_edges: set[str] = set()
        frontier = {center_id}

        for _ in range(hops):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited_nodes:
                    continue
                visited_nodes.add(nid)
                for eid in self._outgoing.get(nid, []):
                    edge = self._edges.get(eid)
                    if edge:
                        visited_edges.add(eid)
                        next_frontier.add(edge.target_id)
                for eid in self._incoming.get(nid, []):
                    edge = self._edges.get(eid)
                    if edge:
                        visited_edges.add(eid)
                        next_frontier.add(edge.source_id)
            frontier = next_frontier - visited_nodes

        # Include final frontier nodes
        visited_nodes |= frontier

        return {
            "nodes": [self._nodes[nid].model_dump(mode="json") for nid in visited_nodes if nid in self._nodes],
            "edges": [self._edges[eid].model_dump(mode="json") for eid in visited_edges if eid in self._edges],
        }

    def get_contradictions(self, edge: KGEdge) -> list[KGEdge]:
        return self._detect_contradictions(edge)

    def get_recent_edges(self, n: int = 20) -> list[KGEdge]:
        edges = sorted(self._edges.values(), key=lambda e: e.created_at, reverse=True)
        return edges[:n]

    def get_edges_by_hypothesis(self, branch_id: str) -> list[KGEdge]:
        return [self._edges[eid] for eid in self._hypothesis_edges.get(branch_id, []) if eid in self._edges]

    def get_weakest_edges(self, n: int = 10) -> list[KGEdge]:
        non_falsified = [e for e in self._edges.values() if not e.falsified]
        return sorted(non_falsified, key=lambda e: e.confidence.overall)[:n]

    def get_orphan_nodes(self) -> list[KGNode]:
        result = []
        for nid, node in self._nodes.items():
            has_edges = bool(self._outgoing.get(nid)) or bool(self._incoming.get(nid))
            if not has_edges:
                result.append(node)
        return result

    def shortest_path(self, source_id: str, target_id: str) -> list[str] | None:
        if source_id not in self._nodes or target_id not in self._nodes:
            return None
        if source_id == target_id:
            return [source_id]

        visited: set[str] = set()
        queue: deque[list[str]] = deque([[source_id]])

        while queue:
            path = queue.popleft()
            current = path[-1]
            if current in visited:
                continue
            visited.add(current)

            neighbors: set[str] = set()
            for eid in self._outgoing.get(current, []):
                edge = self._edges.get(eid)
                if edge:
                    neighbors.add(edge.target_id)
            for eid in self._incoming.get(current, []):
                edge = self._edges.get(eid)
                if edge:
                    neighbors.add(edge.source_id)

            for neighbor in neighbors:
                if neighbor == target_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    queue.append(path + [neighbor])

        return None

    def get_upstream(self, node_id: str, depth: int = 3) -> dict[str, Any]:
        return self._traverse_directed(node_id, depth, direction="upstream")

    def get_downstream(self, node_id: str, depth: int = 3) -> dict[str, Any]:
        return self._traverse_directed(node_id, depth, direction="downstream")

    # ═══════════════════════════════════════════════════════════════════════
    # Stats
    # ═══════════════════════════════════════════════════════════════════════

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def avg_confidence(self) -> float:
        if not self._edges:
            return 0.0
        return sum(e.confidence.overall for e in self._edges.values()) / len(self._edges)

    def edges_added_since(self, timestamp: datetime) -> int:
        return sum(1 for e in self._edges.values() if e.created_at >= timestamp)

    # ═══════════════════════════════════════════════════════════════════════
    # Serialization
    # ═══════════════════════════════════════════════════════════════════════

    def to_cytoscape(self) -> dict[str, Any]:
        """Serialize the full graph for Cytoscape.js with rich visualization metadata.

        Includes: temporal metadata, auto-clustering, visual weight, edge animation
        hints, and hypothesis-branch coloring.
        """
        clusters = self._compute_clusters()
        now = datetime.now(UTC)
        elements: list[dict[str, Any]] = []

        for node in self._nodes.values():
            out_degree = len(self._outgoing.get(node.id, []))
            in_degree = len(self._incoming.get(node.id, []))
            degree = out_degree + in_degree
            importance = min(1.0, 0.2 + (degree / max(len(self._edges), 1)) * 3.0)

            branch_color = self._get_branch_color(node.hypothesis_branch)
            type_color = _NODE_TYPE_COLORS.get(node.type, "#94a3b8")

            age_seconds = (now - node.created_at).total_seconds()
            is_recent = age_seconds < 60

            elements.append({
                "group": "nodes",
                "data": {
                    "id": node.id,
                    "label": node.name,
                    "type": node.type.value,
                    "confidence": node.confidence,
                    "created_by": node.created_by,
                    "hypothesis_branch": node.hypothesis_branch,
                    "source_count": len(node.sources),
                    "external_ids": node.external_ids,
                    "description": node.description,
                    # Visualization metadata
                    "cluster_id": clusters.get(node.id, f"type_{node.type.value}"),
                    "visual_weight": importance,
                    "type_color": type_color,
                    "branch_color": branch_color,
                    "animation_state": "pulsing" if is_recent else "stable",
                    # Temporal
                    "created_at": node.created_at.isoformat(),
                    "updated_at": node.updated_at.isoformat(),
                },
                "position": {
                    "x": node.viz.position_x or 0,
                    "y": node.viz.position_y or 0,
                },
            })

        for edge in self._edges.values():
            branch_color = self._get_branch_color(edge.hypothesis_branch)
            evidence_strength = edge.confidence.overall
            age_seconds = (now - edge.created_at).total_seconds()
            is_recent = age_seconds < 60

            elements.append({
                "group": "edges",
                "data": {
                    "id": edge.id,
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "relation": edge.relation.value,
                    "label": edge.relation.value.replace("_", " ").lower(),
                    # Confidence decomposition
                    "confidence_overall": edge.confidence.overall,
                    "confidence_evidence_quality": edge.confidence.evidence_quality,
                    "evidence_count": edge.confidence.evidence_count,
                    "replication_count": edge.confidence.replication_count,
                    "falsification_attempts": edge.confidence.falsification_attempts,
                    "falsification_failures": edge.confidence.falsification_failures,
                    # Provenance
                    "created_by": edge.created_by,
                    "hypothesis_branch": edge.hypothesis_branch,
                    "is_contradiction": edge.is_contradiction,
                    "falsified": edge.falsified,
                    "source_count": len(edge.evidence),
                    # Animation hints
                    "branch_color": branch_color,
                    "flow_speed": max(0.1, evidence_strength),
                    "direction": "forward",
                    "pulse": is_recent,
                    "visual_weight": evidence_strength,
                    "animation_state": "pulsing" if is_recent else ("dimmed" if edge.falsified else "stable"),
                    # Temporal
                    "created_at": edge.created_at.isoformat(),
                    "updated_at": edge.updated_at.isoformat(),
                },
            })

        return {
            "elements": elements,
            "metadata": {
                "graph_id": self.graph_id,
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "branch_colors": dict(self._branch_colors),
                "cluster_count": len(set(clusters.values())) if clusters else 0,
                "snapshot_at": now.isoformat(),
            },
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "nodes": [n.model_dump(mode="json") for n in self._nodes.values()],
            "edges": [e.model_dump(mode="json") for e in self._edges.values()],
            "metadata": {
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "avg_confidence": self.avg_confidence(),
                "branch_colors": dict(self._branch_colors),
            },
        }

    def to_markdown_summary(self) -> str:
        lines = [
            "# Knowledge Graph Summary",
            "",
            f"- **Nodes:** {len(self._nodes)}",
            f"- **Edges:** {len(self._edges)}",
            f"- **Avg confidence:** {self.avg_confidence():.3f}",
            "",
        ]

        # Node type breakdown
        lines.append("## Nodes by Type")
        for ntype, nids in sorted(self._type_index.items(), key=lambda x: -len(x[1])):
            lines.append(f"- {ntype.value}: {len(nids)}")

        # Top confidence edges
        if self._edges:
            lines.append("")
            lines.append("## Top Edges (by confidence)")
            top = sorted(self._edges.values(), key=lambda e: e.confidence.overall, reverse=True)[:10]
            for edge in top:
                src = self._nodes.get(edge.source_id)
                tgt = self._nodes.get(edge.target_id)
                src_name = src.name if src else edge.source_id
                tgt_name = tgt.name if tgt else edge.target_id
                lines.append(
                    f"- {src_name} --[{edge.relation.value}]--> {tgt_name} "
                    f"(confidence: {edge.confidence.overall:.2f}, "
                    f"evidence: {edge.confidence.evidence_count})"
                )

        # Contradictions
        contradictions = [e for e in self._edges.values() if e.is_contradiction]
        if contradictions:
            lines.append("")
            lines.append("## Contradictions")
            for edge in contradictions:
                src = self._nodes.get(edge.source_id)
                tgt = self._nodes.get(edge.target_id)
                src_name = src.name if src else edge.source_id
                tgt_name = tgt.name if tgt else edge.target_id
                lines.append(f"- {src_name} --[{edge.relation.value}]--> {tgt_name}")

        # Falsified edges
        falsified = [e for e in self._edges.values() if e.falsified]
        if falsified:
            lines.append("")
            lines.append("## Falsified Edges")
            for edge in falsified:
                src = self._nodes.get(edge.source_id)
                tgt = self._nodes.get(edge.target_id)
                src_name = src.name if src else edge.source_id
                tgt_name = tgt.name if tgt else edge.target_id
                lines.append(f"- ~~{src_name} --[{edge.relation.value}]--> {tgt_name}~~")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════
    # Persistence (JSON snapshot via DB)
    # ═══════════════════════════════════════════════════════════════════════

    async def save(self, session_id: str) -> None:
        """Persist graph as JSON snapshot to the kg_snapshots table."""
        from db.session import async_session_factory
        from db.tables import KGSnapshotRow

        snapshot = self.to_json()
        row = KGSnapshotRow(session_id=session_id, snapshot=snapshot)

        async with async_session_factory() as session:
            session.add(row)
            await session.commit()

        _audit.log("kg_saved", session_id=session_id, node_count=len(self._nodes), edge_count=len(self._edges))

    async def load(self, session_id: str) -> None:
        """Load graph from the most recent JSON snapshot for the given session."""
        from sqlalchemy import select

        from db.session import async_session_factory
        from db.tables import KGSnapshotRow

        async with async_session_factory() as session:
            stmt = (
                select(KGSnapshotRow)
                .where(KGSnapshotRow.session_id == session_id)
                .order_by(KGSnapshotRow.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()

        if row is None:
            raise GraphError(
                f"No snapshot found for session {session_id}",
                error_code="KG_NO_SNAPSHOT",
            )

        self._load_from_dict(row.snapshot)
        _audit.log("kg_loaded", session_id=session_id, node_count=len(self._nodes), edge_count=len(self._edges))

    def load_from_json(self, data: dict[str, Any]) -> None:
        """Load graph from a JSON dict (for testing or in-memory restore)."""
        self._load_from_dict(data)

    # ═══════════════════════════════════════════════════════════════════════
    # Batch operations
    # ═══════════════════════════════════════════════════════════════════════

    def batch_add_nodes(self, nodes: list[KGNode]) -> list[str]:
        return [self.add_node(n) for n in nodes]

    def batch_add_edges(self, edges: list[KGEdge]) -> list[str]:
        return [self.add_edge(e) for e in edges]

    # ═══════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _find_existing_node(self, name: str, node_type: NodeType) -> str | None:
        """Check if a node with the same name and type already exists."""
        node_id = self._name_index.get(name.lower())
        if node_id and node_id in self._nodes and self._nodes[node_id].type == node_type:
            return node_id
        return None

    def _merge_node(self, existing_id: str, new_node: KGNode) -> None:
        """Merge incoming node data into an existing node."""
        existing = self._nodes[existing_id]

        # Merge aliases
        for alias in new_node.aliases:
            if alias not in existing.aliases:
                existing.aliases.append(alias)
                self._name_index[alias.lower()] = existing_id

        # Merge external_ids
        existing.external_ids.update(new_node.external_ids)

        # Merge properties
        existing.properties.update(new_node.properties)

        # Merge sources (dedup by source_id)
        existing_source_ids = {s.source_id for s in existing.sources if s.source_id}
        for source in new_node.sources:
            if source.source_id and source.source_id in existing_source_ids:
                continue
            existing.sources.append(source)

        # Update confidence to max
        existing.confidence = max(existing.confidence, new_node.confidence)
        existing.updated_at = datetime.now(UTC)

    def _detect_contradictions(self, edge: KGEdge) -> list[KGEdge]:
        """Find existing edges that contradict the given edge."""
        contradicting: list[KGEdge] = []

        opposite = _CONTRADICTION_PAIRS.get(edge.relation)
        if not opposite:
            return []

        # Check same pair in same direction
        for eid in self._edge_pair_index.get((edge.source_id, edge.target_id), []):
            existing = self._edges.get(eid)
            if existing and existing.relation == opposite and not existing.falsified:
                contradicting.append(existing)

        # Check same pair in reverse direction for symmetric contradictions
        for eid in self._edge_pair_index.get((edge.target_id, edge.source_id), []):
            existing = self._edges.get(eid)
            if existing and existing.relation == opposite and not existing.falsified:
                contradicting.append(existing)

        return contradicting

    def _traverse_directed(self, node_id: str, depth: int, direction: str) -> dict[str, Any]:
        """BFS traversal in a single direction (upstream = follow incoming, downstream = follow outgoing)."""
        visited_nodes: set[str] = set()
        visited_edges: set[str] = set()
        frontier = {node_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited_nodes:
                    continue
                visited_nodes.add(nid)
                edge_index = self._incoming if direction == "upstream" else self._outgoing
                for eid in edge_index.get(nid, []):
                    edge = self._edges.get(eid)
                    if edge:
                        visited_edges.add(eid)
                        neighbor = edge.source_id if direction == "upstream" else edge.target_id
                        next_frontier.add(neighbor)
            frontier = next_frontier - visited_nodes

        visited_nodes |= frontier

        return {
            "nodes": [self._nodes[nid].model_dump(mode="json") for nid in visited_nodes if nid in self._nodes],
            "edges": [self._edges[eid].model_dump(mode="json") for eid in visited_edges if eid in self._edges],
        }

    def _compute_clusters(self) -> dict[str, str]:
        """Auto-cluster nodes by type and connectivity using label propagation."""
        if not self._nodes:
            return {}

        # Start with type-based clusters
        labels: dict[str, str] = {}
        for nid, node in self._nodes.items():
            labels[nid] = f"type_{node.type.value}"

        # Refine with a few rounds of label propagation on the undirected graph
        for _ in range(3):
            new_labels: dict[str, str] = {}
            for nid in self._nodes:
                neighbor_labels: dict[str, int] = defaultdict(int)
                neighbor_labels[labels[nid]] = 1  # self-vote

                for eid in self._outgoing.get(nid, []):
                    edge = self._edges.get(eid)
                    if edge and edge.target_id in labels:
                        neighbor_labels[labels[edge.target_id]] += 1
                for eid in self._incoming.get(nid, []):
                    edge = self._edges.get(eid)
                    if edge and edge.source_id in labels:
                        neighbor_labels[labels[edge.source_id]] += 1

                new_labels[nid] = max(neighbor_labels, key=neighbor_labels.get)  # type: ignore[arg-type]
            labels = new_labels

        return labels

    def _load_from_dict(self, data: dict[str, Any]) -> None:
        """Restore graph state from a JSON dict."""
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._name_index.clear()
            self._type_index.clear()
            self._outgoing.clear()
            self._incoming.clear()
            self._hypothesis_edges.clear()
            self._edge_pair_index.clear()

            for node_data in data.get("nodes", []):
                node = KGNode.model_validate(node_data)
                self._nodes[node.id] = node
                self._name_index[node.name.lower()] = node.id
                for alias in node.aliases:
                    self._name_index[alias.lower()] = node.id
                self._type_index[node.type].add(node.id)

            for edge_data in data.get("edges", []):
                edge = KGEdge.model_validate(edge_data)
                self._edges[edge.id] = edge
                self._outgoing[edge.source_id].append(edge.id)
                self._incoming[edge.target_id].append(edge.id)
                self._edge_pair_index[(edge.source_id, edge.target_id)].append(edge.id)
                if edge.hypothesis_branch:
                    self._hypothesis_edges[edge.hypothesis_branch].append(edge.id)

            if graph_id := data.get("graph_id"):
                self.graph_id = graph_id
