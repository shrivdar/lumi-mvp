"""Living Document — auto-updated markdown research report driven by KG events.

Subscribes to KnowledgeGraph events (node_created, edge_created, edge_falsified, etc.)
and maintains a continuously-updated markdown document with structured sections:
Executive Summary, Hypotheses, Evidence Map, Findings, Contradictions, Uncertainties.
Tracks version history as diffs between updates.
"""

from __future__ import annotations

import difflib
import threading
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import structlog

from core.audit import AuditLogger
from core.models import KGEdge, KGNode, NodeType

# Match the EventCallback type from knowledge_graph.py
EventCallback = Callable[[str, dict[str, Any]], None]

logger = structlog.get_logger(__name__)
audit = AuditLogger("living_document")


class DocumentVersion:
    """A single snapshot of the living document."""

    __slots__ = ("version", "content", "timestamp", "trigger_event", "diff")

    def __init__(
        self,
        version: int,
        content: str,
        timestamp: datetime,
        trigger_event: str,
        diff: str = "",
    ) -> None:
        self.version = version
        self.content = content
        self.timestamp = timestamp
        self.trigger_event = trigger_event
        self.diff = diff


class LivingDocument:
    """Continuously-updated research document that listens to KG mutations.

    Usage::

        kg = InMemoryKnowledgeGraph(graph_id="session-1")
        doc = LivingDocument(session_id="session-1", title="B7-H3 in NSCLC")
        doc.attach(kg)
        # ... agents mutate the KG ...
        print(doc.render())
    """

    def __init__(self, session_id: str, title: str = "Research Report") -> None:
        self.session_id = session_id
        self.title = title

        # Internal state rebuilt from KG events
        self._nodes: dict[str, KGNode] = {}
        self._edges: dict[str, KGEdge] = {}
        self._hypotheses: dict[str, list[str]] = defaultdict(list)  # branch → [edge_ids]
        self._contradictions: list[str] = []  # edge_ids
        self._falsified: list[str] = []  # edge_ids
        self._uncertainties: list[str] = []  # edge_ids with low confidence

        # Version history
        self._versions: list[DocumentVersion] = []
        self._current_content: str = ""

        # Thread safety
        self._lock = threading.Lock()

        # Stats
        self._update_count = 0

        # Reference to attached KG (for remove_listener)
        self._kg: Any = None
        self._callback: EventCallback | None = None

    # ── Listener attachment ────────────────────────────────────────────

    def attach(self, kg: Any) -> None:
        """Subscribe to KG events via add_listener."""
        self._kg = kg
        self._callback = self._on_kg_event
        kg.add_listener(self._callback)
        audit.log("living_doc_attached", session_id=self.session_id, graph_id=getattr(kg, "graph_id", ""))

    def detach(self) -> None:
        """Unsubscribe from KG events."""
        if self._kg is not None and self._callback is not None:
            self._kg.remove_listener(self._callback)
            self._kg = None
            self._callback = None
            audit.log("living_doc_detached", session_id=self.session_id)

    # ── KG event handler ──────────────────────────────────────────────

    def _on_kg_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Callback invoked by the KnowledgeGraph on mutations.

        The KG emits lightweight dicts with IDs (not full objects), so we
        look up the actual node/edge from the attached KG reference.
        """
        if self._kg is None:
            return

        with self._lock:
            if event_type == "node_created":
                node_id = data.get("node_id", "")
                node = self._kg.get_node(node_id)
                if node is not None:
                    self._nodes[node.id] = node
            elif event_type == "edge_created":
                edge_id = data.get("edge_id", "")
                edge = self._kg.get_edge(edge_id)
                if edge is not None:
                    self._edges[edge.id] = edge
                    branch = data.get("hypothesis_branch") or edge.hypothesis_branch
                    if branch:
                        self._hypotheses[branch].append(edge.id)
                    if data.get("is_contradiction") or edge.is_contradiction:
                        self._contradictions.append(edge.id)
                    if edge.confidence.overall < 0.5:
                        self._uncertainties.append(edge.id)
            elif event_type == "node_updated":
                node_id = data.get("node_id", "")
                node = self._kg.get_node(node_id)
                if node is not None:
                    self._nodes[node.id] = node
            elif event_type == "edge_confidence_updated":
                edge_id = data.get("edge_id", "")
                new_confidence = data.get("new_confidence")
                if edge_id in self._edges and new_confidence is not None:
                    self._edges[edge_id].confidence.overall = new_confidence
                    if new_confidence < 0.5 and edge_id not in self._uncertainties:
                        self._uncertainties.append(edge_id)
            elif event_type == "edge_falsified":
                edge_id = data.get("edge_id", "")
                if edge_id and edge_id not in self._falsified:
                    self._falsified.append(edge_id)
            else:
                return  # unknown event, skip rebuild

            self._rebuild(trigger_event=event_type)

    # ── Document rendering ────────────────────────────────────────────

    def _rebuild(self, trigger_event: str = "manual") -> None:
        """Regenerate the markdown document from current state."""
        new_content = self.render()
        if new_content == self._current_content:
            return  # no change

        # Compute diff against previous version
        diff = ""
        if self._current_content:
            diff = "\n".join(
                difflib.unified_diff(
                    self._current_content.splitlines(),
                    new_content.splitlines(),
                    fromfile=f"v{len(self._versions)}",
                    tofile=f"v{len(self._versions) + 1}",
                    lineterm="",
                )
            )

        version = DocumentVersion(
            version=len(self._versions) + 1,
            content=new_content,
            timestamp=datetime.now(UTC),
            trigger_event=trigger_event,
            diff=diff,
        )
        self._versions.append(version)
        self._current_content = new_content
        self._update_count += 1

        audit.log(
            "living_doc_updated",
            session_id=self.session_id,
            version=version.version,
            trigger=trigger_event,
            nodes=len(self._nodes),
            edges=len(self._edges),
        )

    def render(self) -> str:
        """Render the full living document as markdown."""
        sections: list[str] = []

        # Title
        sections.append(f"# {self.title}")
        sections.append("")
        sections.append(f"*Session:* `{self.session_id}`  ")
        sections.append(f"*Last updated:* {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
        sections.append(f"*Version:* {len(self._versions) + 1}")
        sections.append("")

        # Executive Summary
        sections.append("## Executive Summary")
        sections.append("")
        sections.append(self._render_executive_summary())
        sections.append("")

        # Hypotheses
        sections.append("## Hypotheses")
        sections.append("")
        sections.append(self._render_hypotheses())
        sections.append("")

        # Evidence Map
        sections.append("## Evidence Map")
        sections.append("")
        sections.append(self._render_evidence_map())
        sections.append("")

        # Key Findings
        sections.append("## Key Findings")
        sections.append("")
        sections.append(self._render_findings())
        sections.append("")

        # Contradictions
        sections.append("## Contradictions")
        sections.append("")
        sections.append(self._render_contradictions())
        sections.append("")

        # Uncertainties
        sections.append("## Uncertainties")
        sections.append("")
        sections.append(self._render_uncertainties())

        return "\n".join(sections)

    def _render_executive_summary(self) -> str:
        """High-level stats and overview."""
        node_types: dict[str, int] = defaultdict(int)
        for node in self._nodes.values():
            node_types[node.type.value if isinstance(node.type, NodeType) else str(node.type)] += 1

        lines = [
            f"- **Entities discovered:** {len(self._nodes)}",
            f"- **Relationships mapped:** {len(self._edges)}",
            f"- **Hypothesis branches:** {len(self._hypotheses)}",
            f"- **Contradictions found:** {len(self._contradictions)}",
            f"- **Falsified claims:** {len(self._falsified)}",
            f"- **Uncertain edges:** {len(self._uncertainties)}",
        ]

        if node_types:
            lines.append("")
            lines.append("**Entity breakdown:**")
            for ntype, count in sorted(node_types.items(), key=lambda x: -x[1]):
                lines.append(f"  - {ntype}: {count}")

        return "\n".join(lines)

    def _render_hypotheses(self) -> str:
        """List hypothesis branches and their associated edges."""
        if not self._hypotheses:
            return "*No hypotheses explored yet.*"

        lines: list[str] = []
        for branch, edge_ids in self._hypotheses.items():
            active = [eid for eid in edge_ids if eid not in self._falsified]
            falsified = [eid for eid in edge_ids if eid in self._falsified]
            lines.append(f"### Branch: `{branch}`")
            lines.append(f"- Active edges: {len(active)}")
            lines.append(f"- Falsified edges: {len(falsified)}")
            lines.append("")
        return "\n".join(lines)

    def _render_evidence_map(self) -> str:
        """Top edges by confidence with source/target names."""
        if not self._edges:
            return "*No evidence collected yet.*"

        sorted_edges = sorted(
            self._edges.values(),
            key=lambda e: e.confidence.overall,
            reverse=True,
        )[:15]

        lines: list[str] = []
        for edge in sorted_edges:
            src = self._nodes.get(edge.source_id)
            tgt = self._nodes.get(edge.target_id)
            src_name = src.name if src else edge.source_id
            tgt_name = tgt.name if tgt else edge.target_id
            status = ""
            if edge.id in self._falsified:
                status = " ~~FALSIFIED~~"
            elif edge.is_contradiction:
                status = " **CONTRADICTION**"
            lines.append(
                f"| {src_name} | {edge.relation.value} | {tgt_name} "
                f"| {edge.confidence.overall:.2f} | {edge.confidence.evidence_count}{status} |"
            )

        header = "| Source | Relation | Target | Confidence | Evidence Count |\n"
        header += "|--------|----------|--------|------------|----------------|\n"
        return header + "\n".join(lines)

    def _render_findings(self) -> str:
        """High-confidence, non-falsified edges as key findings."""
        findings = [
            e for e in self._edges.values()
            if e.confidence.overall >= 0.7
            and e.id not in self._falsified
            and not e.is_contradiction
        ]
        if not findings:
            return "*No high-confidence findings yet.*"

        findings.sort(key=lambda e: e.confidence.overall, reverse=True)
        lines: list[str] = []
        for edge in findings[:10]:
            src = self._nodes.get(edge.source_id)
            tgt = self._nodes.get(edge.target_id)
            src_name = src.name if src else edge.source_id
            tgt_name = tgt.name if tgt else edge.target_id
            lines.append(
                f"- **{src_name}** {edge.relation.value} **{tgt_name}** "
                f"(confidence: {edge.confidence.overall:.2f}, "
                f"evidence: {edge.confidence.evidence_count})"
            )
        return "\n".join(lines)

    def _render_contradictions(self) -> str:
        """List contradiction edges."""
        if not self._contradictions:
            return "*No contradictions detected.*"

        lines: list[str] = []
        for edge_id in self._contradictions:
            edge = self._edges.get(edge_id)
            if not edge:
                continue
            src = self._nodes.get(edge.source_id)
            tgt = self._nodes.get(edge.target_id)
            src_name = src.name if src else edge.source_id
            tgt_name = tgt.name if tgt else edge.target_id
            lines.append(
                f"- {src_name} --[{edge.relation.value}]--> {tgt_name} "
                f"(contradicts existing evidence)"
            )
        return "\n".join(lines) if lines else "*No contradictions detected.*"

    def _render_uncertainties(self) -> str:
        """List low-confidence edges that need more evidence."""
        if not self._uncertainties:
            return "*No uncertain edges.*"

        lines: list[str] = []
        for edge_id in self._uncertainties:
            edge = self._edges.get(edge_id)
            if not edge:
                continue
            if edge.id in self._falsified:
                continue  # already falsified, not uncertain
            src = self._nodes.get(edge.source_id)
            tgt = self._nodes.get(edge.target_id)
            src_name = src.name if src else edge.source_id
            tgt_name = tgt.name if tgt else edge.target_id
            lines.append(
                f"- {src_name} --[{edge.relation.value}]--> {tgt_name} "
                f"(confidence: {edge.confidence.overall:.2f} — needs more evidence)"
            )
        return "\n".join(lines) if lines else "*No uncertain edges.*"

    # ── Version history ───────────────────────────────────────────────

    @property
    def version_count(self) -> int:
        return len(self._versions)

    def get_version(self, version_num: int) -> DocumentVersion | None:
        """Get a specific version (1-indexed)."""
        idx = version_num - 1
        if 0 <= idx < len(self._versions):
            return self._versions[idx]
        return None

    def get_latest_version(self) -> DocumentVersion | None:
        """Get the most recent version."""
        return self._versions[-1] if self._versions else None

    def get_version_history(self) -> list[dict[str, Any]]:
        """Return version metadata without full content."""
        return [
            {
                "version": v.version,
                "timestamp": v.timestamp.isoformat(),
                "trigger_event": v.trigger_event,
                "has_diff": bool(v.diff),
            }
            for v in self._versions
        ]

    def get_diff(self, version_num: int) -> str:
        """Get the unified diff for a specific version."""
        v = self.get_version(version_num)
        return v.diff if v else ""

    @property
    def current_content(self) -> str:
        return self._current_content
