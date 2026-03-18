"""MCTS Hypothesis Tree — UCB1 selection, expansion, backpropagation, pruning.

The tree models scientific hypotheses as MCTS nodes. Each iteration:
1. SELECT  — walk from root to a leaf using UCB1
2. EXPAND  — LLM generates child hypotheses
3. SIMULATE — agents explore the hypothesis (external)
4. BACKPROPAGATE — propagate information gain up the tree

Optimized for LAB-Bench: generates competing hypotheses covering different
reasoning paths and doesn't stop at first confident answer.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import structlog

from core.constants import DEFAULT_UCB_EXPLORATION_CONSTANT
from core.exceptions import OrchestrationError
from core.models import (
    HypothesisNode,
    HypothesisStatus,
    ResearchEvent,
    _uuid,
)

logger = structlog.get_logger(__name__)

# Evidence quality weights for info-gain calculation
EVIDENCE_TYPE_WEIGHTS: dict[str, float] = {
    "rct": 1.0,
    "meta_analysis": 0.95,
    "systematic_review": 0.9,
    "clinical_trial": 0.85,
    "cohort_study": 0.7,
    "case_control": 0.6,
    "observational": 0.5,
    "in_vitro": 0.4,
    "computational": 0.3,
    "expert_opinion": 0.2,
}


class HypothesisTree:
    """Monte Carlo Tree Search over hypothesis space.

    Nodes are ``HypothesisNode`` instances. The tree supports:
    - UCB1 selection with configurable exploration constant
    - LLM-driven expansion (via external call)
    - Info-gain backpropagation with evidence quality weighting
    - Pruning of low-value branches
    - Rich event emission for frontend visualization
    """

    def __init__(
        self,
        *,
        tree_id: str | None = None,
        exploration_constant: float = DEFAULT_UCB_EXPLORATION_CONSTANT,
        max_depth: int = 5,
        max_breadth: int = 30,
        session_id: str = "",
    ) -> None:
        self.tree_id = tree_id or _uuid()
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.max_breadth = max_breadth
        self.session_id = session_id

        self._nodes: dict[str, HypothesisNode] = {}
        self._root_id: str | None = None
        self._total_visits: int = 0
        self._events: list[ResearchEvent] = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def set_root(self, hypothesis: str, rationale: str = "") -> HypothesisNode:
        """Create and set the root hypothesis node."""
        root = HypothesisNode(
            hypothesis=hypothesis,
            rationale=rationale,
            depth=0,
            status=HypothesisStatus.UNEXPLORED,
        )
        self._nodes[root.id] = root
        self._root_id = root.id
        self._emit("hypothesis_tree_initialized", node_id=root.id, hypothesis=hypothesis)
        return root

    @property
    def root(self) -> HypothesisNode | None:
        if self._root_id is None:
            return None
        return self._nodes.get(self._root_id)

    @property
    def total_visits(self) -> int:
        return self._total_visits

    # ------------------------------------------------------------------
    # Node access
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> HypothesisNode | None:
        return self._nodes.get(node_id)

    def get_children(self, node_id: str) -> list[HypothesisNode]:
        node = self._nodes.get(node_id)
        if node is None:
            return []
        return [self._nodes[cid] for cid in node.children if cid in self._nodes]

    @property
    def all_nodes(self) -> list[HypothesisNode]:
        return list(self._nodes.values())

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------
    # UCB1 Selection
    # ------------------------------------------------------------------

    def _ucb1(self, node: HypothesisNode, parent_visits: int) -> float:
        """Compute UCB1 score for a node."""
        if node.visit_count == 0:
            return float("inf")  # always explore unvisited

        exploitation = node.avg_info_gain
        exploration = self.exploration_constant * math.sqrt(
            math.log(parent_visits) / node.visit_count
        )
        return exploitation + exploration

    def select(self) -> HypothesisNode:
        """Select the most promising leaf node via UCB1 tree-walk.

        Walks from root to a leaf, at each level picking the child with
        the highest UCB1 score. Returns the selected leaf for expansion
        or simulation.
        """
        if self._root_id is None:
            raise OrchestrationError("Cannot select: tree has no root")

        current = self._nodes[self._root_id]

        while current.children:
            # Only consider expandable children
            candidates = []
            for child_id in current.children:
                child = self._nodes.get(child_id)
                if child is None:
                    continue
                if child.status in (HypothesisStatus.PRUNED, HypothesisStatus.REFUTED):
                    continue
                candidates.append(child)

            if not candidates:
                break  # all children pruned/refuted — return current

            parent_visits = max(current.visit_count, 1)
            best = max(candidates, key=lambda c: self._ucb1(c, parent_visits))

            # Update UCB scores for all candidates (for visualization)
            for c in candidates:
                c.ucb_score = self._ucb1(c, parent_visits)
                c.updated_at = datetime.now(UTC)

            current = best

        current.status = HypothesisStatus.EXPLORING
        current.updated_at = datetime.now(UTC)
        self._emit(
            "hypothesis_selected",
            node_id=current.id,
            hypothesis=current.hypothesis,
            depth=current.depth,
            ucb_score=current.ucb_score,
            visit_count=current.visit_count,
        )
        return current

    def select_leaves(self, max_leaves: int = 5) -> list[HypothesisNode]:
        """Select multiple promising leaf nodes for parallel exploration.

        Returns up to *max_leaves* distinct unexplored/explored leaves,
        ranked by UCB1. Each selected leaf is marked EXPLORING.
        """
        if self._root_id is None:
            raise OrchestrationError("Cannot select: tree has no root")

        # Gather all candidate leaves
        candidates: list[HypothesisNode] = []
        for node in self._nodes.values():
            if node.status in (HypothesisStatus.PRUNED, HypothesisStatus.REFUTED):
                continue
            if node.id == self._root_id:
                continue
            # A leaf is a node with no expandable children
            expandable_children = [
                cid for cid in node.children
                if cid in self._nodes
                and self._nodes[cid].status not in (
                    HypothesisStatus.PRUNED, HypothesisStatus.REFUTED,
                )
            ]
            if not expandable_children:
                candidates.append(node)

        if not candidates:
            # Return least-visited leaf nodes instead of root
            all_leaves = [
                n for n in self._nodes.values()
                if n.id != self._root_id
                and n.status not in (HypothesisStatus.PRUNED, HypothesisStatus.REFUTED)
            ]
            if all_leaves:
                all_leaves.sort(key=lambda n: n.visit_count)
                selected_fallback = all_leaves[:max_leaves]
                for node in selected_fallback:
                    node.status = HypothesisStatus.EXPLORING
                    node.updated_at = datetime.now(UTC)
                return selected_fallback
            # Absolute fallback: use regular select (may return root)
            return [self.select()]

        # Score by UCB1 using root visit count as parent proxy
        parent_visits = max(self._total_visits, 1)
        candidates.sort(key=lambda c: self._ucb1(c, parent_visits), reverse=True)

        selected = candidates[:max_leaves]
        for node in selected:
            node.status = HypothesisStatus.EXPLORING
            node.updated_at = datetime.now(UTC)
            self._emit(
                "hypothesis_selected",
                node_id=node.id,
                hypothesis=node.hypothesis,
                depth=node.depth,
                ucb_score=self._ucb1(node, parent_visits),
                visit_count=node.visit_count,
                batch_size=len(selected),
            )

        return selected

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def expand(
        self,
        parent_id: str,
        child_hypotheses: list[dict[str, str]],
    ) -> list[HypothesisNode]:
        """Expand a node with child hypotheses.

        Args:
            parent_id: ID of the node to expand.
            child_hypotheses: List of dicts with 'hypothesis' and 'rationale'.

        Returns:
            List of newly created child nodes.
        """
        parent = self._nodes.get(parent_id)
        if parent is None:
            raise OrchestrationError(
                f"Cannot expand: node {parent_id} not found",
                error_code="NODE_NOT_FOUND",
            )

        if parent.depth >= self.max_depth:
            logger.info("expansion_skipped_max_depth", node_id=parent_id, depth=parent.depth)
            return []

        # Enforce breadth limit — cap children per node
        existing_children = len(parent.children)
        slots = max(0, self.max_breadth - existing_children)
        if slots == 0:
            logger.info(
                "expansion_skipped_max_breadth",
                node_id=parent_id,
                existing=existing_children,
                max_breadth=self.max_breadth,
            )
            return []
        child_hypotheses = child_hypotheses[:slots]

        children: list[HypothesisNode] = []
        for h in child_hypotheses:
            child = HypothesisNode(
                parent_id=parent_id,
                hypothesis=h["hypothesis"],
                rationale=h.get("rationale", ""),
                depth=parent.depth + 1,
                status=HypothesisStatus.UNEXPLORED,
            )
            self._nodes[child.id] = child
            parent.children.append(child.id)
            children.append(child)

            self._emit(
                "hypothesis_expanded",
                parent_id=parent_id,
                child_id=child.id,
                hypothesis=child.hypothesis,
                depth=child.depth,
                parent_hypothesis=parent.hypothesis,
            )

        parent.updated_at = datetime.now(UTC)
        return children

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def backpropagate(
        self,
        node_id: str,
        info_gain: float,
        *,
        edges_added: int = 0,
        edges_falsified: int = 0,
        contradictions_found: int = 0,
        confidence_delta: float = 0.0,
    ) -> None:
        """Backpropagate information gain from a leaf up to the root.

        Info gain is a composite of:
        - New edges added (weighted by evidence quality)
        - Confidence changes (both up and down are informative)
        - Contradictions found (valuable for falsification)
        - Falsified edges (shows the system is self-correcting)
        """
        node = self._nodes.get(node_id)
        if node is None:
            raise OrchestrationError(f"Cannot backpropagate: node {node_id} not found")

        self._total_visits += 1

        # Walk up to root
        current: HypothesisNode | None = node
        while current is not None:
            current.visit_count += 1
            current.total_info_gain += info_gain
            current.avg_info_gain = current.total_info_gain / current.visit_count

            # Update status
            if current.status == HypothesisStatus.EXPLORING:
                current.status = HypothesisStatus.EXPLORED

            current.updated_at = datetime.now(UTC)

            self._emit(
                "hypothesis_backpropagated",
                node_id=current.id,
                info_gain=info_gain,
                visit_count=current.visit_count,
                avg_info_gain=current.avg_info_gain,
                edges_added=edges_added,
            )

            # Move to parent
            if current.parent_id:
                current = self._nodes.get(current.parent_id)
            else:
                current = None

    # ------------------------------------------------------------------
    # Info Gain Computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_info_gain(
        *,
        edges_added: int = 0,
        edges_falsified: int = 0,
        contradictions_found: int = 0,
        avg_confidence_delta: float = 0.0,
        avg_evidence_quality: float = 0.5,
    ) -> float:
        """Compute information gain for a single MCTS iteration.

        Weights:
        - New edges: 0.3 * count * evidence_quality
        - Falsified edges: 0.25 * count (falsification is highly informative)
        - Contradictions: 0.25 * count (contradictions reveal complexity)
        - Confidence changes: 0.2 * |delta| (any change is informative)

        Returns a float in [0, ∞), typically [0, 5].
        """
        edge_gain = 0.3 * edges_added * avg_evidence_quality
        falsification_gain = 0.25 * edges_falsified
        contradiction_gain = 0.25 * contradictions_found
        confidence_gain = 0.2 * abs(avg_confidence_delta) * 10  # scale up

        return edge_gain + falsification_gain + contradiction_gain + confidence_gain

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune(self, node_id: str, reason: str = "") -> int:
        """Prune a node and all its descendants. Returns count of pruned nodes."""
        node = self._nodes.get(node_id)
        if node is None:
            return 0

        pruned = 0

        def _prune_recursive(nid: str) -> None:
            nonlocal pruned
            n = self._nodes.get(nid)
            if n is None:
                return
            for child_id in n.children:
                _prune_recursive(child_id)
            n.status = HypothesisStatus.PRUNED
            n.updated_at = datetime.now(UTC)
            pruned += 1

        _prune_recursive(node_id)

        self._emit(
            "hypothesis_pruned",
            node_id=node_id,
            hypothesis=node.hypothesis,
            reason=reason,
            pruned_count=pruned,
        )
        return pruned

    def auto_prune(self, min_visits: int = 3, min_avg_gain: float = 0.1) -> int:
        """Prune branches that consistently show low info gain after sufficient visits."""
        total_pruned = 0
        for node in list(self._nodes.values()):
            if node.status == HypothesisStatus.PRUNED:
                continue
            # Never prune the root node — it accumulates visits from all
            # children's backpropagation and would be incorrectly pruned first
            if node.id == self._root_id:
                continue
            if node.visit_count >= min_visits and node.avg_info_gain < min_avg_gain:
                count = self.prune(
                    node.id,
                    reason=f"Low avg info gain ({node.avg_info_gain:.3f}) after {node.visit_count} visits",
                )
                total_pruned += count
        return total_pruned

    # ------------------------------------------------------------------
    # Confirmation / Refutation
    # ------------------------------------------------------------------

    def confirm(self, node_id: str, confidence: float) -> None:
        """Mark a hypothesis as confirmed with given confidence."""
        node = self._nodes.get(node_id)
        if node is None:
            return
        node.status = HypothesisStatus.CONFIRMED
        node.confidence = confidence
        node.updated_at = datetime.now(UTC)
        self._emit(
            "hypothesis_confirmed",
            node_id=node_id,
            hypothesis=node.hypothesis,
            confidence=confidence,
        )

    def refute(self, node_id: str, reason: str = "") -> None:
        """Mark a hypothesis as refuted."""
        node = self._nodes.get(node_id)
        if node is None:
            return
        node.status = HypothesisStatus.REFUTED
        node.confidence = 0.0
        node.updated_at = datetime.now(UTC)
        self._emit(
            "hypothesis_refuted",
            node_id=node_id,
            hypothesis=node.hypothesis,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Ranking / Best hypothesis
    # ------------------------------------------------------------------

    def get_best_hypothesis(self) -> HypothesisNode | None:
        """Return the hypothesis with highest avg info gain among visited non-root nodes."""
        candidates = [
            n for n in self._nodes.values()
            if n.visit_count > 0
            and n.status not in (HypothesisStatus.PRUNED, HypothesisStatus.REFUTED)
            and n.id != self._root_id  # exclude root — it's a container, not a real hypothesis
        ]
        if not candidates:
            return self.root
        return max(candidates, key=lambda n: n.avg_info_gain)

    def get_ranking(self, top_k: int = 10) -> list[HypothesisNode]:
        """Return hypotheses ranked by avg info gain (explored only)."""
        candidates = [
            n for n in self._nodes.values()
            if n.status not in (HypothesisStatus.PRUNED, HypothesisStatus.REFUTED)
            and n.visit_count > 0
        ]
        candidates.sort(key=lambda n: n.avg_info_gain, reverse=True)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Termination check
    # ------------------------------------------------------------------

    def should_terminate(
        self,
        *,
        confidence_threshold: float = 0.7,
        max_iterations: int = 15,
        current_iteration: int = 0,
    ) -> tuple[bool, str]:
        """Check if the MCTS loop should terminate.

        Returns (should_stop, reason).

        Does NOT stop at first confident answer — checks for alternative paths
        that might contradict it (per LAB-Bench optimization).
        """
        # Budget exhausted
        if current_iteration >= max_iterations:
            return True, f"max_iterations_reached ({max_iterations})"

        # All nodes pruned or terminal
        active = [
            n for n in self._nodes.values()
            if n.status not in (HypothesisStatus.PRUNED, HypothesisStatus.REFUTED)
        ]
        unexplored = [n for n in active if n.status == HypothesisStatus.UNEXPLORED]

        if not active:
            return True, "all_hypotheses_pruned_or_refuted"

        # Check for high-confidence confirmed hypothesis
        confirmed = [
            n for n in self._nodes.values()
            if n.status == HypothesisStatus.CONFIRMED
            and n.confidence >= confidence_threshold
        ]

        if confirmed and not unexplored:
            # Only terminate if no unexplored alternatives remain
            return True, "confident_hypothesis_found_no_alternatives"

        # If we have a confident answer but unexplored paths exist,
        # keep exploring to find potential contradictions (LAB-Bench optimization)
        if confirmed and unexplored:
            return False, "exploring_alternatives_for_robustness"

        return False, "exploring"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tree for API responses / persistence."""
        return {
            "tree_id": self.tree_id,
            "session_id": self.session_id,
            "root_id": self._root_id,
            "total_visits": self._total_visits,
            "node_count": len(self._nodes),
            "nodes": {nid: n.model_dump(mode="json") for nid, n in self._nodes.items()},
        }

    def get_exploration_path(self, node_id: str) -> list[HypothesisNode]:
        """Get the path from root to the given node."""
        path: list[HypothesisNode] = []
        current = self._nodes.get(node_id)
        while current is not None:
            path.append(current)
            if current.parent_id:
                current = self._nodes.get(current.parent_id)
            else:
                current = None
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, **data: Any) -> None:
        event = ResearchEvent(
            session_id=self.session_id,
            event_type=event_type,
            data=data,
        )
        self._events.append(event)
        logger.info(event_type, session_id=self.session_id, **data)

    def drain_events(self) -> list[ResearchEvent]:
        """Return and clear pending events."""
        events = self._events[:]
        self._events.clear()
        return events
