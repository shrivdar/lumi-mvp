"""Uncertainty aggregation and HITL (Human-in-the-Loop) triggering.

Aggregates UncertaintyVector values from multiple agents, decides whether
human input is needed, and manages Slack-based HITL interactions.
"""

from __future__ import annotations

from typing import Any

import structlog

from core.audit import AuditLogger
from core.models import (
    AgentResult,
    ResearchConfig,
    ResearchEvent,
    UncertaintyVector,
)

logger = structlog.get_logger(__name__)


class UncertaintyAggregator:
    """Aggregates uncertainty across agents and triggers HITL when needed."""

    def __init__(self, *, session_id: str = "") -> None:
        self.session_id = session_id
        self.audit = AuditLogger("uncertainty")
        self._history: list[UncertaintyVector] = []
        self._hitl_triggered: bool = False
        self._hitl_responses: list[dict[str, Any]] = []
        self._events: list[ResearchEvent] = []

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self, results: list[AgentResult]) -> UncertaintyVector:
        """Aggregate uncertainty vectors from multiple agent results.

        Uses weighted average where agents with more data (edges added)
        get higher weight — they've seen more of the problem.
        """
        if not results:
            return UncertaintyVector()

        vectors = [r.uncertainty for r in results if r.success]
        if not vectors:
            return UncertaintyVector()

        # Weight by number of edges added (more data → more informative)
        weights = []
        for r in results:
            if r.success:
                w = max(1.0, len(r.edges_added) + len(r.edges_updated) * 0.5)
                weights.append(w)

        total_weight = sum(weights)

        agg = UncertaintyVector(
            input_ambiguity=sum(v.input_ambiguity * w for v, w in zip(vectors, weights)) / total_weight,
            data_quality=sum(v.data_quality * w for v, w in zip(vectors, weights)) / total_weight,
            reasoning_divergence=self._compute_divergence(vectors),
            model_disagreement=sum(v.model_disagreement * w for v, w in zip(vectors, weights)) / total_weight,
            conflict_uncertainty=max(v.conflict_uncertainty for v in vectors),  # worst case
            novelty_uncertainty=sum(v.novelty_uncertainty * w for v, w in zip(vectors, weights)) / total_weight,
        )

        agg.compute_composite()
        agg.is_critical = agg.composite > 0.6

        self._history.append(agg)

        self.audit.log(
            "uncertainty_aggregated",
            session_id=self.session_id,
            composite=agg.composite,
            is_critical=agg.is_critical,
            agent_count=len(vectors),
        )

        self._emit(
            "uncertainty_computed",
            composite=agg.composite,
            is_critical=agg.is_critical,
            input_ambiguity=agg.input_ambiguity,
            data_quality=agg.data_quality,
            reasoning_divergence=agg.reasoning_divergence,
            conflict_uncertainty=agg.conflict_uncertainty,
        )

        return agg

    def _compute_divergence(self, vectors: list[UncertaintyVector]) -> float:
        """Measure how much agents disagree with each other.

        Uses variance of data_quality and conflict scores as a proxy.
        """
        if len(vectors) < 2:
            return 0.0

        qualities = [v.data_quality for v in vectors]
        mean_q = sum(qualities) / len(qualities)
        variance = sum((q - mean_q) ** 2 for q in qualities) / len(qualities)

        # Normalize: sqrt(variance) ranges [0, 0.5] for [0, 1] bounded values
        return min(1.0, variance ** 0.5 * 2)

    # ------------------------------------------------------------------
    # HITL Decision
    # ------------------------------------------------------------------

    def should_trigger_hitl(
        self,
        uncertainty: UncertaintyVector,
        config: ResearchConfig,
    ) -> tuple[bool, str]:
        """Decide whether to trigger HITL based on uncertainty.

        Returns (should_trigger, reason).
        """
        if not config.enable_hitl:
            return False, "hitl_disabled"

        if self._hitl_triggered:
            return False, "hitl_already_triggered_this_session"

        if uncertainty.composite >= config.hitl_uncertainty_threshold:
            reasons: list[str] = []
            if uncertainty.conflict_uncertainty > 0.5:
                reasons.append("high_conflict_between_evidence")
            if uncertainty.data_quality > 0.6:
                reasons.append("low_data_quality")
            if uncertainty.reasoning_divergence > 0.5:
                reasons.append("agents_disagree")
            if uncertainty.input_ambiguity > 0.5:
                reasons.append("ambiguous_input")

            reason = "; ".join(reasons) if reasons else "composite_uncertainty_exceeds_threshold"

            self.audit.log(
                "hitl_trigger_recommended",
                session_id=self.session_id,
                composite=uncertainty.composite,
                threshold=config.hitl_uncertainty_threshold,
                reason=reason,
            )
            return True, reason

        return False, "uncertainty_within_bounds"

    def mark_hitl_triggered(self) -> None:
        self._hitl_triggered = True
        self._emit("hitl_triggered")

    def record_hitl_response(self, response: dict[str, Any]) -> None:
        self._hitl_responses.append(response)
        self._hitl_triggered = False  # allow re-trigger if uncertainty persists
        self._emit("hitl_response_received", response_preview=str(response)[:200])

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------

    def get_trend(self) -> dict[str, Any]:
        """Return uncertainty trend over iterations."""
        if not self._history:
            return {"trend": "no_data", "composites": []}

        composites = [v.composite for v in self._history]

        if len(composites) >= 2:
            recent = composites[-3:]
            if all(recent[i] <= recent[i - 1] for i in range(1, len(recent))):
                trend = "decreasing"
            elif all(recent[i] >= recent[i - 1] for i in range(1, len(recent))):
                trend = "increasing"
            else:
                trend = "fluctuating"
        else:
            trend = "insufficient_data"

        return {
            "trend": trend,
            "composites": composites,
            "latest": composites[-1],
            "mean": sum(composites) / len(composites),
            "hitl_triggered": self._hitl_triggered,
            "hitl_response_count": len(self._hitl_responses),
        }

    # ------------------------------------------------------------------
    # HITL message formatting
    # ------------------------------------------------------------------

    def format_hitl_message(
        self,
        query: str,
        hypothesis: str,
        uncertainty: UncertaintyVector,
        reason: str,
    ) -> str:
        """Format a Slack message for HITL review."""
        return (
            f":warning: *YOHAS HITL Request*\n\n"
            f"*Research Query:* {query}\n"
            f"*Current Hypothesis:* {hypothesis}\n\n"
            f"*Uncertainty Score:* {uncertainty.composite:.2f}\n"
            f"*Reason:* {reason}\n\n"
            f"*Breakdown:*\n"
            f"  - Input Ambiguity: {uncertainty.input_ambiguity:.2f}\n"
            f"  - Data Quality: {uncertainty.data_quality:.2f}\n"
            f"  - Reasoning Divergence: {uncertainty.reasoning_divergence:.2f}\n"
            f"  - Conflict: {uncertainty.conflict_uncertainty:.2f}\n\n"
            f"Please reply with guidance on how to proceed."
        )

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

    def drain_events(self) -> list[ResearchEvent]:
        events = self._events[:]
        self._events.clear()
        return events
