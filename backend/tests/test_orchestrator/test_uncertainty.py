"""Tests for uncertainty aggregation and HITL triggering."""

from __future__ import annotations

import pytest

from core.models import (
    AgentResult,
    AgentType,
    EdgeRelationType,
    KGEdge,
    ResearchConfig,
    UncertaintyVector,
)
from orchestrator.uncertainty import UncertaintyAggregator


@pytest.fixture()
def aggregator() -> UncertaintyAggregator:
    return UncertaintyAggregator(session_id="test-session")


def _make_result(
    *,
    data_quality: float = 0.3,
    conflict: float = 0.1,
    edges: int = 2,
    success: bool = True,
) -> AgentResult:
    return AgentResult(
        task_id="t1",
        agent_id="a1",
        agent_type=AgentType.LITERATURE_ANALYST,
        success=success,
        edges_added=[
            KGEdge(source_id="n1", target_id="n2", relation=EdgeRelationType.ASSOCIATED_WITH)
            for _ in range(edges)
        ],
        uncertainty=UncertaintyVector(
            input_ambiguity=0.2,
            data_quality=data_quality,
            reasoning_divergence=0.1,
            model_disagreement=0.1,
            conflict_uncertainty=conflict,
            novelty_uncertainty=0.1,
            composite=0.3,
        ),
    )


class TestAggregation:
    def test_aggregate_single(self, aggregator: UncertaintyAggregator) -> None:
        results = [_make_result(data_quality=0.4)]
        agg = aggregator.aggregate(results)
        assert 0 < agg.composite < 1.0
        assert agg.data_quality == pytest.approx(0.4, abs=0.05)

    def test_aggregate_multiple(self, aggregator: UncertaintyAggregator) -> None:
        results = [
            _make_result(data_quality=0.2, edges=5),
            _make_result(data_quality=0.8, edges=1),
        ]
        agg = aggregator.aggregate(results)
        # Should be weighted toward first result (more edges)
        assert agg.data_quality < 0.5

    def test_aggregate_empty(self, aggregator: UncertaintyAggregator) -> None:
        agg = aggregator.aggregate([])
        assert agg.composite == 0.0

    def test_aggregate_failed_results_excluded(self, aggregator: UncertaintyAggregator) -> None:
        results = [
            _make_result(success=True, data_quality=0.3),
            _make_result(success=False, data_quality=0.9),
        ]
        agg = aggregator.aggregate(results)
        # Only the successful result should count
        assert agg.data_quality == pytest.approx(0.3, abs=0.05)

    def test_conflict_uses_max(self, aggregator: UncertaintyAggregator) -> None:
        results = [
            _make_result(conflict=0.2),
            _make_result(conflict=0.8),
        ]
        agg = aggregator.aggregate(results)
        assert agg.conflict_uncertainty == 0.8  # worst case

    def test_critical_flag(self, aggregator: UncertaintyAggregator) -> None:
        # Use uniformly high uncertainty to guarantee composite > 0.6
        result = AgentResult(
            task_id="t1",
            agent_id="a1",
            agent_type=AgentType.LITERATURE_ANALYST,
            success=True,
            edges_added=[
                KGEdge(source_id="n1", target_id="n2", relation=EdgeRelationType.ASSOCIATED_WITH)
            ],
            uncertainty=UncertaintyVector(
                input_ambiguity=0.9,
                data_quality=0.9,
                reasoning_divergence=0.9,
                model_disagreement=0.9,
                conflict_uncertainty=0.9,
                novelty_uncertainty=0.9,
                composite=0.9,
            ),
        )
        agg = aggregator.aggregate([result])
        assert agg.is_critical
        assert agg.composite > 0.6


class TestDivergence:
    def test_low_divergence_when_agents_agree(self, aggregator: UncertaintyAggregator) -> None:
        results = [
            _make_result(data_quality=0.3),
            _make_result(data_quality=0.3),
        ]
        agg = aggregator.aggregate(results)
        assert agg.reasoning_divergence < 0.1

    def test_high_divergence_when_agents_disagree(self, aggregator: UncertaintyAggregator) -> None:
        results = [
            _make_result(data_quality=0.1),
            _make_result(data_quality=0.9),
        ]
        agg = aggregator.aggregate(results)
        assert agg.reasoning_divergence > 0.3


class TestHITLDecision:
    def test_trigger_when_above_threshold(self, aggregator: UncertaintyAggregator) -> None:
        config = ResearchConfig(hitl_uncertainty_threshold=0.3, enable_hitl=True)
        uncertainty = UncertaintyVector(composite=0.7, conflict_uncertainty=0.6)
        should_trigger, reason = aggregator.should_trigger_hitl(uncertainty, config)
        assert should_trigger
        assert "conflict" in reason or "threshold" in reason

    def test_no_trigger_below_threshold(self, aggregator: UncertaintyAggregator) -> None:
        config = ResearchConfig(hitl_uncertainty_threshold=0.8, enable_hitl=True)
        uncertainty = UncertaintyVector(composite=0.3)
        should_trigger, _ = aggregator.should_trigger_hitl(uncertainty, config)
        assert not should_trigger

    def test_no_trigger_when_disabled(self, aggregator: UncertaintyAggregator) -> None:
        config = ResearchConfig(enable_hitl=False)
        uncertainty = UncertaintyVector(composite=0.9)
        should_trigger, reason = aggregator.should_trigger_hitl(uncertainty, config)
        assert not should_trigger
        assert "disabled" in reason

    def test_no_double_trigger(self, aggregator: UncertaintyAggregator) -> None:
        config = ResearchConfig(hitl_uncertainty_threshold=0.3, enable_hitl=True)
        uncertainty = UncertaintyVector(composite=0.7)

        aggregator.mark_hitl_triggered()
        should_trigger, reason = aggregator.should_trigger_hitl(uncertainty, config)
        assert not should_trigger
        assert "already_triggered" in reason

    def test_re_trigger_after_response(self, aggregator: UncertaintyAggregator) -> None:
        config = ResearchConfig(hitl_uncertainty_threshold=0.3, enable_hitl=True)
        uncertainty = UncertaintyVector(composite=0.7)

        aggregator.mark_hitl_triggered()
        aggregator.record_hitl_response({"text": "proceed"})

        should_trigger, _ = aggregator.should_trigger_hitl(uncertainty, config)
        assert should_trigger  # can re-trigger after response received


class TestTrend:
    def test_no_data(self, aggregator: UncertaintyAggregator) -> None:
        trend = aggregator.get_trend()
        assert trend["trend"] == "no_data"

    def test_decreasing_trend(self, aggregator: UncertaintyAggregator) -> None:
        aggregator._history = [
            UncertaintyVector(composite=0.8),
            UncertaintyVector(composite=0.6),
            UncertaintyVector(composite=0.4),
        ]
        trend = aggregator.get_trend()
        assert trend["trend"] == "decreasing"

    def test_increasing_trend(self, aggregator: UncertaintyAggregator) -> None:
        aggregator._history = [
            UncertaintyVector(composite=0.2),
            UncertaintyVector(composite=0.4),
            UncertaintyVector(composite=0.6),
        ]
        trend = aggregator.get_trend()
        assert trend["trend"] == "increasing"


class TestHITLMessage:
    def test_format_message(self, aggregator: UncertaintyAggregator) -> None:
        uncertainty = UncertaintyVector(
            composite=0.75,
            input_ambiguity=0.3,
            data_quality=0.6,
            reasoning_divergence=0.4,
            conflict_uncertainty=0.8,
        )
        msg = aggregator.format_hitl_message(
            "B7-H3 NSCLC", "B7-H3 promotes immune evasion", uncertainty, "high_conflict",
        )
        assert "B7-H3" in msg
        assert "0.75" in msg
        assert "high_conflict" in msg


class TestEvents:
    def test_aggregate_emits_event(self, aggregator: UncertaintyAggregator) -> None:
        results = [_make_result()]
        aggregator.aggregate(results)
        events = aggregator.drain_events()
        assert any(e.event_type == "uncertainty_computed" for e in events)
