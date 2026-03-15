"""Tests for multi-dimensional reward function."""

from __future__ import annotations

import pytest

from core.models import (
    AgentResult,
    AgentType,
    EdgeConfidence,
    EdgeRelationType,
    EvidenceSource,
    EvidenceSourceType,
    FalsificationResult,
    KGEdge,
    KGNode,
    NodeType,
)
from rl.training.config import RewardWeights
from rl.training.reward import (
    RewardBreakdown,
    TrajectoryContext,
    compute_batch_rewards,
    compute_reward,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    *,
    nodes: list[KGNode] | None = None,
    edges: list[KGEdge] | None = None,
    falsifications: list[FalsificationResult] | None = None,
    summary: str = "",
    reasoning_trace: str = "",
    success: bool = True,
) -> AgentResult:
    return AgentResult(
        task_id="test-task",
        agent_id="test-agent",
        agent_type=AgentType.LITERATURE_ANALYST,
        nodes_added=nodes or [],
        edges_added=edges or [],
        falsification_results=falsifications or [],
        summary=summary,
        reasoning_trace=reasoning_trace,
        success=success,
    )


def _make_node(confidence: float = 0.8) -> KGNode:
    return KGNode(
        type=NodeType.GENE,
        name="BRCA1",
        confidence=confidence,
        created_by="test-agent",
        hypothesis_branch="h-test",
        sources=[
            EvidenceSource(
                source_type=EvidenceSourceType.PUBMED,
                source_id="PMID:12345",
                quality_score=0.9,
            )
        ],
    )


def _make_edge(
    confidence: float = 0.85,
    with_evidence: bool = True,
    is_contradiction: bool = False,
) -> KGEdge:
    return KGEdge(
        source_id="n1",
        target_id="n2",
        relation=EdgeRelationType.ASSOCIATED_WITH,
        confidence=EdgeConfidence(overall=confidence, evidence_quality=confidence),
        created_by="test-agent",
        hypothesis_branch="h-test",
        is_contradiction=is_contradiction,
        evidence=[
            EvidenceSource(
                source_type=EvidenceSourceType.PUBMED,
                source_id="PMID:99999",
                quality_score=0.85,
            )
        ]
        if with_evidence
        else [],
    )


def _make_falsification(
    counter_evidence_found: bool = True,
    confidence_delta: float = -0.1,
) -> FalsificationResult:
    return FalsificationResult(
        edge_id="e1",
        agent_id="test-agent",
        original_confidence=0.8,
        revised_confidence=0.8 + confidence_delta,
        falsified=counter_evidence_found,
        counter_evidence_found=counter_evidence_found,
        counter_evidence=[
            EvidenceSource(
                source_type=EvidenceSourceType.PUBMED,
                source_id="PMID:55555",
                quality_score=0.7,
            )
        ]
        if counter_evidence_found
        else [],
        confidence_delta=confidence_delta,
        reasoning="Test falsification reasoning",
    )


# ---------------------------------------------------------------------------
# Tests: R1 — Answer Correctness
# ---------------------------------------------------------------------------

class TestR1AnswerCorrectness:
    def test_perfect_score(self):
        result = _make_result()
        ctx = TrajectoryContext(evaluator_score=1.0)
        reward = compute_reward(result, ctx)
        assert reward.r1_answer_correctness == 1.0

    def test_zero_score(self):
        result = _make_result()
        ctx = TrajectoryContext(evaluator_score=0.0)
        reward = compute_reward(result, ctx)
        assert reward.r1_answer_correctness == 0.0

    def test_clamped_above_one(self):
        result = _make_result()
        ctx = TrajectoryContext(evaluator_score=1.5)
        reward = compute_reward(result, ctx)
        assert reward.r1_answer_correctness == 1.0

    def test_clamped_below_zero(self):
        result = _make_result()
        ctx = TrajectoryContext(evaluator_score=-0.5)
        reward = compute_reward(result, ctx)
        assert reward.r1_answer_correctness == 0.0


# ---------------------------------------------------------------------------
# Tests: R2 — KG Quality
# ---------------------------------------------------------------------------

class TestR2KGQuality:
    def test_high_quality_kg(self):
        nodes = [_make_node(0.9), _make_node(0.85)]
        edges = [_make_edge(0.9, with_evidence=True)]
        result = _make_result(nodes=nodes, edges=edges)
        ctx = TrajectoryContext()
        reward = compute_reward(result, ctx)
        assert reward.r2_kg_quality > 0.8

    def test_no_kg_mutations(self):
        result = _make_result()
        ctx = TrajectoryContext()
        reward = compute_reward(result, ctx)
        # With no nodes/edges, avg_confidence=0, provenance=0, contradiction=1.0 (vacuous)
        # Score = (0 + 0 + 1) / 3 ≈ 0.333
        assert abs(reward.r2_kg_quality - 1.0 / 3.0) < 1e-9

    def test_edges_without_evidence_penalized(self):
        edges = [_make_edge(0.9, with_evidence=False)]
        result = _make_result(edges=edges)
        ctx = TrajectoryContext()
        reward = compute_reward(result, ctx)
        # provenance_ratio = 0, so r2 < perfect
        assert reward.r2_kg_quality < 0.7


# ---------------------------------------------------------------------------
# Tests: R3 — Hypothesis Efficiency
# ---------------------------------------------------------------------------

class TestR3HypothesisEfficiency:
    def test_high_info_gain(self):
        result = _make_result()
        ctx = TrajectoryContext(
            hypothesis_info_gain=5.0,
            exploration_breadth=5,
            total_iterations=3,
        )
        reward = compute_reward(result, ctx)
        assert reward.r3_hypothesis_efficiency > 0.5

    def test_zero_gain(self):
        result = _make_result()
        ctx = TrajectoryContext(
            hypothesis_info_gain=0.0,
            exploration_breadth=0,
            total_iterations=1,
        )
        reward = compute_reward(result, ctx)
        assert reward.r3_hypothesis_efficiency < 0.3


# ---------------------------------------------------------------------------
# Tests: R4 — Falsification Quality
# ---------------------------------------------------------------------------

class TestR4FalsificationQuality:
    def test_good_falsification(self):
        fals = [_make_falsification(counter_evidence_found=True, confidence_delta=-0.15)]
        result = _make_result(falsifications=fals)
        ctx = TrajectoryContext()
        reward = compute_reward(result, ctx)
        assert reward.r4_falsification_quality > 0.5

    def test_no_falsification(self):
        result = _make_result()
        ctx = TrajectoryContext()
        reward = compute_reward(result, ctx)
        assert reward.r4_falsification_quality == 0.0

    def test_falsification_without_evidence(self):
        fals = [_make_falsification(counter_evidence_found=False, confidence_delta=0.0)]
        result = _make_result(falsifications=fals)
        ctx = TrajectoryContext()
        reward = compute_reward(result, ctx)
        # attempt_ratio=1.0, evidence_quality=0.0, confidence_update=0.0
        assert 0.0 < reward.r4_falsification_quality < 0.5


# ---------------------------------------------------------------------------
# Tests: R5 — Format Compliance
# ---------------------------------------------------------------------------

class TestR5FormatCompliance:
    def test_valid_json_with_structure(self):
        import json

        response = json.dumps({"summary": "BRCA1 finding", "evidence": ["PMID:123"]})
        result = _make_result(reasoning_trace=response)
        ctx = TrajectoryContext(expected_format="json")
        reward = compute_reward(result, ctx)
        assert reward.r5_format_compliance == 1.0

    def test_empty_response(self):
        result = _make_result(reasoning_trace="")
        ctx = TrajectoryContext()
        reward = compute_reward(result, ctx)
        assert reward.r5_format_compliance < 0.5

    def test_text_response_with_structure(self):
        result = _make_result(
            reasoning_trace="The summary of findings shows evidence for the hypothesis."
        )
        ctx = TrajectoryContext(expected_format="text")
        reward = compute_reward(result, ctx)
        assert reward.r5_format_compliance > 0.5


# ---------------------------------------------------------------------------
# Tests: Total Reward
# ---------------------------------------------------------------------------

class TestTotalReward:
    def test_total_is_weighted_sum(self):
        nodes = [_make_node(0.9)]
        edges = [_make_edge(0.9)]
        fals = [_make_falsification()]
        result = _make_result(
            nodes=nodes,
            edges=edges,
            falsifications=fals,
            reasoning_trace='{"summary": "test finding"}',
        )
        ctx = TrajectoryContext(
            evaluator_score=0.8,
            hypothesis_info_gain=2.0,
            exploration_breadth=3,
        )
        weights = RewardWeights()
        reward = compute_reward(result, ctx, weights)

        expected = (
            weights.answer_correctness * reward.r1_answer_correctness
            + weights.kg_quality * reward.r2_kg_quality
            + weights.hypothesis_efficiency * reward.r3_hypothesis_efficiency
            + weights.falsification_quality * reward.r4_falsification_quality
            + weights.format_compliance * reward.r5_format_compliance
        )
        assert abs(reward.total - expected) < 1e-9

    def test_total_in_zero_one_range(self):
        result = _make_result()
        ctx = TrajectoryContext(evaluator_score=0.5)
        reward = compute_reward(result, ctx)
        assert 0.0 <= reward.total <= 1.0

    def test_custom_weights(self):
        result = _make_result()
        ctx = TrajectoryContext(evaluator_score=1.0)
        # All weight on R1
        weights = RewardWeights(
            answer_correctness=1.0,
            kg_quality=0.0,
            hypothesis_efficiency=0.0,
            falsification_quality=0.0,
            format_compliance=0.0,
        )
        reward = compute_reward(result, ctx, weights)
        assert abs(reward.total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Tests: Batch
# ---------------------------------------------------------------------------

class TestBatchRewards:
    def test_batch_matches_individual(self):
        results = [_make_result() for _ in range(3)]
        contexts = [TrajectoryContext(evaluator_score=0.5 + i * 0.1) for i in range(3)]
        batch = compute_batch_rewards(results, contexts)
        individual = [compute_reward(r, c) for r, c in zip(results, contexts)]
        for b, ind in zip(batch, individual):
            assert abs(b.total - ind.total) < 1e-9

    def test_batch_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_batch_rewards([_make_result()], [])


# ---------------------------------------------------------------------------
# Tests: RewardBreakdown
# ---------------------------------------------------------------------------

class TestRewardBreakdown:
    def test_as_dict(self):
        b = RewardBreakdown(
            r1_answer_correctness=0.8,
            r2_kg_quality=0.7,
            total=0.75,
            details={"kg_avg_confidence": 0.9},
        )
        d = b.as_dict()
        assert d["r1_answer_correctness"] == 0.8
        assert d["total"] == 0.75
        assert d["kg_avg_confidence"] == 0.9
