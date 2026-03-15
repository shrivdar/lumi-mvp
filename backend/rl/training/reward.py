"""Multi-dimensional reward function for YOHAS RL training.

Five reward components:
  R1 (0.50) — Answer correctness from evaluator
  R2 (0.20) — KG quality (confidence calibration, provenance, contradictions)
  R3 (0.15) — Hypothesis efficiency (info-gain per iteration, exploration breadth)
  R4 (0.10) — Falsification quality (real counter-evidence found)
  R5 (0.05) — Format compliance (structured output adherence)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from core.models import AgentResult, FalsificationResult, KGEdge, KGNode
from rl.training.config import RewardWeights

# ---------------------------------------------------------------------------
# Reward breakdown container
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    """Detailed per-component reward scores before weighting."""

    r1_answer_correctness: float = 0.0
    r2_kg_quality: float = 0.0
    r3_hypothesis_efficiency: float = 0.0
    r4_falsification_quality: float = 0.0
    r5_format_compliance: float = 0.0
    total: float = 0.0
    details: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {
            "r1_answer_correctness": self.r1_answer_correctness,
            "r2_kg_quality": self.r2_kg_quality,
            "r3_hypothesis_efficiency": self.r3_hypothesis_efficiency,
            "r4_falsification_quality": self.r4_falsification_quality,
            "r5_format_compliance": self.r5_format_compliance,
            "total": self.total,
            **self.details,
        }


# ---------------------------------------------------------------------------
# Trajectory context passed alongside the AgentResult
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryContext:
    """Supplemental signals not contained in AgentResult itself."""

    evaluator_score: float = 0.0
    hypothesis_info_gain: float = 0.0
    exploration_breadth: int = 0
    total_iterations: int = 1
    expected_format: str = "json"


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def _score_answer_correctness(evaluator_score: float) -> float:
    """R1: Direct evaluator score, clamped to [0, 1]."""
    return max(0.0, min(1.0, evaluator_score))


def _score_kg_quality(
    nodes: list[KGNode],
    edges: list[KGEdge],
) -> tuple[float, dict[str, float]]:
    """R2: KG quality — confidence calibration, provenance depth, contradiction awareness.

    Sub-components (equally weighted within R2):
      - avg_confidence: mean confidence across nodes + edges
      - provenance_ratio: fraction of edges with ≥1 evidence source
      - contradiction_awareness: fraction of contradictions flagged vs total
    """
    details: dict[str, float] = {}

    # Confidence calibration
    confidences: list[float] = []
    for n in nodes:
        confidences.append(n.confidence)
    for e in edges:
        confidences.append(e.confidence.overall)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    details["kg_avg_confidence"] = avg_conf

    # Provenance depth — edges with at least one evidence source
    edges_with_evidence = sum(1 for e in edges if len(e.evidence) > 0)
    provenance_ratio = edges_with_evidence / len(edges) if edges else 0.0
    details["kg_provenance_ratio"] = provenance_ratio

    # Contradiction awareness — flagged contradictions vs. potential
    contradiction_flagged = sum(1 for e in edges if e.is_contradiction)
    contradiction_total = contradiction_flagged + sum(1 for e in edges if e.falsified)
    contradiction_score = (
        contradiction_flagged / contradiction_total if contradiction_total > 0 else 1.0
    )
    details["kg_contradiction_score"] = contradiction_score

    score = (avg_conf + provenance_ratio + contradiction_score) / 3.0
    return score, details


def _score_hypothesis_efficiency(
    info_gain: float,
    exploration_breadth: int,
    total_iterations: int,
) -> tuple[float, dict[str, float]]:
    """R3: Hypothesis efficiency — info-gain per iteration + exploration breadth.

    Sub-components:
      - gain_per_iter: info_gain / iterations (log-scaled, capped at 1.0)
      - breadth_bonus: sigmoid of exploration_breadth centered at 3
    """
    details: dict[str, float] = {}

    gain_per_iter = info_gain / max(total_iterations, 1)
    # Log-scale so diminishing returns; cap at 1.0
    scaled_gain = min(1.0, math.log1p(gain_per_iter * 10) / math.log1p(10))
    details["hyp_gain_per_iter"] = scaled_gain

    # Sigmoid centered at 3 hypotheses explored — reward breadth but not spam
    breadth_bonus = 1.0 / (1.0 + math.exp(-(exploration_breadth - 3)))
    details["hyp_breadth_bonus"] = breadth_bonus

    score = (scaled_gain + breadth_bonus) / 2.0
    return score, details


def _score_falsification_quality(
    falsification_results: list[FalsificationResult],
) -> tuple[float, dict[str, float]]:
    """R4: Falsification quality — real counter-evidence found.

    Sub-components:
      - attempt_ratio: did the agent try to falsify its claims?
      - evidence_quality: when counter-evidence was found, was it real (sourced)?
      - confidence_update: did the agent update confidence based on evidence?
    """
    details: dict[str, float] = {}

    if not falsification_results:
        details["fals_attempt_ratio"] = 0.0
        details["fals_evidence_quality"] = 0.0
        details["fals_confidence_update"] = 0.0
        return 0.0, details

    n = len(falsification_results)

    # Attempt ratio — did the agent actually try?  (always 1.0 if results exist)
    attempt_ratio = 1.0
    details["fals_attempt_ratio"] = attempt_ratio

    # Evidence quality — fraction of attempts that found real counter-evidence
    real_evidence = sum(
        1 for f in falsification_results if f.counter_evidence_found and len(f.counter_evidence) > 0
    )
    evidence_quality = real_evidence / n
    details["fals_evidence_quality"] = evidence_quality

    # Confidence update — did the agent adjust confidence? (abs delta > 0)
    updates = sum(1 for f in falsification_results if abs(f.confidence_delta) > 0.01)
    confidence_update = updates / n
    details["fals_confidence_update"] = confidence_update

    score = (attempt_ratio + evidence_quality + confidence_update) / 3.0
    return score, details


def _score_format_compliance(
    raw_response: str,
    expected_format: str,
) -> tuple[float, dict[str, float]]:
    """R5: Format compliance — does the output conform to the expected structure?

    Checks:
      - valid JSON (if expected)
      - has required keys (summary, reasoning_trace)
      - reasonable length (not empty, not excessively long)
    """
    details: dict[str, float] = {}
    checks_passed = 0
    total_checks = 3

    # Check 1: parseable format
    is_valid_format = False
    if expected_format == "json":
        try:
            parsed = json.loads(raw_response)
            is_valid_format = isinstance(parsed, dict)
        except (json.JSONDecodeError, TypeError):
            is_valid_format = False
    else:
        is_valid_format = len(raw_response.strip()) > 0
    if is_valid_format:
        checks_passed += 1
    details["fmt_valid_format"] = 1.0 if is_valid_format else 0.0

    # Check 2: non-empty, not excessively long
    length = len(raw_response)
    reasonable_length = 10 < length < 100_000
    if reasonable_length:
        checks_passed += 1
    details["fmt_reasonable_length"] = 1.0 if reasonable_length else 0.0

    # Check 3: contains structured markers (summary-like content)
    has_structure = any(
        marker in raw_response.lower()
        for marker in ["summary", "finding", "evidence", "hypothesis", "conclusion"]
    )
    if has_structure:
        checks_passed += 1
    details["fmt_has_structure"] = 1.0 if has_structure else 0.0

    score = checks_passed / total_checks
    return score, details


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def compute_reward(
    result: AgentResult,
    context: TrajectoryContext,
    weights: RewardWeights | None = None,
) -> RewardBreakdown:
    """Compute multi-dimensional reward for a single agent trajectory.

    Args:
        result: The completed agent result containing KG mutations,
                falsification results, turns, and summary.
        context: Supplemental trajectory context (evaluator score,
                 hypothesis info-gain, etc.) not stored in AgentResult.
        weights: Reward component weights. Defaults to task-spec weights.

    Returns:
        RewardBreakdown with per-component scores and weighted total.
    """
    if weights is None:
        weights = RewardWeights()

    breakdown = RewardBreakdown()
    all_details: dict[str, float] = {}

    # R1: Answer correctness
    breakdown.r1_answer_correctness = _score_answer_correctness(context.evaluator_score)

    # R2: KG quality
    r2, r2_details = _score_kg_quality(result.nodes_added, result.edges_added)
    breakdown.r2_kg_quality = r2
    all_details.update(r2_details)

    # R3: Hypothesis efficiency
    r3, r3_details = _score_hypothesis_efficiency(
        context.hypothesis_info_gain,
        context.exploration_breadth,
        context.total_iterations,
    )
    breakdown.r3_hypothesis_efficiency = r3
    all_details.update(r3_details)

    # R4: Falsification quality
    r4, r4_details = _score_falsification_quality(result.falsification_results)
    breakdown.r4_falsification_quality = r4
    all_details.update(r4_details)

    # R5: Format compliance
    raw_response = result.reasoning_trace or result.summary or ""
    r5, r5_details = _score_format_compliance(raw_response, context.expected_format)
    breakdown.r5_format_compliance = r5
    all_details.update(r5_details)

    # Weighted total
    breakdown.total = (
        weights.answer_correctness * breakdown.r1_answer_correctness
        + weights.kg_quality * breakdown.r2_kg_quality
        + weights.hypothesis_efficiency * breakdown.r3_hypothesis_efficiency
        + weights.falsification_quality * breakdown.r4_falsification_quality
        + weights.format_compliance * breakdown.r5_format_compliance
    )

    breakdown.details = all_details
    return breakdown


def compute_batch_rewards(
    results: list[AgentResult],
    contexts: list[TrajectoryContext],
    weights: RewardWeights | None = None,
) -> list[RewardBreakdown]:
    """Compute rewards for a batch of trajectories."""
    if len(results) != len(contexts):
        msg = f"results ({len(results)}) and contexts ({len(contexts)}) must have same length"
        raise ValueError(msg)
    return [compute_reward(r, c, weights) for r, c in zip(results, contexts)]
