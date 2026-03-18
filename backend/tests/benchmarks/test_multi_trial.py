"""Tests for the multi-trial benchmark protocol and strategy memory."""

from __future__ import annotations

import pytest

from benchmarks.evaluator import BenchmarkEvaluator
from benchmarks.models import (
    BenchmarkInstance,
    BenchmarkSuite,
    InstanceStatus,
    RunMode,
    TrialResult,
)
from benchmarks.strategy_memory import StrategyMemory, TrialSummary


def _make_instance(
    instance_id: str = "test_001",
    question: str = "What is BRCA1?",
    ground_truth: str = "A",
    choices: list[str] | None = None,
) -> BenchmarkInstance:
    return BenchmarkInstance(
        suite=BenchmarkSuite.BIOMNI_EVAL1,
        instance_id=instance_id,
        question=question,
        ground_truth=ground_truth,
        choices=choices or ["A", "B", "C", "D"],
        category="test",
    )


# ------------------------------------------------------------------
# Multi-trial evaluator tests
# ------------------------------------------------------------------


class TestMultiTrialEvaluator:
    @pytest.mark.asyncio
    async def test_single_trial_default(self):
        """Default max_trials=1 behaves like original single-trial."""
        evaluator = BenchmarkEvaluator(mode=RunMode.ZERO_SHOT, max_trials=1)
        inst = _make_instance()
        result = await evaluator.evaluate_instance(inst)
        assert result.status == InstanceStatus.COMPLETED
        assert result.best_trial == 0  # 0 = single trial mode
        assert result.trial_results == []

    @pytest.mark.asyncio
    async def test_multi_trial_creates_trial_results(self):
        """Multi-trial mode produces TrialResult entries."""
        evaluator = BenchmarkEvaluator(mode=RunMode.ZERO_SHOT, max_trials=3)
        inst = _make_instance()
        result = await evaluator.evaluate_instance(inst)
        assert result.status == InstanceStatus.COMPLETED
        assert len(result.trial_results) >= 1  # at least 1, up to 3
        assert result.best_trial >= 1  # 1-indexed for multi-trial

        for tr in result.trial_results:
            assert isinstance(tr, TrialResult)
            assert tr.trial_number >= 1
            assert tr.tokens_used >= 0

    @pytest.mark.asyncio
    async def test_multi_trial_early_exit_on_correct(self):
        """Multi-trial stops early when correct answer is found."""
        import random
        random.seed(0)  # seed so first trial is correct

        evaluator = BenchmarkEvaluator(mode=RunMode.ZERO_SHOT, max_trials=5)

        # Run many times; at least some should early-exit
        results = []
        for i in range(20):
            random.seed(i)
            inst = _make_instance(instance_id=f"test_{i:03d}")
            result = await evaluator.evaluate_instance(inst)
            results.append(result)

        # At least one should have < 5 trials if any got correct early
        correct_results = [r for r in results if r.correct]
        if correct_results:
            some_early = any(len(r.trial_results) < 5 for r in correct_results)
            assert some_early, "Expected early exit on correct answer"

    @pytest.mark.asyncio
    async def test_multi_trial_yohas_full_requires_factory(self):
        """YOHAS_FULL mode without orchestrator_factory should fail."""
        evaluator = BenchmarkEvaluator(mode=RunMode.YOHAS_FULL, max_trials=2)
        inst = _make_instance()
        result = await evaluator.evaluate_instance(inst)
        # Without orchestrator_factory, YOHAS_FULL raises RuntimeError → status=failed
        assert result.status == InstanceStatus.FAILED

    @pytest.mark.asyncio
    async def test_multi_trial_tokens_accumulated(self):
        """Total tokens should be sum across all trials."""
        evaluator = BenchmarkEvaluator(mode=RunMode.ZERO_SHOT, max_trials=3)
        inst = _make_instance()
        result = await evaluator.evaluate_instance(inst)

        if len(result.trial_results) > 1:
            trial_token_sum = sum(t.tokens_used for t in result.trial_results)
            assert result.tokens_used == trial_token_sum

    @pytest.mark.asyncio
    async def test_multi_trial_batch(self):
        """Multi-trial works with batch evaluation."""
        evaluator = BenchmarkEvaluator(
            mode=RunMode.ZERO_SHOT,
            max_trials=2,
            max_concurrency=3,
        )
        instances = [_make_instance(instance_id=f"test_{i:03d}") for i in range(5)]
        results = await evaluator.evaluate_batch(instances)
        assert len(results) == 5
        assert all(r.status == InstanceStatus.COMPLETED for r in results)

    @pytest.mark.asyncio
    async def test_code_first_single_trial(self):
        """CODE_FIRST mode routes through _run_single_trial without crashing."""
        evaluator = BenchmarkEvaluator(mode=RunMode.CODE_FIRST, max_trials=1)
        inst = _make_instance()
        result = await evaluator.evaluate_instance(inst)
        assert result.status == InstanceStatus.COMPLETED
        assert result.mode == RunMode.CODE_FIRST

    @pytest.mark.asyncio
    async def test_code_first_multi_trial(self):
        """CODE_FIRST mode works with multi-trial protocol."""
        evaluator = BenchmarkEvaluator(mode=RunMode.CODE_FIRST, max_trials=3)
        inst = _make_instance()
        result = await evaluator.evaluate_instance(inst)
        assert result.status == InstanceStatus.COMPLETED
        assert result.mode == RunMode.CODE_FIRST
        assert len(result.trial_results) >= 1

    @pytest.mark.asyncio
    async def test_multi_trial_trajectories_collected(self):
        """Trajectories are collected across all trials."""
        evaluator = BenchmarkEvaluator(
            mode=RunMode.ZERO_SHOT,
            max_trials=3,
            collect_trajectories=True,
        )
        inst = _make_instance()
        await evaluator.evaluate_instance(inst)
        # Should have at least 1 trajectory (one per trial)
        assert len(evaluator.trajectories) >= 1


# ------------------------------------------------------------------
# Strategy memory tests
# ------------------------------------------------------------------


class TestStrategyMemory:
    def test_extract_template(self):
        mem = StrategyMemory()

        class FakeResult:
            reasoning_trace = "Found that BRCA1 is a tumor suppressor gene involved in DNA repair."
            report_markdown = ""
            key_findings = []
            total_iterations = 3

        template = mem.extract_template(
            name="brca1_test",
            query="What is the role of BRCA1 in cancer?",
            result=FakeResult(),
            score=0.85,
            tools_used=["pubmed", "uniprot"],
            agent_types=["literature_analyst", "protein_analyst"],
        )

        assert template.name == "brca1_test"
        assert template.score == 0.85
        assert "pubmed" in template.tools_used
        assert template.query_archetype != ""
        assert len(mem.templates) == 1

    def test_get_hint_empty(self):
        mem = StrategyMemory()
        hint = mem.get_hint([])
        assert hint == ""

    def test_get_hint_single_trial(self):
        mem = StrategyMemory()
        summaries = [
            TrialSummary(
                trial_number=1,
                predicted="tumor suppressor",
                score=0.5,
                reasoning_trace="BRCA1 is involved in DNA repair pathways.",
                tools_used=["pubmed"],
                tokens_used=1000,
            ),
        ]
        hint = mem.get_hint(summaries)
        assert "Trial 1" in hint
        assert "tumor suppressor" in hint
        assert "0.50" in hint

    def test_get_hint_multiple_trials(self):
        mem = StrategyMemory()
        summaries = [
            TrialSummary(
                trial_number=1,
                predicted="answer1",
                score=0.3,
                reasoning_trace="First attempt reasoning.",
            ),
            TrialSummary(
                trial_number=2,
                predicted="answer2",
                score=0.6,
                reasoning_trace="Second attempt with more tools.",
                tools_used=["pubmed", "uniprot"],
            ),
        ]
        hint = mem.get_hint(summaries)
        assert "Trial 1" in hint
        assert "Trial 2" in hint
        assert "answer1" in hint
        assert "answer2" in hint

    def test_classify_archetype(self):
        mem = StrategyMemory()
        assert mem._classify_archetype("EGFR TKI resistance in NSCLC") == "drug_resistance"
        assert mem._classify_archetype("PROTAC degrader design for BRD4") == "drug_design"
        assert mem._classify_archetype("CRISPR base editing safety") == "gene_editing"
        assert mem._classify_archetype("CAR-T cell therapy outcomes") == "immunotherapy"
        assert mem._classify_archetype("random unrelated query") == "general"

    def test_get_best_template(self):
        mem = StrategyMemory()

        class FakeResult:
            reasoning_trace = "test"
            report_markdown = ""
            key_findings = []
            total_iterations = 1

        mem.extract_template(name="t1", query="EGFR resistance", result=FakeResult(), score=0.5)
        mem.extract_template(name="t2", query="KRAS targeting", result=FakeResult(), score=0.9)
        mem.extract_template(name="t3", query="PARP resistance", result=FakeResult(), score=0.7)

        best = mem.get_best_template()
        assert best is not None
        assert best.name == "t2"
        assert best.score == 0.9

    def test_storage_persistence(self, tmp_path):
        """Templates are saved and loaded from disk."""
        storage_dir = tmp_path / "strategies"

        class FakeResult:
            reasoning_trace = "test reasoning"
            report_markdown = ""
            key_findings = []
            total_iterations = 2

        # Save
        mem1 = StrategyMemory(storage_dir=storage_dir)
        mem1.extract_template(name="persist_test", query="test", result=FakeResult(), score=0.8)
        assert len(mem1.templates) == 1

        # Load in new instance
        mem2 = StrategyMemory(storage_dir=storage_dir)
        assert len(mem2.templates) == 1
        assert mem2.templates[0].name == "persist_test"
        assert mem2.templates[0].score == 0.8

    def test_hint_with_strategy_templates(self):
        """Hint includes relevant strategy templates when available."""
        mem = StrategyMemory()

        class FakeResult:
            reasoning_trace = "Used pubmed to find key papers on resistance mechanisms."
            report_markdown = ""
            key_findings = []
            total_iterations = 3

        mem.extract_template(
            name="proven_strategy",
            query="resistance mechanisms",
            result=FakeResult(),
            score=0.9,
            tools_used=["pubmed", "semantic_scholar"],
        )

        summaries = [
            TrialSummary(trial_number=1, predicted="test", score=0.3),
        ]
        hint = mem.get_hint(summaries)
        assert "Proven Strategies" in hint
        assert "proven_strategy" in hint


# ------------------------------------------------------------------
# Trial result model tests
# ------------------------------------------------------------------


class TestTrialResultModel:
    def test_trial_result_creation(self):
        tr = TrialResult(
            trial_number=1,
            predicted="A",
            correct=True,
            score=1.0,
            tokens_used=500,
            latency_ms=100,
            turns=1,
        )
        assert tr.trial_number == 1
        assert tr.correct
        assert tr.score == 1.0

    def test_trial_result_serialization(self):
        tr = TrialResult(
            trial_number=2,
            predicted="B",
            score=0.5,
            hint_injected="some hint",
        )
        data = tr.model_dump()
        assert data["trial_number"] == 2
        assert data["hint_injected"] == "some hint"


# ------------------------------------------------------------------
# Best trial selection tests
# ------------------------------------------------------------------


class TestBestTrialSelection:
    def test_select_correct_over_incorrect(self):
        trials = [
            TrialResult(trial_number=1, predicted="wrong", score=0.0, reasoning_trace="short"),
            TrialResult(trial_number=2, predicted="right", score=1.0, reasoning_trace="longer trace"),
        ]
        idx = BenchmarkEvaluator._select_best_trial(trials)
        assert idx == 1  # trial 2 (score=1.0)

    def test_select_longer_trace_on_tie(self):
        trials = [
            TrialResult(trial_number=1, predicted="a", score=0.0, reasoning_trace="short"),
            TrialResult(trial_number=2, predicted="b", score=0.0, reasoning_trace="much longer reasoning trace"),
        ]
        idx = BenchmarkEvaluator._select_best_trial(trials)
        assert idx == 1  # longer trace

    def test_select_empty(self):
        idx = BenchmarkEvaluator._select_best_trial([])
        assert idx == 0
