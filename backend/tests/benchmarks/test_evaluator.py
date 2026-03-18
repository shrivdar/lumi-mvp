"""Tests for the benchmark evaluator."""

from __future__ import annotations

import pytest

from benchmarks.evaluator import BenchmarkEvaluator
from benchmarks.models import (
    BenchmarkInstance,
    BenchmarkSuite,
    InstanceStatus,
    RunMode,
)


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


class TestZeroShotEvaluator:
    @pytest.mark.asyncio
    async def test_simulated_evaluation(self, zero_shot_evaluator):
        inst = _make_instance()
        result = await zero_shot_evaluator.evaluate_instance(inst)
        assert result.instance_id == "test_001"
        assert result.suite == BenchmarkSuite.BIOMNI_EVAL1
        assert result.mode == RunMode.ZERO_SHOT
        assert result.status == InstanceStatus.COMPLETED
        assert result.ground_truth == "A"
        assert result.latency_ms >= 0
        assert result.turns >= 1

    @pytest.mark.asyncio
    async def test_batch_evaluation(self, zero_shot_evaluator):
        instances = [_make_instance(instance_id=f"test_{i:03d}") for i in range(10)]
        results = await zero_shot_evaluator.evaluate_batch(instances)
        assert len(results) == 10
        assert all(r.status == InstanceStatus.COMPLETED for r in results)

    @pytest.mark.asyncio
    async def test_trajectories_collected(self):
        evaluator = BenchmarkEvaluator(mode=RunMode.ZERO_SHOT, collect_trajectories=True)
        inst = _make_instance()
        await evaluator.evaluate_instance(inst)
        assert len(evaluator.trajectories) == 1
        assert evaluator.trajectories[0].instance_id == "test_001"

    @pytest.mark.asyncio
    async def test_trajectories_not_collected_when_disabled(self):
        evaluator = BenchmarkEvaluator(mode=RunMode.ZERO_SHOT, collect_trajectories=False)
        inst = _make_instance()
        await evaluator.evaluate_instance(inst)
        assert len(evaluator.trajectories) == 0


class TestYohasFullEvaluator:
    @pytest.mark.asyncio
    async def test_fails_without_orchestrator_factory(self, yohas_full_evaluator):
        """YOHAS_FULL mode now requires an orchestrator_factory (no more simulation)."""
        inst = _make_instance()
        result = await yohas_full_evaluator.evaluate_instance(inst)
        assert result.instance_id == "test_001"
        assert result.mode == RunMode.YOHAS_FULL
        assert result.status == InstanceStatus.FAILED
        assert "orchestrator_factory" in result.error

    @pytest.mark.asyncio
    async def test_batch_with_concurrency(self):
        evaluator = BenchmarkEvaluator(
            mode=RunMode.YOHAS_FULL,
            max_concurrency=3,
        )
        instances = [_make_instance(instance_id=f"test_{i:03d}") for i in range(15)]
        results = await evaluator.evaluate_batch(instances)
        assert len(results) == 15
        # All should fail without orchestrator_factory
        assert all(r.status == InstanceStatus.FAILED for r in results)


class TestCodeFirstEvaluator:
    @pytest.mark.asyncio
    async def test_code_first_fallback_without_factory(self):
        """CODE_FIRST falls back to dry-run simulation when no orchestrator_factory."""
        evaluator = BenchmarkEvaluator(mode=RunMode.CODE_FIRST)
        inst = _make_instance()
        result = await evaluator.evaluate_instance(inst)
        assert result.instance_id == "test_001"
        assert result.mode == RunMode.CODE_FIRST
        assert result.status == InstanceStatus.COMPLETED


class TestAnswerExtraction:
    def test_extract_letter_answer(self, zero_shot_evaluator):
        answer = zero_shot_evaluator._extract_answer("The answer is B because...", ["X", "Y", "Z", "W"])
        assert answer == "Y"

    def test_extract_first_line_fallback(self, zero_shot_evaluator):
        answer = zero_shot_evaluator._extract_answer("tumor suppressor\nmore details", [])
        assert answer == "tumor suppressor"

    def test_check_answer_exact(self, zero_shot_evaluator):
        assert zero_shot_evaluator._check_answer("A", "A")
        assert zero_shot_evaluator._check_answer("a", "A")
        assert not zero_shot_evaluator._check_answer("B", "A")

    def test_check_answer_prefix(self, zero_shot_evaluator):
        assert zero_shot_evaluator._check_answer("tumor suppressor gene", "tumor suppressor")

    def test_check_answer_empty(self, zero_shot_evaluator):
        assert not zero_shot_evaluator._check_answer("", "A")
        assert not zero_shot_evaluator._check_answer("A", "")


class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_callback_called(self, zero_shot_evaluator):
        calls = []

        def callback(done, total, result):
            calls.append((done, total, result.instance_id))

        instances = [_make_instance(instance_id=f"test_{i:03d}") for i in range(5)]
        await zero_shot_evaluator.evaluate_batch(instances, progress_callback=callback)
        assert len(calls) == 5
        # All should have total=5
        assert all(t == 5 for _, t, _ in calls)
