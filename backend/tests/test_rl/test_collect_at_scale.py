"""Tests for batch trajectory collection pipeline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkSuite, RunMode
from rl.collect_at_scale import collect_trajectories, _benchmark_trajectory_to_rl
from benchmarks.models import BenchmarkInstance, InstanceResult, InstanceStatus


class TestBenchmarkTrajectoryConversion:
    def test_converts_correct_result(self):
        instance = BenchmarkInstance(
            suite=BenchmarkSuite.BIOMNI_EVAL1,
            instance_id="biomni_0001",
            question="What is the role of BRCA1?",
            ground_truth="A",
            choices=["A", "B", "C", "D"],
            category="gwas_causal_gene",
        )
        result = InstanceResult(
            instance_id="biomni_0001",
            suite=BenchmarkSuite.BIOMNI_EVAL1,
            mode=RunMode.YOHAS_FULL,
            predicted="A",
            ground_truth="A",
            correct=True,
            score=1.0,
            tokens_used=500,
            latency_ms=1000,
            tools_used=["pubmed", "semantic_scholar"],
            reasoning_trace="BRCA1 is a tumor suppressor gene...",
            status=InstanceStatus.COMPLETED,
        )

        traj = _benchmark_trajectory_to_rl(instance, result)

        assert traj.task_id == "biomni_0001"
        assert traj.reward == 1.0
        assert traj.success is True
        assert traj.instruction == "What is the role of BRCA1?"
        assert traj.context["suite"] == "biomni_eval1"
        assert traj.final_answer == "A"
        assert traj.total_tokens == 500
        assert len(traj.turns) >= 2  # at least instruction + answer

    def test_converts_failed_result(self):
        instance = BenchmarkInstance(
            suite=BenchmarkSuite.BIOMNI_EVAL1,
            instance_id="biomni_0002",
            question="What is TP53?",
            ground_truth="B",
        )
        result = InstanceResult(
            instance_id="biomni_0002",
            suite=BenchmarkSuite.BIOMNI_EVAL1,
            mode=RunMode.ZERO_SHOT,
            predicted="C",
            ground_truth="B",
            correct=False,
            score=0.0,
            status=InstanceStatus.COMPLETED,
        )

        traj = _benchmark_trajectory_to_rl(instance, result)

        assert traj.reward == 0.0
        assert traj.success is False

    def test_converts_timeout_result(self):
        instance = BenchmarkInstance(
            suite=BenchmarkSuite.BIOMNI_EVAL1,
            instance_id="biomni_0003",
            question="Test?",
            ground_truth="A",
        )
        result = InstanceResult(
            instance_id="biomni_0003",
            suite=BenchmarkSuite.BIOMNI_EVAL1,
            mode=RunMode.ZERO_SHOT,
            status=InstanceStatus.TIMEOUT,
            error="timeout",
        )

        traj = _benchmark_trajectory_to_rl(instance, result)
        assert traj.reward == 0.0


class TestCollectTrajectories:
    def test_collect_dry_run(self, tmp_path: Path):
        """Collect trajectories in dry-run mode (no LLM)."""
        out_path = asyncio.run(
            collect_trajectories(
                suite=BenchmarkSuite.BIOMNI_EVAL1,
                mode=RunMode.ZERO_SHOT,
                limit=5,
                live=False,
                output_dir=tmp_path,
                seed=42,
            )
        )

        assert out_path.exists()
        assert out_path.suffix == ".jsonl"

        # Check JSONL content
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) > 0

        for line in lines:
            data = json.loads(line)
            assert "task_id" in data
            assert "reward" in data
            assert "turns" in data
            assert data["research_id"] == "bench_biomni_eval1"

        # Check metadata file
        meta_path = out_path.with_suffix(".meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["suite"] == "biomni_eval1"
        assert meta["instances_loaded"] == 5
        assert meta["live"] is False

    def test_collect_with_different_suites(self, tmp_path: Path):
        """Can collect from different benchmark suites."""
        for suite in [BenchmarkSuite.BIOMNI_EVAL1, BenchmarkSuite.BIXBENCH]:
            out_path = asyncio.run(
                collect_trajectories(
                    suite=suite,
                    mode=RunMode.ZERO_SHOT,
                    limit=3,
                    live=False,
                    output_dir=tmp_path,
                    seed=42,
                )
            )
            assert out_path.exists()
            assert suite.value in out_path.name
