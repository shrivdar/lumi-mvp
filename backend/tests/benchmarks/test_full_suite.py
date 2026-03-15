"""Full benchmark suite integration test.

Runs all benchmark suites in simulated (dry-run) mode and validates
the end-to-end pipeline: load → evaluate → aggregate → report → trajectories.
"""

from __future__ import annotations

import pytest

from benchmarks.adapters import get_adapter
from benchmarks.evaluator import BenchmarkEvaluator
from benchmarks.models import BenchmarkSuite, InstanceStatus, RunMode, SuiteResults
from benchmarks.report import aggregate_results, generate_report
from benchmarks.trajectory_store import TrajectoryStore


class TestFullBenchmarkPipeline:
    """End-to-end test: adapters → evaluator → report → trajectories."""

    @pytest.mark.asyncio
    async def test_biomni_eval1_full_pipeline(self, tmp_path):
        """Run Biomni-Eval1 through the full pipeline (limited to 20 instances)."""
        adapter = get_adapter(BenchmarkSuite.BIOMNI_EVAL1)
        instances = adapter.load_instances(limit=20)
        assert len(instances) == 20

        all_suite_results: list[SuiteResults] = []
        trajectory_store = TrajectoryStore(output_dir=tmp_path / "trajectories")

        for mode in RunMode:
            evaluator = BenchmarkEvaluator(mode=mode, collect_trajectories=True)
            results = await evaluator.evaluate_batch(instances)
            assert len(results) == 20
            assert all(r.status == InstanceStatus.COMPLETED for r in results)

            sr = aggregate_results(results, BenchmarkSuite.BIOMNI_EVAL1, mode)
            all_suite_results.append(sr)
            assert sr.total == 20
            assert 0.0 <= sr.accuracy <= 1.0

            if evaluator.trajectories:
                trajectory_store.save(evaluator.trajectories, tag=f"biomni_{mode.value}")

        report = generate_report(all_suite_results)
        assert report.markdown
        assert "YOHAS 3.0 Benchmark Report" in report.markdown

        # Write report to disk
        report_path = tmp_path / "report.md"
        report_path.write_text(report.markdown)
        assert report_path.exists()

    @pytest.mark.asyncio
    async def test_bixbench_pipeline(self, tmp_path):
        """Run BixBench through pipeline (limited)."""
        adapter = get_adapter(BenchmarkSuite.BIXBENCH)
        instances = adapter.load_instances(limit=10)
        assert len(instances) == 10

        evaluator = BenchmarkEvaluator(mode=RunMode.YOHAS_FULL, collect_trajectories=True)
        results = await evaluator.evaluate_batch(instances)
        sr = aggregate_results(results, BenchmarkSuite.BIXBENCH, RunMode.YOHAS_FULL)
        assert sr.total == 10

        # Trajectories should be collected for BixBench
        assert len(evaluator.trajectories) > 0
        store = TrajectoryStore(output_dir=tmp_path)
        store.save(evaluator.trajectories, tag="bixbench")

    @pytest.mark.asyncio
    async def test_lab_bench_all_categories(self):
        """Run all LAB-Bench categories (DbQA, SeqQA, LitQA2)."""
        lab_bench_suites = [
            BenchmarkSuite.LAB_BENCH_DBQA,
            BenchmarkSuite.LAB_BENCH_SEQQA,
            BenchmarkSuite.LAB_BENCH_LITQA2,
        ]
        suite_results: list[SuiteResults] = []

        for suite in lab_bench_suites:
            adapter = get_adapter(suite)
            instances = adapter.load_instances(limit=10)
            evaluator = BenchmarkEvaluator(mode=RunMode.YOHAS_FULL)
            results = await evaluator.evaluate_batch(instances)
            sr = aggregate_results(results, suite, RunMode.YOHAS_FULL)
            suite_results.append(sr)
            assert sr.total == 10

        report = generate_report(suite_results)
        assert "lab_bench_dbqa" in report.markdown
        assert "lab_bench_seqqa" in report.markdown
        assert "lab_bench_litqa2" in report.markdown

    @pytest.mark.asyncio
    async def test_all_suites_all_modes(self, tmp_path):
        """Run every suite in every mode with small limit — validates the full matrix."""
        limit = 5
        all_results: list[SuiteResults] = []

        for suite in BenchmarkSuite:
            adapter = get_adapter(suite)
            instances = adapter.load_instances(limit=limit)

            for mode in RunMode:
                evaluator = BenchmarkEvaluator(mode=mode)
                results = await evaluator.evaluate_batch(instances)
                sr = aggregate_results(results, suite, mode)
                all_results.append(sr)

        # 5 suites × 2 modes = 10 suite results
        assert len(all_results) == 10

        report = generate_report(all_results)
        assert report.markdown
        # Report should mention all suites
        for suite in BenchmarkSuite:
            assert suite.value in report.markdown

    @pytest.mark.asyncio
    async def test_per_instance_metrics_recorded(self):
        """Verify each instance has complete metrics."""
        adapter = get_adapter(BenchmarkSuite.BIOMNI_EVAL1)
        instances = adapter.load_instances(limit=5)
        evaluator = BenchmarkEvaluator(mode=RunMode.YOHAS_FULL)
        results = await evaluator.evaluate_batch(instances)

        for r in results:
            assert r.instance_id
            assert r.suite == BenchmarkSuite.BIOMNI_EVAL1
            assert r.mode == RunMode.YOHAS_FULL
            assert r.status == InstanceStatus.COMPLETED
            assert r.latency_ms >= 0
            assert r.turns >= 1
            assert isinstance(r.tokens_used, int)
            assert isinstance(r.tools_used, list)
            assert r.ground_truth  # should have ground truth

    @pytest.mark.asyncio
    async def test_comparison_report_structure(self, tmp_path):
        """Validate the comparison report has the expected sections."""
        suites_data = [
            (BenchmarkSuite.BIOMNI_EVAL1, RunMode.ZERO_SHOT, 10),
            (BenchmarkSuite.BIOMNI_EVAL1, RunMode.YOHAS_FULL, 10),
            (BenchmarkSuite.LAB_BENCH_DBQA, RunMode.ZERO_SHOT, 10),
            (BenchmarkSuite.LAB_BENCH_DBQA, RunMode.YOHAS_FULL, 10),
        ]
        all_sr = []
        for suite, mode, limit in suites_data:
            adapter = get_adapter(suite)
            instances = adapter.load_instances(limit=limit)
            evaluator = BenchmarkEvaluator(mode=mode)
            results = await evaluator.evaluate_batch(instances)
            all_sr.append(aggregate_results(results, suite, mode))

        report = generate_report(all_sr)
        md = report.markdown

        # Required sections
        assert "# YOHAS 3.0 Benchmark Report" in md
        assert "## Summary" in md
        assert "## Comparison: Zero-shot vs YOHAS Full vs Published Baselines" in md
        assert "## Per-Category Breakdown" in md
        assert "## Trajectory Statistics" in md

        # Should have the comparison table
        assert "Zero-shot (Opus)" in md
        assert "YOHAS Full" in md

        # Write to file for inspection
        (tmp_path / "comparison_report.md").write_text(md)
