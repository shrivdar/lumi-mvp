"""Tests for the benchmark report generator."""

from __future__ import annotations

import pytest

from benchmarks.models import (
    BenchmarkSuite,
    InstanceResult,
    InstanceStatus,
    RunMode,
)
from benchmarks.report import (
    PUBLISHED_BASELINES,
    aggregate_results,
    generate_report,
)


def _make_results(
    suite: BenchmarkSuite,
    mode: RunMode,
    count: int = 20,
    accuracy: float = 0.7,
) -> list[InstanceResult]:
    """Generate mock results with a target accuracy."""
    results = []
    n_correct = int(count * accuracy)
    for i in range(count):
        correct = i < n_correct
        results.append(
            InstanceResult(
                instance_id=f"inst_{i:04d}",
                suite=suite,
                mode=mode,
                predicted="A" if correct else "B",
                ground_truth="A",
                correct=correct,
                score=1.0 if correct else 0.0,
                tokens_used=500 + i * 10,
                latency_ms=100 + i * 5,
                turns=1 if mode == RunMode.ZERO_SHOT else 3,
                tools_used=["pubmed"] if mode == RunMode.YOHAS_FULL else [],
                status=InstanceStatus.COMPLETED,
            )
        )
    return results


class TestAggregateResults:
    def test_basic_aggregation(self):
        results = _make_results(BenchmarkSuite.BIOMNI_EVAL1, RunMode.ZERO_SHOT, count=10, accuracy=0.6)
        sr = aggregate_results(results, BenchmarkSuite.BIOMNI_EVAL1, RunMode.ZERO_SHOT)
        assert sr.total == 10
        assert sr.correct == 6
        assert sr.accuracy == pytest.approx(0.6)
        assert sr.avg_tokens > 0
        assert sr.avg_latency_ms > 0

    def test_all_correct(self):
        results = _make_results(BenchmarkSuite.LAB_BENCH_DBQA, RunMode.YOHAS_FULL, count=5, accuracy=1.0)
        sr = aggregate_results(results, BenchmarkSuite.LAB_BENCH_DBQA, RunMode.YOHAS_FULL)
        assert sr.accuracy == 1.0
        assert sr.correct == 5

    def test_none_correct(self):
        results = _make_results(BenchmarkSuite.BIXBENCH, RunMode.ZERO_SHOT, count=5, accuracy=0.0)
        sr = aggregate_results(results, BenchmarkSuite.BIXBENCH, RunMode.ZERO_SHOT)
        assert sr.accuracy == 0.0
        assert sr.correct == 0

    def test_empty_results(self):
        sr = aggregate_results([], BenchmarkSuite.BIOMNI_EVAL1, RunMode.ZERO_SHOT)
        assert sr.total == 0
        assert sr.accuracy == 0.0


class TestGenerateReport:
    def test_report_has_markdown(self):
        results_zs = _make_results(BenchmarkSuite.BIOMNI_EVAL1, RunMode.ZERO_SHOT, accuracy=0.5)
        results_yf = _make_results(BenchmarkSuite.BIOMNI_EVAL1, RunMode.YOHAS_FULL, accuracy=0.75)
        sr_zs = aggregate_results(results_zs, BenchmarkSuite.BIOMNI_EVAL1, RunMode.ZERO_SHOT)
        sr_yf = aggregate_results(results_yf, BenchmarkSuite.BIOMNI_EVAL1, RunMode.YOHAS_FULL)

        report = generate_report([sr_zs, sr_yf])
        assert report.markdown
        assert "YOHAS 3.0 Benchmark Report" in report.markdown
        assert "biomni_eval1" in report.markdown
        assert "50.0%" in report.markdown
        assert "75.0%" in report.markdown

    def test_report_includes_baselines(self):
        results = _make_results(BenchmarkSuite.LAB_BENCH_DBQA, RunMode.ZERO_SHOT)
        sr = aggregate_results(results, BenchmarkSuite.LAB_BENCH_DBQA, RunMode.ZERO_SHOT)
        report = generate_report([sr])
        assert "Published Baselines" in report.markdown

    def test_report_includes_failure_analysis(self):
        results = _make_results(BenchmarkSuite.BIOMNI_EVAL1, RunMode.ZERO_SHOT, count=5)
        # Add a failed instance
        results.append(
            InstanceResult(
                instance_id="failed_001",
                suite=BenchmarkSuite.BIOMNI_EVAL1,
                mode=RunMode.ZERO_SHOT,
                status=InstanceStatus.FAILED,
                error="API timeout",
                ground_truth="A",
            )
        )
        sr = aggregate_results(results, BenchmarkSuite.BIOMNI_EVAL1, RunMode.ZERO_SHOT)
        report = generate_report([sr])
        assert "Failure Analysis" in report.markdown
        assert "failed_001" in report.markdown

    def test_multi_suite_report(self):
        suites_data = [
            (BenchmarkSuite.BIOMNI_EVAL1, RunMode.ZERO_SHOT, 0.5),
            (BenchmarkSuite.BIOMNI_EVAL1, RunMode.YOHAS_FULL, 0.8),
            (BenchmarkSuite.LAB_BENCH_DBQA, RunMode.ZERO_SHOT, 0.45),
            (BenchmarkSuite.LAB_BENCH_DBQA, RunMode.YOHAS_FULL, 0.7),
            (BenchmarkSuite.LAB_BENCH_SEQQA, RunMode.ZERO_SHOT, 0.4),
            (BenchmarkSuite.LAB_BENCH_SEQQA, RunMode.YOHAS_FULL, 0.6),
        ]
        suite_results = []
        for suite, mode, acc in suites_data:
            results = _make_results(suite, mode, accuracy=acc)
            suite_results.append(aggregate_results(results, suite, mode))

        report = generate_report(suite_results)
        assert "biomni_eval1" in report.markdown
        assert "lab_bench_dbqa" in report.markdown
        assert "lab_bench_seqqa" in report.markdown

    def test_trajectory_stats_section(self):
        results = _make_results(BenchmarkSuite.BIOMNI_EVAL1, RunMode.YOHAS_FULL)
        sr = aggregate_results(results, BenchmarkSuite.BIOMNI_EVAL1, RunMode.YOHAS_FULL)
        report = generate_report([sr])
        assert "Trajectory Statistics" in report.markdown


class TestPublishedBaselines:
    def test_baselines_have_required_suites(self):
        assert "biomni_eval1" in PUBLISHED_BASELINES
        assert "lab_bench_dbqa" in PUBLISHED_BASELINES
        assert "lab_bench_seqqa" in PUBLISHED_BASELINES

    def test_baseline_values_are_valid(self):
        for suite, baselines in PUBLISHED_BASELINES.items():
            for name, value in baselines.items():
                assert 0.0 <= value <= 1.0, f"Invalid baseline {name} for {suite}: {value}"
