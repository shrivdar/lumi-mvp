"""Report generator — produces comparison markdown from benchmark results."""

from __future__ import annotations

from datetime import UTC, datetime

from benchmarks.models import (
    BenchmarkReport,
    BenchmarkSuite,
    InstanceResult,
    InstanceStatus,
    RunMode,
    SuiteResults,
)

# Published baselines for comparison
PUBLISHED_BASELINES: dict[str, dict[str, float]] = {
    "biomni_eval1": {
        "Biomni A1 (published)": 0.744,
        "GPT-4o (zero-shot)": 0.62,
        "Claude 3.5 Sonnet (zero-shot)": 0.65,
    },
    "lab_bench_dbqa": {
        "Biomni A1 (published)": 0.744,
        "GPT-4o (zero-shot)": 0.58,
        "PaperQA2": 0.61,
    },
    "lab_bench_seqqa": {
        "Biomni A1 (published)": 0.819,
        "GPT-4o (zero-shot)": 0.45,
        "PaperQA2": 0.52,
    },
    "lab_bench_litqa2": {
        "PaperQA2 (published)": 0.61,
        "GPT-4o (zero-shot)": 0.55,
    },
    "bixbench": {
        "GPT-4o (baseline)": 0.35,
        "Claude 3.5 Sonnet (baseline)": 0.38,
    },
}


def aggregate_results(
    results: list[InstanceResult],
    suite: BenchmarkSuite,
    mode: RunMode,
) -> SuiteResults:
    """Aggregate per-instance results into suite-level metrics."""
    completed = [r for r in results if r.status == InstanceStatus.COMPLETED]
    correct = [r for r in completed if r.correct]

    # Per-category breakdown
    by_category: dict[str, dict[str, float]] = {}
    cat_counts: dict[str, dict[str, int]] = {}

    for r in results:
        cat = r.instance_id.rsplit("_", 1)[0] if "_" in r.instance_id else "general"
        # Use the suite+category from the original instance metadata if possible
        # For now, derive from instance_id prefix
        for inst_r in results:
            if inst_r.instance_id == r.instance_id:
                break

    # Simpler per-category aggregation from instance IDs
    for r in completed:
        # Extract category from instance results
        cat = _infer_category(r.instance_id)
        if cat not in cat_counts:
            cat_counts[cat] = {"total": 0, "correct": 0, "tokens": 0, "latency_ms": 0}
        cat_counts[cat]["total"] += 1
        cat_counts[cat]["correct"] += 1 if r.correct else 0
        cat_counts[cat]["tokens"] += r.tokens_used
        cat_counts[cat]["latency_ms"] += r.latency_ms

    for cat, counts in cat_counts.items():
        total = counts["total"]
        by_category[cat] = {
            "accuracy": counts["correct"] / total if total else 0.0,
            "total": float(total),
            "correct": float(counts["correct"]),
            "avg_tokens": counts["tokens"] / total if total else 0.0,
            "avg_latency_ms": counts["latency_ms"] / total if total else 0.0,
        }

    total = len(results)
    n_correct = len(correct)
    n_completed = len(completed)

    return SuiteResults(
        suite=suite,
        mode=mode,
        total=total,
        correct=n_correct,
        accuracy=n_correct / n_completed if n_completed else 0.0,
        avg_tokens=sum(r.tokens_used for r in completed) / n_completed if n_completed else 0.0,
        avg_latency_ms=sum(r.latency_ms for r in completed) / n_completed if n_completed else 0.0,
        avg_turns=sum(r.turns for r in completed) / n_completed if n_completed else 0.0,
        by_category=by_category,
        results=results,
        completed_at=datetime.now(UTC),
    )


def generate_report(suite_results: list[SuiteResults]) -> BenchmarkReport:
    """Generate a full comparison report from suite results."""
    report = BenchmarkReport(
        suites=suite_results,
        baselines=PUBLISHED_BASELINES,
    )
    report.markdown = _render_markdown(report)
    return report


def _render_markdown(report: BenchmarkReport) -> str:
    """Render the benchmark report as markdown."""
    lines: list[str] = []
    lines.append("# YOHAS 3.0 Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    # ── Summary table ──
    lines.append("## Summary")
    lines.append("")
    lines.append("| Suite | Mode | Instances | Correct | Accuracy | Avg Tokens | Avg Latency (ms) | Avg Turns |")
    lines.append("|-------|------|-----------|---------|----------|------------|-------------------|-----------|")
    for sr in report.suites:
        lines.append(
            f"| {sr.suite.value} | {sr.mode.value} | {sr.total} | {sr.correct} | "
            f"{sr.accuracy:.1%} | {sr.avg_tokens:.0f} | {sr.avg_latency_ms:.0f} | {sr.avg_turns:.1f} |"
        )
    lines.append("")

    # ── Comparison table: Zero-shot vs YOHAS Full vs Published Baselines ──
    lines.append("## Comparison: Zero-shot vs YOHAS Full vs Published Baselines")
    lines.append("")

    # Group results by suite
    by_suite: dict[str, dict[str, SuiteResults]] = {}
    for sr in report.suites:
        key = sr.suite.value
        if key not in by_suite:
            by_suite[key] = {}
        by_suite[key][sr.mode.value] = sr

    # Build comparison table
    header_cols = ["Suite", "Zero-shot (Opus)", "YOHAS Full"]
    # Collect all unique baseline names
    all_baselines: set[str] = set()
    for baselines in report.baselines.values():
        all_baselines.update(baselines.keys())
    sorted_baselines = sorted(all_baselines)
    header_cols.extend(sorted_baselines)

    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join("---" for _ in header_cols) + "|")

    for suite_name, modes in by_suite.items():
        row = [suite_name]
        zs = modes.get("zero_shot")
        yf = modes.get("yohas_full")
        row.append(f"{zs.accuracy:.1%}" if zs else "—")
        row.append(f"{yf.accuracy:.1%}" if yf else "—")

        baselines = report.baselines.get(suite_name, {})
        for bl_name in sorted_baselines:
            val = baselines.get(bl_name)
            row.append(f"{val:.1%}" if val is not None else "—")

        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Per-category breakdown ──
    lines.append("## Per-Category Breakdown")
    lines.append("")

    for sr in report.suites:
        if not sr.by_category:
            continue
        lines.append(f"### {sr.suite.value} ({sr.mode.value})")
        lines.append("")
        lines.append("| Category | Total | Correct | Accuracy | Avg Tokens | Avg Latency |")
        lines.append("|----------|-------|---------|----------|------------|-------------|")
        for cat, metrics in sorted(sr.by_category.items()):
            lines.append(
                f"| {cat} | {metrics['total']:.0f} | {metrics['correct']:.0f} | "
                f"{metrics['accuracy']:.1%} | {metrics['avg_tokens']:.0f} | {metrics['avg_latency_ms']:.0f}ms |"
            )
        lines.append("")

    # ── Failure analysis ──
    lines.append("## Failure Analysis")
    lines.append("")

    for sr in report.suites:
        failed = [r for r in sr.results if r.status in (InstanceStatus.FAILED, InstanceStatus.TIMEOUT)]
        if not failed:
            continue
        lines.append(f"### {sr.suite.value} ({sr.mode.value}): {len(failed)} failures")
        lines.append("")
        for r in failed[:10]:  # Show first 10
            lines.append(f"- **{r.instance_id}**: {r.status.value} — {r.error or 'unknown'}")
        if len(failed) > 10:
            lines.append(f"- ... and {len(failed) - 10} more")
        lines.append("")

    # ── Trajectory stats ──
    lines.append("## Trajectory Statistics")
    lines.append("")
    for sr in report.suites:
        if sr.mode == RunMode.YOHAS_FULL:
            tools_used_counts: dict[str, int] = {}
            for r in sr.results:
                for t in r.tools_used:
                    tools_used_counts[t] = tools_used_counts.get(t, 0) + 1
            if tools_used_counts:
                lines.append(f"### {sr.suite.value} — Tool Usage")
                lines.append("")
                lines.append("| Tool | Times Used |")
                lines.append("|------|-----------|")
                for tool, count in sorted(tools_used_counts.items(), key=lambda x: -x[1]):
                    lines.append(f"| {tool} | {count} |")
                lines.append("")

    return "\n".join(lines)


def _infer_category(instance_id: str) -> str:
    """Infer category from instance ID prefix."""
    # E.g., "biomni_0042" -> "biomni", "dbqa_0012" -> "dbqa"
    parts = instance_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else instance_id
