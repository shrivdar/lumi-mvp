"""Benchmark runner — CLI entry-point for the full benchmark suite.

Usage:
    python -m benchmarks.runner                     # full suite, dry-run
    python -m benchmarks.runner --suite biomni_eval1 --limit 10
    python -m benchmarks.runner --mode yohas_full --live
    python -m benchmarks.runner --output results/bench_report.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from benchmarks.adapters import get_adapter
from benchmarks.evaluator import BenchmarkEvaluator
from benchmarks.models import (
    BenchmarkSuite,
    InstanceResult,
    RunMode,
    SuiteResults,
)
from benchmarks.report import aggregate_results, generate_report
from benchmarks.trajectory_store import TrajectoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_tool_instances() -> dict:
    """Import and instantiate tool classes for live orchestrator."""
    tools: dict = {}
    _tool_defs = [
        ("pubmed", "integrations.pubmed", "PubMedTool"),
        ("uniprot", "integrations.uniprot", "UniProtTool"),
        ("clinicaltrials", "integrations.clinicaltrials", "ClinicalTrialsTool"),
        ("python_repl", "integrations.python_repl", "PythonREPLTool"),
    ]
    for name, module_path, class_name in _tool_defs:
        try:
            import importlib

            mod = importlib.import_module(module_path)
            tools[name] = getattr(mod, class_name)()
        except Exception:
            pass
    return tools

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOHAS 3.0 Benchmark Runner")
    p.add_argument(
        "--suite",
        type=str,
        nargs="*",
        choices=[s.value for s in BenchmarkSuite],
        help="Which suites to run (default: all)",
    )
    p.add_argument(
        "--mode",
        type=str,
        nargs="*",
        choices=[m.value for m in RunMode],
        default=["zero_shot", "yohas_full"],
        help="Evaluation modes (default: both)",
    )
    p.add_argument("--limit", type=int, default=None, help="Limit instances per suite")
    p.add_argument("--concurrency", type=int, default=5, help="Max concurrent evaluations")
    p.add_argument("--timeout", type=int, default=300, help="Per-instance timeout (seconds)")
    p.add_argument("--live", action="store_true", help="Use live LLM calls (requires API keys)")
    p.add_argument("--output", type=str, default=None, help="Output path for markdown report")
    p.add_argument("--trajectories-dir", type=str, default=None, help="Trajectory output directory")
    p.add_argument("--no-trajectories", action="store_true", help="Disable trajectory collection")
    p.add_argument("--max-trials", type=int, default=1, help="Number of trials per instance (multi-trial protocol)")
    p.add_argument("--strategy-dir", type=str, default=None, help="Directory for strategy template storage")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


async def run_benchmark(args: argparse.Namespace) -> None:
    """Run the full benchmark suite."""
    import random

    random.seed(args.seed)

    suites = [BenchmarkSuite(s) for s in args.suite] if args.suite else list(BenchmarkSuite)
    modes = [RunMode(m) for m in args.mode]

    logger.info(
        "Starting benchmark: suites=%s, modes=%s, limit=%s, live=%s",
        [s.value for s in suites],
        [m.value for m in modes],
        args.limit,
        args.live,
    )

    # Set up LLM and orchestrator for live mode
    llm = None
    orchestrator_factory = None
    if args.live:
        from core.llm import LLMClient

        llm = LLMClient()

        def make_orchestrator():
            from agents.factory import create_agent
            from integrations.tool_catalog import get_catalog
            from orchestrator.research_loop import ResearchOrchestrator
            from world_model.knowledge_graph import InMemoryKnowledgeGraph

            kg = InMemoryKnowledgeGraph()
            tool_entries = get_catalog()
            tool_instances = _build_tool_instances()
            return ResearchOrchestrator(
                llm=llm,
                kg=kg,
                agent_factory=create_agent,
                tool_entries=tool_entries,
                tool_instances=tool_instances,
            )

        orchestrator_factory = make_orchestrator

    from benchmarks.strategy_memory import StrategyMemory

    trajectory_store = TrajectoryStore(args.trajectories_dir) if not args.no_trajectories else None
    strategy_memory = StrategyMemory(storage_dir=args.strategy_dir) if args.strategy_dir else StrategyMemory()
    all_suite_results: list[SuiteResults] = []

    for suite in suites:
        adapter = get_adapter(suite)
        instances = adapter.load_instances(limit=args.limit)
        logger.info("Loaded %d instances for %s", len(instances), suite.value)

        for mode in modes:
            logger.info("Evaluating %s in %s mode...", suite.value, mode.value)

            evaluator = BenchmarkEvaluator(
                mode=mode,
                llm=llm,
                orchestrator_factory=orchestrator_factory if mode in (RunMode.YOHAS_FULL, RunMode.CODE_FIRST) else None,
                max_concurrency=args.concurrency,
                timeout_seconds=args.timeout,
                collect_trajectories=not args.no_trajectories,
                max_trials=args.max_trials,
                strategy_memory=strategy_memory,
            )

            def _progress(done: int, total: int, result: InstanceResult) -> None:
                status = "correct" if result.correct else "wrong"
                logger.info(
                    "  [%d/%d] %s — %s (%dms)",
                    done,
                    total,
                    result.instance_id,
                    status,
                    result.latency_ms,
                )

            results = await evaluator.evaluate_batch(instances, progress_callback=_progress)
            suite_result = aggregate_results(results, suite, mode)
            all_suite_results.append(suite_result)

            logger.info(
                "%s %s: %d/%d correct (%.1f%%)",
                suite.value,
                mode.value,
                suite_result.correct,
                suite_result.total,
                suite_result.accuracy * 100,
            )

            # Save trajectories
            if trajectory_store and evaluator.trajectories:
                trajectory_store.save(
                    evaluator.trajectories,
                    tag=f"{suite.value}_{mode.value}",
                )

    # Generate report
    report = generate_report(all_suite_results)

    # Write report
    output_path = Path(args.output) if args.output else RESULTS_DIR / "benchmark_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.markdown)
    logger.info("Report written to %s", output_path)

    # Also write raw JSON results
    json_path = output_path.with_suffix(".json")
    json_data = {
        "generated_at": report.generated_at.isoformat(),
        "suites": [sr.model_dump(mode="json") for sr in all_suite_results],
        "baselines": report.baselines,
    }
    json_path.write_text(json.dumps(json_data, indent=2, default=str))
    logger.info("Raw results written to %s", json_path)

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    for sr in all_suite_results:
        print(f"  {sr.suite.value:25s} {sr.mode.value:12s}  {sr.accuracy:.1%}  ({sr.correct}/{sr.total})")
    print("=" * 60)

    if trajectory_store:
        summary = trajectory_store.summary()
        print(f"\nTrajectories: {summary['total_trajectories']} saved ({summary['total_steps']} steps)")
        print(f"  Output dir: {summary['output_dir']}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
