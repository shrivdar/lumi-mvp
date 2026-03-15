"""Batch trajectory collection — run benchmark instances through agents and save trajectories.

Wires together:
  - benchmarks/adapters (load Biomni-Eval1 instances)
  - benchmarks/evaluator (run instances through agents)
  - rl/trajectory_collector (capture structured trajectories)

Usage:
    python -m rl.collect_at_scale                                # 50 instances, dry-run
    python -m rl.collect_at_scale --limit 100 --live             # 100 instances, real LLM
    python -m rl.collect_at_scale --suite biomni_eval1 --limit 50 --concurrency 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from benchmarks.adapters import get_adapter
from benchmarks.evaluator import BenchmarkEvaluator
from benchmarks.models import (
    BenchmarkInstance,
    BenchmarkSuite,
    InstanceResult,
    InstanceStatus,
    RunMode,
)
from rl.trajectory_format import (
    Trajectory,
    Turn,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "trajectories"


def _benchmark_trajectory_to_rl(
    instance: BenchmarkInstance,
    result: InstanceResult,
    bench_trajectory: dict | None = None,
) -> Trajectory:
    """Convert a benchmark InstanceResult into an RL Trajectory for training.

    This bridges the benchmark evaluation format (step-based) with the RL format
    (turn-based with tool calls, KG mutations, etc.).
    """
    turns: list[Turn] = []

    # Build instruction turn from the benchmark question
    turns.append(Turn(
        turn_number=0,
        role="system",
        content=f"Task: {instance.question}",
        turn_type="instruction",
    ))

    # Build turns from the reasoning trace or trajectory steps
    if result.reasoning_trace:
        turns.append(Turn(
            turn_number=1,
            role="assistant",
            content=result.reasoning_trace,
            turn_type="reasoning",
            tokens_used=result.tokens_used,
        ))

    # If we have tool usage, add tool turns
    for i, tool in enumerate(result.tools_used):
        turns.append(Turn(
            turn_number=len(turns),
            role="assistant",
            content=f"Using tool: {tool}",
            turn_type="tool_call",
        ))

    # Final answer turn
    if result.predicted:
        turns.append(Turn(
            turn_number=len(turns),
            role="assistant",
            content=result.predicted,
            turn_type="answer",
        ))

    reward = result.score if result.status == InstanceStatus.COMPLETED else 0.0

    return Trajectory(
        task_id=result.instance_id,
        research_id=f"bench_{instance.suite.value}",
        agent_type="benchmark_agent",
        agent_id=f"bench_{result.mode.value}",
        instruction=instance.question,
        context={
            "suite": instance.suite.value,
            "category": instance.category,
            "choices": instance.choices,
            "ground_truth": instance.ground_truth,
            "instance_context": instance.context,
        },
        turns=turns,
        final_answer=result.predicted,
        reward=reward,
        success=result.correct,
        token_usage={"total": result.tokens_used},
        wall_time_ms=result.latency_ms,
        total_tokens=result.tokens_used,
        benchmark_run_id=f"{instance.suite.value}_{result.mode.value}",
    )


async def collect_trajectories(
    *,
    suite: BenchmarkSuite = BenchmarkSuite.BIOMNI_EVAL1,
    mode: RunMode = RunMode.YOHAS_FULL,
    limit: int = 50,
    concurrency: int = 5,
    timeout: int = 300,
    live: bool = False,
    output_dir: Path | None = None,
    seed: int = 42,
) -> Path:
    """Run benchmark instances and collect trajectories for RL training.

    Args:
        suite: Which benchmark suite to use.
        mode: Evaluation mode (zero_shot or yohas_full).
        limit: Max instances to evaluate.
        concurrency: Max parallel evaluations.
        timeout: Per-instance timeout in seconds.
        live: Whether to use real LLM calls.
        output_dir: Where to save trajectory JSONL files.
        seed: Random seed for reproducibility.

    Returns:
        Path to the output JSONL file.
    """
    import random

    random.seed(seed)

    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark instances
    adapter = get_adapter(suite)
    instances = adapter.load_instances(limit=limit)
    logger.info(
        "Loaded %d instances from %s (requested %d)",
        len(instances), suite.value, limit,
    )

    # Set up evaluator
    llm = None
    orchestrator_factory = None
    if live:
        from core.llm import LLMClient
        llm = LLMClient()

    evaluator = BenchmarkEvaluator(
        mode=mode,
        llm=llm,
        orchestrator_factory=orchestrator_factory,
        max_concurrency=concurrency,
        timeout_seconds=timeout,
        collect_trajectories=True,
    )

    # Run evaluation
    completed = 0
    failed = 0

    def _progress(done: int, total: int, result: InstanceResult) -> None:
        nonlocal completed, failed
        if result.correct:
            completed += 1
        else:
            failed += 1
        if done % 10 == 0 or done == total:
            logger.info(
                "Progress: %d/%d (correct=%d, failed=%d)",
                done, total, completed, failed,
            )

    logger.info("Starting trajectory collection: %d instances, mode=%s", len(instances), mode.value)
    start_time = time.monotonic()

    results = await evaluator.evaluate_batch(instances, progress_callback=_progress)

    elapsed = time.monotonic() - start_time
    logger.info(
        "Evaluation complete in %.1fs — %d results, %d correct",
        elapsed, len(results), sum(1 for r in results if r.correct),
    )

    # Convert benchmark results to RL trajectories
    rl_trajectories: list[Trajectory] = []
    for inst, result in zip(instances, results):
        if result.status == InstanceStatus.COMPLETED:
            traj = _benchmark_trajectory_to_rl(inst, result)
            rl_trajectories.append(traj)

    logger.info("Converted %d results to RL trajectories", len(rl_trajectories))

    # Write trajectories to JSONL
    ts = int(time.time())
    filename = f"{suite.value}_{mode.value}_{ts}.jsonl"
    out_path = out_dir / filename

    with open(out_path, "w") as f:
        for traj in rl_trajectories:
            f.write(traj.model_dump_json() + "\n")

    logger.info("Wrote %d trajectories to %s", len(rl_trajectories), out_path)

    # Write collection metadata
    meta = {
        "suite": suite.value,
        "mode": mode.value,
        "instances_loaded": len(instances),
        "results_total": len(results),
        "results_completed": sum(1 for r in results if r.status == InstanceStatus.COMPLETED),
        "results_correct": sum(1 for r in results if r.correct),
        "results_failed": sum(1 for r in results if r.status == InstanceStatus.FAILED),
        "results_timeout": sum(1 for r in results if r.status == InstanceStatus.TIMEOUT),
        "trajectories_saved": len(rl_trajectories),
        "output_file": str(out_path),
        "elapsed_seconds": round(elapsed, 1),
        "live": live,
        "seed": seed,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote metadata to %s", meta_path)

    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="YOHAS Batch Trajectory Collection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=[s.value for s in BenchmarkSuite],
        default="biomni_eval1",
        help="Benchmark suite to collect from",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[m.value for m in RunMode],
        default="yohas_full",
        help="Evaluation mode",
    )
    parser.add_argument("--limit", type=int, default=50, help="Max instances to evaluate")
    parser.add_argument("--concurrency", type=int, default=5, help="Max parallel evaluations")
    parser.add_argument("--timeout", type=int, default=300, help="Per-instance timeout (seconds)")
    parser.add_argument("--live", action="store_true", help="Use real LLM calls")
    parser.add_argument("--output-dir", type=Path, default=None, help="Trajectory output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(
        collect_trajectories(
            suite=BenchmarkSuite(args.suite),
            mode=RunMode(args.mode),
            limit=args.limit,
            concurrency=args.concurrency,
            timeout=args.timeout,
            live=args.live,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
