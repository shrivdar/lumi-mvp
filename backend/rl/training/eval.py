"""Checkpoint evaluation on Biomni-Eval1 benchmark.

Loads a fine-tuned checkpoint, runs it against the evaluation dataset,
and computes the multi-dimensional reward breakdown + aggregate metrics.

Usage:
    python -m rl.training.eval --checkpoint checkpoints/rl/final
    python -m rl.training.eval --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path

from rl.training.config import EvalConfig
from rl.training.reward import TrajectoryContext, compute_reward

logger = logging.getLogger(__name__)


def _load_eval_dataset(
    dataset_path: Path,
    max_samples: int | None = None,
) -> list[dict]:
    """Load evaluation dataset.

    Expected format per line:
        {
            "prompt": "...",
            "reference_answer": "...",
            "metadata": {...}
        }
    """
    samples: list[dict] = []
    if not dataset_path.exists():
        logger.warning("Eval dataset %s does not exist — generating synthetic samples for dry run", dataset_path)
        return _generate_synthetic_samples(max_samples or 5)

    for jsonl_file in sorted(dataset_path.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
                if max_samples and len(samples) >= max_samples:
                    return samples
    return samples


def _generate_synthetic_samples(n: int) -> list[dict]:
    """Generate minimal synthetic eval samples for dry-run testing."""
    samples = []
    queries = [
        "What are the molecular mechanisms of B7-H3 in NSCLC immune evasion?",
        "Identify drug targets in the PI3K/AKT pathway for triple-negative breast cancer.",
        "What is the role of BRCA1 in homologous recombination repair?",
        "Analyze the druggability of KRAS G12C in pancreatic adenocarcinoma.",
        "How does PD-L1 expression correlate with checkpoint inhibitor response?",
    ]
    for i in range(min(n, len(queries))):
        samples.append({
            "prompt": queries[i],
            "reference_answer": f"Reference answer for query {i}",
            "metadata": {"category": "biomedical_qa", "difficulty": "hard"},
        })
    return samples


def _evaluate_response(
    response: str,
    reference: str,
) -> float:
    """Simple evaluator score based on overlap (placeholder for LLM-as-judge).

    In production, this would use an LLM evaluator or domain-specific metrics.
    Returns a score in [0, 1].
    """
    if not response or not reference:
        return 0.0

    response_tokens = set(response.lower().split())
    reference_tokens = set(reference.lower().split())

    if not reference_tokens:
        return 0.0

    overlap = len(response_tokens & reference_tokens)
    precision = overlap / len(response_tokens) if response_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def run_eval(config: EvalConfig) -> dict:
    """Run evaluation on a checkpoint.

    Returns:
        Dict with aggregate metrics and per-sample breakdowns.
    """
    logger.info("Starting evaluation")
    logger.info("  Checkpoint: %s", config.checkpoint_path)
    logger.info("  Base model: %s", config.base_model)
    logger.info("  Eval dataset: %s", config.eval_dataset)
    logger.info("  Dry run: %s", config.dry_run)

    max_samples = config.dry_run_samples if config.dry_run else None
    eval_data = _load_eval_dataset(config.eval_dataset_path, max_samples=max_samples)
    logger.info("  Eval samples: %d", len(eval_data))

    if config.dry_run:
        return _run_dry_eval(config, eval_data)

    # ------------------------------------------------------------------
    # Real eval path — requires torch + transformers
    # ------------------------------------------------------------------
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.error(
            "Evaluation requires: torch, transformers. "
            "Install with: pip install torch transformers\n%s",
            e,
        )
        raise

    tokenizer = AutoTokenizer.from_pretrained(str(config.checkpoint_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(config.checkpoint_path),
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    results = []
    for i, sample in enumerate(eval_data):
        prompt = sample["prompt"]
        reference = sample.get("reference_answer", "")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                do_sample=config.temperature > 0,
                num_return_sequences=config.num_samples,
            )

        response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        evaluator_score = _evaluate_response(response, reference)

        from core.models import AgentResult, AgentType

        dummy_result = AgentResult(
            task_id=f"eval-{i}",
            agent_id="eval-agent",
            agent_type=AgentType.LITERATURE_ANALYST,
            reasoning_trace=response,
            summary=response,
        )
        ctx = TrajectoryContext(
            evaluator_score=evaluator_score,
            expected_format="text",
        )
        reward = compute_reward(dummy_result, ctx, config.reward_weights)

        results.append({
            "sample_id": i,
            "prompt": prompt,
            "response": response[:500],
            "reference": reference[:200],
            "evaluator_score": evaluator_score,
            "reward": reward.as_dict(),
        })

        if i % 10 == 0:
            logger.info("  Evaluated %d/%d samples", i + 1, len(eval_data))

    return _compile_results(config, results)


def _run_dry_eval(config: EvalConfig, eval_data: list[dict]) -> dict:
    """Dry-run evaluation without model loading."""
    logger.info("[DRY RUN] Simulating evaluation")

    from core.models import AgentResult, AgentType

    results = []
    for i, sample in enumerate(eval_data):
        # Simulate a response
        simulated_response = f"Simulated response for: {sample['prompt'][:50]}..."
        reference = sample.get("reference_answer", "")
        evaluator_score = _evaluate_response(simulated_response, reference)

        dummy_result = AgentResult(
            task_id=f"eval-{i}",
            agent_id="eval-agent",
            agent_type=AgentType.LITERATURE_ANALYST,
            reasoning_trace=simulated_response,
            summary=simulated_response,
        )
        ctx = TrajectoryContext(
            evaluator_score=evaluator_score,
            expected_format="text",
        )
        reward = compute_reward(dummy_result, ctx, config.reward_weights)

        results.append({
            "sample_id": i,
            "prompt": sample["prompt"],
            "response": simulated_response,
            "reference": reference[:200],
            "evaluator_score": evaluator_score,
            "reward": reward.as_dict(),
        })

    report = _compile_results(config, results)
    report["status"] = "dry_run"

    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.output_dir / "dry_run_eval.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("[DRY RUN] Wrote eval report to %s", report_path)

    return report


def _compile_results(config: EvalConfig, results: list[dict]) -> dict:
    """Compile per-sample results into aggregate metrics."""
    if not results:
        return {"status": "no_samples", "metrics": {}}

    total_rewards = [r["reward"]["total"] for r in results]
    r1_scores = [r["reward"]["r1_answer_correctness"] for r in results]
    r2_scores = [r["reward"]["r2_kg_quality"] for r in results]
    r3_scores = [r["reward"]["r3_hypothesis_efficiency"] for r in results]
    r4_scores = [r["reward"]["r4_falsification_quality"] for r in results]
    r5_scores = [r["reward"]["r5_format_compliance"] for r in results]

    metrics = {
        "total_samples": len(results),
        "avg_total_reward": statistics.mean(total_rewards),
        "std_total_reward": statistics.stdev(total_rewards) if len(total_rewards) > 1 else 0.0,
        "avg_r1_correctness": statistics.mean(r1_scores),
        "avg_r2_kg_quality": statistics.mean(r2_scores),
        "avg_r3_efficiency": statistics.mean(r3_scores),
        "avg_r4_falsification": statistics.mean(r4_scores),
        "avg_r5_format": statistics.mean(r5_scores),
    }

    return {
        "checkpoint": str(config.checkpoint_path),
        "base_model": config.base_model,
        "eval_dataset": config.eval_dataset,
        "metrics": metrics,
        "samples": results,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="YOHAS Checkpoint Evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Run without loading model")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path")
    parser.add_argument(
        "--base-model",
        choices=["qwen", "llama"],
        default="qwen",
        help="Base model selection",
    )
    parser.add_argument("--eval-dataset", type=Path, default=None, help="Eval dataset path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for results")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    config = EvalConfig()
    if args.dry_run:
        config.dry_run = True
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if args.base_model == "llama":
        config.base_model = "meta-llama/Llama-3.1-8B-Instruct"
    if args.eval_dataset:
        config.eval_dataset_path = args.eval_dataset
    if args.output_dir:
        config.output_dir = args.output_dir

    report = run_eval(config)

    logger.info("Evaluation complete")
    logger.info("  Total samples: %d", report.get("metrics", {}).get("total_samples", 0))
    logger.info("  Avg reward: %.4f", report.get("metrics", {}).get("avg_total_reward", 0.0))


if __name__ == "__main__":
    main()
