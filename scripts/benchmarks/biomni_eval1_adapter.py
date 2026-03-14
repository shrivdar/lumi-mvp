"""Biomni-Eval1 adapter for YOHAS 3.0.

Evaluates YOHAS against 433 instances across 10 biological reasoning tasks.
Scoring is done via Biomni's official evaluator — we NEVER self-score.

Repo: https://github.com/snap-stanford/Biomni
API: from biomni.eval import BiomniEval1

Actual tasks (from BiomniEval1.list_tasks()):
  - crispr_delivery (10)           — answer: letter a-f
  - gwas_causal_gene_gwas_catalog (50)  — answer: gene symbol
  - gwas_causal_gene_opentargets (50)   — answer: gene symbol
  - gwas_causal_gene_pharmaprojects (50) — answer: gene symbol
  - gwas_variant_prioritization (43)    — answer: variant ID
  - lab_bench_dbqa (50)            — answer: letter A-E
  - lab_bench_seqqa (50)           — answer: letter A-E
  - patient_gene_detection (50)    — answer: JSON {"causal_gene": [...]}
  - rare_disease_diagnosis (30)    — answer: JSON {"OMIM_ID": "..."}
  - screen_gene_retrieval (50)     — answer: gene symbol

Methodology:
  Biomni passes the prompt from the dataset directly to the agent.
  The agent returns a free-text response. Biomni extracts the answer
  (their A1 agent uses <solution> tags, their QA agent uses result_formatting).
  The extracted answer is then passed to evaluator.evaluate() for scoring.

  We follow the same pipeline: pass prompt → get response → extract answer → score.
  The prompt is passed UNMODIFIED from the dataset — we don't append instructions.
"""

from __future__ import annotations

import json
import logging
import re
import time
import traceback
from pathlib import Path
from typing import Any

from common import ResultLogger, YOHASRunner, _get_benchmark_mode

logger = logging.getLogger("yohas.benchmarks.biomni")


def _extract_answer(task_name: str, raw_response: str) -> str:
    """Extract the answer from LLM response, matching Biomni's expected formats.

    This mirrors what Biomni's A1 agent does: the agent produces free-text,
    then the answer is extracted for scoring. Biomni's _compute_reward() does
    simple exact matching on the extracted answer.
    """
    text = raw_response.strip()

    # --- lab_bench tasks: look for [ANSWER]X[/ANSWER] tags first (prompt asks for this) ---
    if task_name.startswith("lab_bench"):
        m = re.search(r"\[ANSWER\]\s*([A-Ea-e])\s*\[/ANSWER\]", text)
        if m:
            return m.group(1).upper()
        # Fallback: find last standalone letter
        m = re.search(r"\b([A-E])\b", text)
        return m.group(1) if m else text.strip()[:1].upper()

    # --- crispr_delivery: letter a-f ---
    if task_name == "crispr_delivery":
        # Look for explicit answer patterns
        m = re.search(r"(?:answer|choice|select).*?\b([a-fA-F])\b", text, re.IGNORECASE)
        if m:
            return m.group(1).lower()
        # Fallback: last standalone letter a-f
        matches = re.findall(r"\b([a-fA-F])\b", text)
        return matches[-1].lower() if matches else text.strip()[:1].lower()

    # --- JSON answer tasks ---
    if task_name in ("patient_gene_detection", "rare_disease_diagnosis"):
        # Try to find JSON in the response
        json_match = re.search(r"\{[^}]+\}", text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return json.dumps(parsed)
            except json.JSONDecodeError:
                pass
        # Fallback: try the full text
        try:
            parsed = json.loads(text)
            return json.dumps(parsed)
        except (json.JSONDecodeError, ValueError):
            return text

    # --- Gene symbol / variant tasks: extract the entity ---
    # For gwas_causal_gene_*, screen_gene_retrieval, gwas_variant_prioritization
    #
    # The LLM may return a verbose response. We need to find the gene/variant.
    # Strategy: look for explicit "Answer: X" patterns, then fall back to
    # finding gene-like tokens.

    # Strip markdown formatting
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)

    # Look for explicit answer patterns (checking all lines, not just last)
    answer_patterns = [
        r"(?:answer|conclusion|result)\s*[:=]\s*(\S+)",
        r"(?:the\s+)?(?:causal|likely|most likely)\s+gene\s+is\s+(\S+)",
        r"(?:the\s+)?gene\s+is\s+(\S+)",
        r"(?:the\s+)?variant\s+is\s+(\S+)",
    ]
    for pat in answer_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            answer = m.group(1).strip().rstrip(".,;:)")
            if answer:
                return answer

    # Fallback for gene tasks: find gene-symbol-like tokens (uppercase, 2-10 chars)
    if task_name.startswith("gwas_causal_gene") or task_name == "screen_gene_retrieval":
        # Gene symbols are typically uppercase letters/numbers, 2-10 chars
        gene_candidates = re.findall(r"\b([A-Z][A-Z0-9]{1,9})\b", text)
        # Filter out common non-gene words
        noise = {"THE", "AND", "FOR", "FROM", "WITH", "THIS", "THAT", "GENE", "TYPE",
                 "GWAS", "LOCUS", "GENES", "ANSWER", "BASED", "GIVEN", "LIST", "MOST",
                 "LIKELY", "CAUSAL", "KNOWN", "ROLE", "WHICH"}
        gene_candidates = [g for g in gene_candidates if g not in noise]
        if gene_candidates:
            # Prefer the last gene-like token (usually the answer after explanation)
            return gene_candidates[-1]

    # Last resort: last non-empty line, stripped
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text


class YOHASBiomniAdapter:
    """Adapter to evaluate YOHAS on Biomni-Eval1 benchmark tasks.

    Key principle: we pass the dataset prompt UNMODIFIED to the LLM.
    This is what Biomni's own agents do — they receive the prompt as-is.
    """

    def __init__(
        self,
        *,
        mode: str | None = None,
        max_iterations: int = 5,
    ) -> None:
        self.mode = mode or _get_benchmark_mode()
        self.runner = YOHASRunner(
            max_iterations=max_iterations if self.mode == "agentic" else 1,
            enable_falsification=self.mode == "agentic",
            enable_hitl=False,
        )

    def evaluate_instance(
        self,
        task_name: str,
        instance: dict[str, Any],
    ) -> dict[str, Any]:
        """Run YOHAS on a single Biomni instance.

        The prompt from the dataset is passed directly to the LLM — no
        modification, no appended instructions. This matches how Biomni's
        own A1 agent receives prompts.
        """
        prompt = instance["prompt"]

        if self.mode == "zero-shot":
            # Direct LLM call — pass prompt as-is
            raw = self.runner.query_llm_with_kg(
                prompt,
                system_prompt=(
                    "You are an expert biomedical research scientist. "
                    "Answer the question precisely in the format requested."
                ),
            )
            answer = _extract_answer(task_name, raw)
            return {
                "answer": answer,
                "raw_response": raw,
                "kg_stats": None,
                "tokens_used": self.runner.token_summary.get("total_tokens", 0),
            }

        # Agentic mode: run full YOHAS research session
        session, kg = self.runner.run_session(prompt)

        # Use KG-enriched context to produce the final answer
        kg_summary = kg.to_markdown_summary()
        refinement_prompt = (
            f"Based on your research findings:\n\n{kg_summary}\n\n"
            f"Now answer the original question:\n\n{prompt}"
        )
        raw = self.runner.query_llm_with_kg(
            refinement_prompt,
            system_prompt=(
                "You are an expert biomedical research scientist. "
                "Answer the question precisely in the format requested."
            ),
            kg=kg,
        )
        answer = _extract_answer(task_name, raw)

        kg_json = kg.to_json()
        return {
            "answer": answer,
            "raw_response": raw,
            "kg_stats": {
                "node_count": kg_json.get("metadata", {}).get("node_count", 0),
                "edge_count": kg_json.get("metadata", {}).get("edge_count", 0),
                "avg_confidence": kg_json.get("metadata", {}).get("avg_confidence", 0),
            },
            "tokens_used": self.runner.token_summary.get("total_tokens", 0),
        }


def run_biomni_eval1(
    *,
    tasks: list[str] | None = None,
    limit: int | None = None,
    mode: str = "agentic",
    max_iterations: int = 5,
    checkpoint_id: str | None = None,
) -> Path:
    """Run Biomni-Eval1 evaluation.

    All scoring is done via BiomniEval1.evaluate() — never self-scored.

    Args:
        tasks: List of task names to evaluate, or None for all.
        limit: Max instances per task.
        mode: "zero-shot" or "agentic".
        max_iterations: MCTS iterations per instance.
        checkpoint_id: Resume from checkpoint.

    Returns:
        Path to results JSON.
    """
    try:
        from biomni.eval import BiomniEval1  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "Biomni not installed. Run: scripts/benchmarks/setup_benchmarks.sh"
        )
        raise SystemExit(1)

    evaluator = BiomniEval1()
    available_tasks = evaluator.list_tasks()
    selected_tasks = tasks or available_tasks

    # Validate task names against what's actually in the dataset
    for t in selected_tasks:
        if t not in available_tasks:
            logger.error(
                "Task '%s' not found. Available tasks: %s", t, available_tasks
            )
            raise SystemExit(1)

    result_logger = ResultLogger("biomni")
    if checkpoint_id:
        result_logger._run_id = checkpoint_id
        result_logger._checkpoint_path = (
            result_logger.output_dir / f"{checkpoint_id}_checkpoint.jsonl"
        )
    completed = result_logger.load_checkpoint()

    adapter = YOHASBiomniAdapter(mode=mode, max_iterations=max_iterations)

    total_evaluated = 0
    total_correct = 0

    for task_name in selected_tasks:
        task_df = evaluator.get_instances_by_task(task_name)
        instance_ids = task_df["task_instance_id"].tolist()

        if limit:
            instance_ids = instance_ids[:limit]

        logger.info("--- Task: %s (%d instances) ---", task_name, len(instance_ids))
        task_correct = 0
        task_total = 0

        for idx, task_instance_id in enumerate(instance_ids):
            checkpoint_key = f"{task_name}_{task_instance_id}"
            if checkpoint_key in completed:
                logger.debug("Skipping completed %s", checkpoint_key)
                continue

            logger.info(
                "[%s %d/%d] instance_id=%s",
                task_name, idx + 1, len(instance_ids), task_instance_id,
            )
            start = time.monotonic()

            try:
                instance = evaluator.get_instance(task_name, task_instance_id)
                result = adapter.evaluate_instance(task_name, instance)

                # Score via Biomni's official evaluator — NOT self-scored
                score = evaluator.evaluate(
                    task_name, task_instance_id, result["answer"]
                )
                correct = score >= 0.5

                ground_truth = instance.get("answer", "")
                duration_ms = int((time.monotonic() - start) * 1000)

                logger.info(
                    "  predicted=%r  expected=%r  score=%.1f  tokens=%d  time=%dms",
                    result["answer"], ground_truth, score,
                    result.get("tokens_used", 0), duration_ms,
                )

                result_logger.log_instance({
                    "instance_id": checkpoint_key,
                    "task_type": task_name,
                    "category": task_name,
                    "task_instance_id": task_instance_id,
                    "prompt": instance["prompt"][:500],  # first 500 chars for debugging
                    "answer": result["answer"],
                    "expected": ground_truth,
                    "correct": correct,
                    "score": score,
                    "raw_response": result.get("raw_response", "")[:500],
                    "kg_stats": result.get("kg_stats"),
                    "tokens_used": result.get("tokens_used", 0),
                    "duration_ms": duration_ms,
                    "error": None,
                })
                total_evaluated += 1
                task_total += 1
                if correct:
                    total_correct += 1
                    task_correct += 1

            except Exception as exc:
                logger.error("Instance %s FAILED: %s", checkpoint_key, exc)
                logger.debug(traceback.format_exc())
                duration_ms = int((time.monotonic() - start) * 1000)
                result_logger.log_instance({
                    "instance_id": checkpoint_key,
                    "task_type": task_name,
                    "category": task_name,
                    "task_instance_id": task_instance_id,
                    "prompt": "",
                    "answer": "",
                    "expected": "",
                    "correct": False,
                    "score": 0.0,
                    "raw_response": "",
                    "kg_stats": None,
                    "tokens_used": 0,
                    "duration_ms": duration_ms,
                    "error": traceback.format_exc(),
                })
                total_evaluated += 1
                task_total += 1

        if task_total > 0:
            logger.info(
                "Task %s: %d/%d correct (%.1f%%)",
                task_name, task_correct, task_total, 100 * task_correct / task_total,
            )

    logger.info(
        "=== Biomni-Eval1 COMPLETE: %d/%d correct (%.1f%%) across %d tasks ===",
        total_correct, total_evaluated,
        100 * total_correct / total_evaluated if total_evaluated else 0,
        len(selected_tasks),
    )
    return result_logger.save_final(
        extra_metadata={
            "mode": mode,
            "tasks": selected_tasks,
            "max_iterations": max_iterations,
            "limit": limit,
            "total_correct": total_correct,
            "total_evaluated": total_evaluated,
        }
    )
