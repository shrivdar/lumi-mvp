"""LAB-Bench adapter for YOHAS 3.0.

Evaluates YOHAS against 2,457 multiple-choice questions across 8 categories.
Prioritizes categories where YOHAS has tool advantages:
  - DbQA (11 database tools)
  - SeqQA (ESM-2/ESMFold)
  - LitQA2 (PubMed + Semantic Scholar)

Repo: https://github.com/Future-House/LAB-Bench
Interface: agent_fn(input: labbench.AgentInput) -> str
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Any

from common import (
    AnswerExtractor,
    ResultLogger,
    YOHASRunner,
    _get_benchmark_mode,
)

logger = logging.getLogger("yohas.benchmarks.labbench")

# Categories where YOHAS has advantages via its tool integrations
PRIORITY_CATEGORIES = ["DbQA", "SeqQA", "LitQA2"]
ALL_CATEGORIES = [
    "LitQA2", "DbQA", "SuppQA", "FigQA",
    "TableQA", "ProtocolQA", "SeqQA", "CloningScenarios",
]

# Map LAB-Bench categories to YOHAS agent compositions
CATEGORY_AGENT_HINTS: dict[str, str] = {
    "DbQA": "Use database tools (UniProt, KEGG, Reactome, ChEMBL) to look up the answer.",
    "SeqQA": "Use protein sequence analysis tools (ESM-2, ESMFold) to analyze the sequence.",
    "LitQA2": "Search PubMed and Semantic Scholar for relevant literature.",
    "SuppQA": "Analyze supplementary materials and supporting data.",
    "FigQA": "Interpret the figure description to answer the question.",
    "TableQA": "Analyze the table data to extract the answer.",
    "ProtocolQA": "Apply knowledge of laboratory protocols and methods.",
    "CloningScenarios": "Apply molecular cloning knowledge to solve the scenario.",
}


def yohas_agent(input: Any) -> str:
    """LAB-Bench agent function interface.

    Args:
        input: labbench.AgentInput with .question, .options, .context, .category

    Returns:
        Single letter A-E.
    """
    adapter = YOHASLABBenchAdapter()
    return adapter.answer_question(input)


class YOHASLABBenchAdapter:
    """Adapter for LAB-Bench multiple-choice evaluation."""

    def __init__(
        self,
        *,
        mode: str | None = None,
        max_iterations: int = 3,
    ) -> None:
        self.mode = mode or _get_benchmark_mode()
        self.runner = YOHASRunner(
            max_iterations=max_iterations if self.mode == "agentic" else 1,
            enable_falsification=False,  # MCQ doesn't need falsification
            enable_hitl=False,
        )

    def _build_prompt(self, input: Any) -> str:
        """Build a prompt from LAB-Bench AgentInput."""
        question = getattr(input, "question", str(input))
        options = getattr(input, "options", None)
        context = getattr(input, "context", "")
        category = getattr(input, "category", "")

        parts = []

        if context:
            parts.append(f"Context:\n{context}")

        parts.append(f"Question: {question}")

        if options:
            parts.append("Options:")
            if isinstance(options, dict):
                for letter, text in sorted(options.items()):
                    parts.append(f"  {letter}) {text}")
            elif isinstance(options, (list, tuple)):
                for i, opt in enumerate(options):
                    letter = chr(65 + i)  # A, B, C, ...
                    parts.append(f"  {letter}) {opt}")

        # Add category-specific hint
        hint = CATEGORY_AGENT_HINTS.get(category, "")
        if hint:
            parts.append(f"\nHint: {hint}")

        parts.append(
            "\nProvide your answer as a single letter (A, B, C, D, or E). "
            "State your reasoning briefly, then give your final answer on the last line "
            "in the format: Answer: X"
        )

        return "\n\n".join(parts)

    def answer_question(self, input: Any) -> str:
        """Answer a single LAB-Bench question.

        For zero-shot mode: direct LLM call with KG context injection.
        For agentic mode on priority categories: lightweight YOHAS session.
        """
        category = getattr(input, "category", "")
        prompt = self._build_prompt(input)

        if self.mode == "zero-shot" or category not in PRIORITY_CATEGORIES:
            # Direct LLM call — faster for non-priority categories
            response = self.runner.query_llm_with_kg(
                prompt,
                system_prompt=(
                    "You are a biomedical research scientist taking a multiple-choice exam. "
                    "Reason carefully through each option before selecting your answer. "
                    "You must select exactly one answer letter."
                ),
            )
            return AnswerExtractor.extract_multiple_choice(response)

        # Agentic mode for priority categories — run lightweight session
        question = getattr(input, "question", str(input))
        session, kg = self.runner.run_session(
            f"[{category}] {question}"
        )

        # Use KG-enriched context for final answer
        kg_summary = kg.to_markdown_summary()
        enriched_prompt = (
            f"Based on the following research findings:\n\n{kg_summary}\n\n{prompt}"
        )
        response = self.runner.query_llm_with_kg(
            enriched_prompt,
            system_prompt=(
                "You are a biomedical research scientist. You have just completed "
                "a research session and built a knowledge graph. Use these findings "
                "to answer the multiple-choice question. Select exactly one letter."
            ),
            kg=kg,
        )
        return AnswerExtractor.extract_multiple_choice(response)


def run_labbench(
    *,
    categories: list[str] | None = None,
    limit: int | None = None,
    mode: str = "agentic",
    max_iterations: int = 3,
    checkpoint_id: str | None = None,
) -> Path:
    """Run LAB-Bench evaluation.

    Args:
        categories: Which categories to evaluate, or None for all.
        limit: Max questions per category.
        mode: "zero-shot" or "agentic".
        max_iterations: MCTS iterations per question (agentic priority categories).
        checkpoint_id: Resume from checkpoint.

    Returns:
        Path to results JSON.
    """
    try:
        import labbench  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "LAB-Bench not installed. Run: scripts/benchmarks/setup_benchmarks.sh"
        )
        raise SystemExit(1)

    selected = categories or ALL_CATEGORIES
    result_logger = ResultLogger("labbench")
    if checkpoint_id:
        result_logger._run_id = checkpoint_id
        result_logger._checkpoint_path = (
            result_logger.output_dir / f"{checkpoint_id}_checkpoint.jsonl"
        )
    completed = result_logger.load_checkpoint()

    adapter = YOHASLABBenchAdapter(mode=mode, max_iterations=max_iterations)

    try:
        evaluator = labbench.Evaluator()
    except Exception:
        evaluator = None
        logger.warning("labbench.Evaluator not available; scoring will be skipped")

    total_evaluated = 0
    for category in selected:
        if category not in ALL_CATEGORIES:
            logger.warning("Unknown category %s, skipping", category)
            continue

        try:
            dataset = labbench.load_dataset(category)
        except Exception as exc:
            logger.error("Failed to load category %s: %s", category, exc)
            continue

        questions = list(dataset)
        if limit:
            questions = questions[:limit]

        logger.info("Category %s: %d questions", category, len(questions))

        for idx, question_input in enumerate(questions):
            task_id = f"{category}_{idx}"
            if task_id in completed:
                logger.debug("Skipping completed %s", task_id)
                continue

            logger.info("[%s %d/%d] %s", category, idx + 1, len(questions), task_id)
            start = time.monotonic()

            try:
                predicted = adapter.answer_question(question_input)

                # Score via evaluator if available
                correct = False
                expected = ""
                if evaluator:
                    try:
                        eval_result = evaluator.evaluate(
                            category, idx, predicted
                        )
                        correct = bool(eval_result.get("correct", False)) if isinstance(eval_result, dict) else bool(eval_result)
                        expected = str(eval_result.get("expected", "")) if isinstance(eval_result, dict) else ""
                    except Exception as eval_exc:
                        logger.warning("Evaluator failed for %s: %s", task_id, eval_exc)
                else:
                    # Try to extract expected from input
                    expected = getattr(question_input, "answer", "")
                    if expected:
                        correct = predicted.upper() == expected.upper()

                result_logger.log_instance({
                    "instance_id": task_id,
                    "task_id": task_id,
                    "category": category,
                    "predicted": predicted,
                    "expected": expected,
                    "correct": correct,
                    "tokens_used": adapter.runner.token_summary.get("total_tokens", 0),
                    "duration_ms": int((time.monotonic() - start) * 1000),
                    "error": None,
                })
                total_evaluated += 1

            except Exception as exc:
                logger.error("Question %s failed: %s", task_id, exc)
                result_logger.log_instance({
                    "instance_id": task_id,
                    "task_id": task_id,
                    "category": category,
                    "predicted": "",
                    "expected": "",
                    "correct": False,
                    "tokens_used": 0,
                    "duration_ms": int((time.monotonic() - start) * 1000),
                    "error": traceback.format_exc(),
                })

    logger.info("LAB-Bench complete: %d questions evaluated", total_evaluated)
    return result_logger.save_final(
        extra_metadata={
            "mode": mode,
            "categories": selected,
            "max_iterations": max_iterations,
            "limit": limit,
        }
    )
