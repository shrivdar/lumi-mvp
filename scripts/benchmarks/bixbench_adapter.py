"""BixBench adapter for YOHAS 3.0.

Implements the custom_rollout interface from BixBench to evaluate YOHAS
against 205 questions from 60 real-world Jupyter notebooks.

Docker image: futurehouse/bixbench:aviary-notebook-env
Repo: https://github.com/Future-House/BixBench
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from common import AnswerExtractor, ResultLogger, YOHASRunner, _get_benchmark_mode

logger = logging.getLogger("yohas.benchmarks.bixbench")


class YOHASBixBenchAgent:
    """YOHAS agent for BixBench evaluation.

    Implements the custom_rollout interface expected by BixBench's
    evaluation harness. Each capsule contains a question plus data files;
    we translate it into a YOHAS research session and extract the answer.
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

    def _capsule_to_query(self, capsule: dict[str, Any]) -> str:
        """Translate a BixBench capsule into a YOHAS research query."""
        question = capsule.get("question", "")
        context = capsule.get("context", "")
        data_description = capsule.get("data_description", "")

        parts = [f"Research question: {question}"]
        if context:
            parts.append(f"Context: {context}")
        if data_description:
            parts.append(f"Available data: {data_description}")

        # Include any notebook metadata
        notebook = capsule.get("notebook_name", "")
        if notebook:
            parts.append(f"Source notebook: {notebook}")

        return "\n\n".join(parts)

    def _format_trajectory(
        self,
        capsule: dict[str, Any],
        answer: str,
        session: Any,
        kg: Any,
        duration_ms: int,
    ) -> dict[str, Any]:
        """Format output as BixBench trajectory JSON for grading."""
        steps: list[dict[str, Any]] = []

        # Record the research session as trajectory steps
        if session and session.result:
            # Step 1: hypothesis generation
            if session.result.hypothesis_ranking:
                steps.append({
                    "action": "hypothesis_generation",
                    "content": [
                        h.hypothesis for h in session.result.hypothesis_ranking
                    ],
                    "timestamp": time.time(),
                })

            # Step 2: KG construction
            kg_json = kg.to_json() if kg else {}
            steps.append({
                "action": "knowledge_graph_construction",
                "content": {
                    "nodes": kg_json.get("metadata", {}).get("node_count", 0),
                    "edges": kg_json.get("metadata", {}).get("edge_count", 0),
                },
                "timestamp": time.time(),
            })

            # Step 3: key findings
            if session.result.key_findings:
                steps.append({
                    "action": "key_findings",
                    "content": [
                        {
                            "source": kg.get_node(e.source_id).name
                            if kg.get_node(e.source_id)
                            else e.source_id,
                            "target": kg.get_node(e.target_id).name
                            if kg.get_node(e.target_id)
                            else e.target_id,
                            "relation": str(e.relation),
                            "confidence": e.confidence.overall,
                        }
                        for e in session.result.key_findings[:10]
                    ],
                    "timestamp": time.time(),
                })

        # Step 4: final answer
        steps.append({
            "action": "final_answer",
            "content": answer,
            "timestamp": time.time(),
        })

        return {
            "question_id": capsule.get("question_id", capsule.get("id", "")),
            "question": capsule.get("question", ""),
            "answer": answer,
            "trajectory": steps,
            "metadata": {
                "agent": "yohas-3.0",
                "mode": self.mode,
                "duration_ms": duration_ms,
                "tokens": self.runner.token_summary,
            },
        }

    def custom_rollout(self, capsule: dict[str, Any]) -> dict[str, Any]:
        """BixBench custom_rollout interface.

        Args:
            capsule: BixBench question capsule with question, context, data files.

        Returns:
            Trajectory dict with answer and steps for BixBench grading.
        """
        start = time.monotonic()
        query = self._capsule_to_query(capsule)

        if self.mode == "zero-shot":
            # Direct LLM call without full research loop
            prompt = (
                f"{query}\n\n"
                "Provide a concise, specific answer based on your biomedical knowledge. "
                "If the question asks for a specific value, gene, protein, or entity, "
                "state it directly."
            )
            answer = self.runner.query_llm_with_kg(
                prompt,
                system_prompt=(
                    "You are a biomedical research scientist. Answer precisely "
                    "and concisely based on the question and any data provided."
                ),
            )
            duration = int((time.monotonic() - start) * 1000)
            return self._format_trajectory(capsule, answer, None, None, duration)

        # Agentic mode: full YOHAS research session
        session, kg = self.runner.run_session(query)
        answer = AnswerExtractor.extract_answer_text(session, kg)
        duration = int((time.monotonic() - start) * 1000)
        return self._format_trajectory(capsule, answer, session, kg, duration)


def run_bixbench(
    *,
    mode: str = "agentic",
    limit: int | None = None,
    max_iterations: int = 5,
    checkpoint_id: str | None = None,
) -> Path:
    """Run BixBench evaluation end-to-end.

    Args:
        mode: "zero-shot" or "agentic"
        limit: Max number of instances to evaluate
        max_iterations: MCTS iterations per instance (agentic mode)
        checkpoint_id: Resume from a previous run's checkpoint

    Returns:
        Path to results JSON file.
    """
    try:
        from bixbench.dataset import load_bixbench_dataset  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "BixBench not installed. Run: scripts/benchmarks/setup_benchmarks.sh"
        )
        raise SystemExit(1)

    result_logger = ResultLogger("bixbench")
    if checkpoint_id:
        result_logger._run_id = checkpoint_id
        result_logger._checkpoint_path = (
            result_logger.output_dir / f"{checkpoint_id}_checkpoint.jsonl"
        )
    completed = result_logger.load_checkpoint()

    agent = YOHASBixBenchAgent(mode=mode, max_iterations=max_iterations)

    dataset = load_bixbench_dataset()
    instances = list(dataset)
    if limit:
        instances = instances[:limit]

    logger.info(
        "BixBench: %d instances (%d already completed), mode=%s",
        len(instances),
        len(completed),
        mode,
    )

    for i, capsule in enumerate(instances):
        instance_id = str(capsule.get("question_id", capsule.get("id", i)))
        if instance_id in completed:
            logger.debug("Skipping completed instance %s", instance_id)
            continue

        logger.info("[%d/%d] Instance %s", i + 1, len(instances), instance_id)
        start = time.monotonic()

        try:
            trajectory = agent.custom_rollout(capsule)
            # NOTE: BixBench uses its own external grading harness.
            # We do NOT self-score here. The trajectory JSON is submitted
            # to BixBench's grader for evaluation. 'correct' is set to None
            # to indicate "not yet graded".
            result_logger.log_instance({
                "instance_id": instance_id,
                "question": capsule.get("question", ""),
                "answer": trajectory["answer"],
                "correct": None,  # graded externally by BixBench
                "score": None,    # graded externally by BixBench
                "trajectory": trajectory,
                "tokens_used": agent.runner.token_summary.get("total_tokens", 0),
                "duration_ms": int((time.monotonic() - start) * 1000),
                "error": None,
            })
        except Exception as exc:
            logger.error("Instance %s failed: %s", instance_id, exc)
            result_logger.log_instance({
                "instance_id": instance_id,
                "question": capsule.get("question", ""),
                "answer": "",
                "correct": False,
                "score": 0.0,
                "trajectory": None,
                "tokens_used": 0,
                "duration_ms": int((time.monotonic() - start) * 1000),
                "error": traceback.format_exc(),
            })

    return result_logger.save_final(
        extra_metadata={"mode": mode, "max_iterations": max_iterations, "limit": limit}
    )
