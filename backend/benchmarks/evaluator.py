"""Evaluator — runs benchmark instances through YOHAS (zero-shot or full-stack).

The evaluator handles:
- Zero-shot mode: direct LLM call with no tools or KG
- YOHAS full mode: full orchestrator with multi-turn agents, tools, data lake, know-how
- Trajectory collection for RL training
- Per-instance metric recording
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from benchmarks.models import (
    BenchmarkInstance,
    InstanceResult,
    InstanceStatus,
    RunMode,
    Trajectory,
    TrajectoryStep,
)

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Evaluates benchmark instances in zero-shot or YOHAS-full mode."""

    def __init__(
        self,
        *,
        mode: RunMode,
        llm: Any | None = None,
        orchestrator_factory: Any | None = None,
        max_concurrency: int = 5,
        timeout_seconds: int = 300,
        collect_trajectories: bool = True,
    ) -> None:
        self.mode = mode
        self.llm = llm
        self.orchestrator_factory = orchestrator_factory
        self.max_concurrency = max_concurrency
        self.timeout_seconds = timeout_seconds
        self.collect_trajectories = collect_trajectories
        self._trajectories: list[Trajectory] = []

    @property
    def trajectories(self) -> list[Trajectory]:
        return list(self._trajectories)

    async def evaluate_batch(
        self,
        instances: list[BenchmarkInstance],
        *,
        progress_callback: Any | None = None,
    ) -> list[InstanceResult]:
        """Evaluate a batch of instances with concurrency control."""
        sem = asyncio.Semaphore(self.max_concurrency)
        results: list[InstanceResult] = []
        total = len(instances)

        async def _eval_one(idx: int, inst: BenchmarkInstance) -> InstanceResult:
            async with sem:
                result = await self.evaluate_instance(inst)
                if progress_callback:
                    progress_callback(idx + 1, total, result)
                return result

        tasks = [_eval_one(i, inst) for i, inst in enumerate(instances)]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    async def evaluate_instance(self, instance: BenchmarkInstance) -> InstanceResult:
        """Evaluate a single benchmark instance."""
        start = time.monotonic()
        started_at_dt = None

        try:
            from datetime import UTC, datetime

            started_at_dt = datetime.now(UTC)

            if self.mode == RunMode.ZERO_SHOT:
                result = await self._evaluate_zero_shot(instance)
            else:
                result = await self._evaluate_yohas_full(instance)

            result.started_at = started_at_dt
            result.latency_ms = int((time.monotonic() - start) * 1000)
            result.correct = self._check_answer(result.predicted, instance.ground_truth)
            result.score = 1.0 if result.correct else 0.0
            result.status = InstanceStatus.COMPLETED

            from datetime import UTC, datetime

            result.completed_at = datetime.now(UTC)
            return result

        except TimeoutError:
            return InstanceResult(
                instance_id=instance.instance_id,
                suite=instance.suite,
                mode=self.mode,
                ground_truth=instance.ground_truth,
                status=InstanceStatus.TIMEOUT,
                latency_ms=int((time.monotonic() - start) * 1000),
                error="Evaluation timed out",
            )
        except Exception as exc:
            logger.error("Instance %s failed: %s", instance.instance_id, exc)
            return InstanceResult(
                instance_id=instance.instance_id,
                suite=instance.suite,
                mode=self.mode,
                ground_truth=instance.ground_truth,
                status=InstanceStatus.FAILED,
                latency_ms=int((time.monotonic() - start) * 1000),
                error=str(exc),
            )

    async def _evaluate_zero_shot(self, instance: BenchmarkInstance) -> InstanceResult:
        """Zero-shot: direct LLM call, no tools or KG."""
        prompt = self._build_zero_shot_prompt(instance)

        if self.llm is None:
            # Dry-run mode — simulate response
            return self._simulate_zero_shot(instance)

        llm_resp = await asyncio.wait_for(
            self.llm.query(
                prompt,
                system_prompt="You are a biomedical expert. Answer the question precisely and concisely.",
                max_tokens=1024,
            ),
            timeout=self.timeout_seconds,
        )
        response = llm_resp.text

        predicted = self._extract_answer(response, instance.choices)

        trajectory = Trajectory(
            instance_id=instance.instance_id,
            suite=instance.suite,
            mode=self.mode,
            steps=[
                TrajectoryStep(
                    step=0,
                    action_type="answer",
                    action=prompt,
                    observation=response,
                )
            ],
        )
        if self.collect_trajectories:
            self._trajectories.append(trajectory)

        return InstanceResult(
            instance_id=instance.instance_id,
            suite=instance.suite,
            mode=self.mode,
            predicted=predicted,
            ground_truth=instance.ground_truth,
            reasoning_trace=response,
            tokens_used=llm_resp.call_tokens,
            turns=1,
        )

    async def _evaluate_yohas_full(self, instance: BenchmarkInstance) -> InstanceResult:
        """Full YOHAS: orchestrator with multi-turn agents, tools, data lake, know-how."""
        if self.orchestrator_factory is None:
            return self._simulate_yohas_full(instance)

        from core.models import ResearchConfig

        config = ResearchConfig(
            max_mcts_iterations=3,
            max_agents=4,
            max_agents_per_swarm=3,
            max_llm_calls_per_agent=10,
            enable_hitl=False,
            enable_falsification=True,
            agent_token_budget=20_000,
            session_token_budget=100_000,
        )

        orchestrator = self.orchestrator_factory()

        session = await asyncio.wait_for(
            orchestrator.run(instance.question, config=config),
            timeout=self.timeout_seconds,
        )

        # Extract answer from research result
        predicted = ""
        tools_used: list[str] = []
        turns = 0
        tokens = 0

        if session.result:
            predicted = self._extract_answer_from_result(session.result, instance)
            tokens = session.result.total_tokens
            turns = session.current_iteration

        # Collect trajectory from agent results
        if self.collect_trajectories and hasattr(orchestrator, "_all_results"):
            trajectory = self._build_trajectory_from_results(
                instance, orchestrator._all_results
            )
            self._trajectories.append(trajectory)
            for r in orchestrator._all_results:
                tools_used.extend(t.action_type for t in (r.turns or []) if t.turn_type.value == "tool_call")

        return InstanceResult(
            instance_id=instance.instance_id,
            suite=instance.suite,
            mode=self.mode,
            predicted=predicted,
            ground_truth=instance.ground_truth,
            tokens_used=tokens,
            turns=turns,
            tools_used=list(set(tools_used)),
            reasoning_trace=session.result.report_markdown if session.result else "",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_zero_shot_prompt(self, instance: BenchmarkInstance) -> str:
        parts = [instance.question]
        if instance.context:
            parts.insert(0, f"Context: {instance.context}\n")
        if instance.choices:
            choices_str = "\n".join(f"  {c}" for c in instance.choices)
            parts.append(f"\nOptions:\n{choices_str}")
        parts.append("\nProvide your answer. If multiple-choice, state only the letter/option.")
        return "\n".join(parts)

    def _extract_answer(self, response: str, choices: list[str]) -> str:
        """Extract the answer from an LLM response."""
        response = response.strip()
        if choices:
            # Try to match a choice letter (A, B, C, D)
            match = re.search(r"\b([A-D])\b", response)
            if match:
                idx = ord(match.group(1)) - ord("A")
                if idx < len(choices):
                    return choices[idx]
        # Fall back to first line
        return response.split("\n")[0].strip()

    def _extract_answer_from_result(self, result: Any, instance: BenchmarkInstance) -> str:
        """Extract answer from a ResearchResult."""
        if result.report_markdown:
            return self._extract_answer(result.report_markdown, instance.choices)
        if result.key_findings:
            # Use the highest-confidence finding
            best = max(result.key_findings, key=lambda e: e.confidence.overall)
            return str(best.relation)
        return ""

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if the predicted answer matches ground truth."""
        if not predicted or not ground_truth:
            return False
        # Normalize for comparison
        p = predicted.strip().lower()
        g = ground_truth.strip().lower()
        return p == g or p.startswith(g) or g.startswith(p)

    def _build_trajectory_from_results(
        self,
        instance: BenchmarkInstance,
        results: list[Any],
    ) -> Trajectory:
        """Build an RL trajectory from agent results."""
        steps: list[TrajectoryStep] = []
        step_idx = 0

        for result in results:
            for turn in result.turns or []:
                steps.append(
                    TrajectoryStep(
                        step=step_idx,
                        action_type=turn.turn_type.value,
                        action=turn.parsed_action or turn.input_prompt,
                        observation=turn.execution_result or turn.raw_response,
                        metadata={
                            "agent_id": result.agent_id,
                            "agent_type": result.agent_type.value,
                            "tokens": turn.tokens_used,
                            "duration_ms": turn.duration_ms,
                        },
                    )
                )
                step_idx += 1

        return Trajectory(
            instance_id=instance.instance_id,
            suite=instance.suite,
            mode=self.mode,
            steps=steps,
            correct=False,  # set later
        )

    # ------------------------------------------------------------------
    # Simulation (dry-run without LLM)
    # ------------------------------------------------------------------

    def _simulate_zero_shot(self, instance: BenchmarkInstance) -> InstanceResult:
        """Simulate a zero-shot evaluation without calling LLM."""
        import random

        correct = random.random() < 0.5  # 50% baseline
        predicted = instance.ground_truth if correct else "incorrect_answer"

        if self.collect_trajectories:
            self._trajectories.append(
                Trajectory(
                    instance_id=instance.instance_id,
                    suite=instance.suite,
                    mode=self.mode,
                    steps=[
                        TrajectoryStep(step=0, action_type="answer", action=predicted),
                    ],
                    correct=correct,
                )
            )

        return InstanceResult(
            instance_id=instance.instance_id,
            suite=instance.suite,
            mode=self.mode,
            predicted=predicted,
            ground_truth=instance.ground_truth,
            tokens_used=500,
            turns=1,
            reasoning_trace="[simulated zero-shot]",
        )

    def _simulate_yohas_full(self, instance: BenchmarkInstance) -> InstanceResult:
        """Simulate a YOHAS-full evaluation without calling LLM."""
        import random

        # YOHAS-full should outperform zero-shot
        correct = random.random() < 0.75  # 75% with full stack
        predicted = instance.ground_truth if correct else "incorrect_answer"

        tools = ["pubmed", "semantic_scholar", "uniprot", "kegg"]
        used = random.sample(tools, k=random.randint(1, len(tools)))

        if self.collect_trajectories:
            self._trajectories.append(
                Trajectory(
                    instance_id=instance.instance_id,
                    suite=instance.suite,
                    mode=self.mode,
                    steps=[
                        TrajectoryStep(step=0, action_type="think", action="Analyzing question..."),
                        TrajectoryStep(step=1, action_type="tool_call", action=f"search({used[0]})"),
                        TrajectoryStep(step=2, action_type="answer", action=predicted),
                    ],
                    correct=correct,
                )
            )

        return InstanceResult(
            instance_id=instance.instance_id,
            suite=instance.suite,
            mode=self.mode,
            predicted=predicted,
            ground_truth=instance.ground_truth,
            tokens_used=3000,
            turns=random.randint(2, 5),
            tools_used=used,
            reasoning_trace="[simulated yohas-full]",
        )
