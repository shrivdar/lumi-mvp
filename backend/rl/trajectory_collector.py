"""Trajectory collector — hooks into the agent loop to record multi-turn sessions.

Usage:
    collector = TrajectoryCollector(output_dir="data/trajectories")
    trajectory = collector.collect(task, result)  # after agent.execute()
    collector.flush()                              # write buffered trajectories to JSONL
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import structlog

from core.models import AgentResult, AgentTask, AgentTurn, TurnType
from rl.trajectory_format import (
    CodeExecRecord,
    KGMutationRecord,
    ToolCallRecord,
    Trajectory,
    Turn,
)

logger = structlog.get_logger(__name__)


class TrajectoryCollector:
    """Collects agent trajectories and writes them to JSONL files."""

    def __init__(
        self,
        output_dir: str | Path = "data/trajectories",
        benchmark_run_id: str | None = None,
        reward_fn: Any | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_run_id = benchmark_run_id
        self._reward_fn = reward_fn
        self._buffer: list[Trajectory] = []

    def collect(
        self,
        task: AgentTask,
        result: AgentResult,
        *,
        reward: float | None = None,
    ) -> Trajectory:
        """Convert an AgentTask + AgentResult into a Trajectory and buffer it.

        Args:
            task: The task that was executed.
            result: The agent's result.
            reward: Explicit reward override. If None, derived from result.success
                    or self._reward_fn.
        """
        # Build turns from AgentTurn list
        turns = [self._convert_turn(t) for t in result.turns]

        # Build KG mutation records
        kg_mutations = self._extract_kg_mutations(result)

        # Compute reward
        if reward is not None:
            final_reward = reward
        elif self._reward_fn is not None:
            final_reward = self._reward_fn(task, result)
        else:
            final_reward = 1.0 if result.success else 0.0

        trajectory = Trajectory(
            task_id=task.task_id,
            research_id=task.research_id,
            agent_type=str(result.agent_type),
            agent_id=result.agent_id,
            hypothesis_branch=task.hypothesis_branch,
            instruction=task.instruction,
            context=task.context,
            turns=turns,
            final_answer=result.summary,
            reward=final_reward,
            success=result.success,
            kg_mutations=kg_mutations,
            token_usage=result.token_usage,
            wall_time_ms=result.duration_ms,
            llm_calls=result.llm_calls,
            total_tokens=result.llm_tokens_used,
            benchmark_run_id=self.benchmark_run_id,
        )

        self._buffer.append(trajectory)

        # Also collect sub-agent trajectories recursively
        for sub_result in result.sub_agent_results:
            sub_task = AgentTask(
                task_id=sub_result.task_id,
                research_id=task.research_id,
                agent_type=sub_result.agent_type,
                agent_id=sub_result.agent_id,
                hypothesis_branch=sub_result.hypothesis_id or task.hypothesis_branch,
                instruction=f"Sub-agent task for {sub_result.agent_type}",
                context=task.context,
            )
            self.collect(sub_task, sub_result, reward=reward)

        return trajectory

    def flush(self, filename: str | None = None) -> Path:
        """Write all buffered trajectories to a JSONL file.

        Returns the path to the written file.
        """
        if not self._buffer:
            logger.info("trajectory_flush_empty", msg="No trajectories to flush")
            return self.output_dir / "empty"

        if filename is None:
            ts = int(time.time())
            run_tag = self.benchmark_run_id or "run"
            filename = f"{run_tag}_{ts}.jsonl"

        out_path = self.output_dir / filename
        with open(out_path, "a") as f:
            for traj in self._buffer:
                f.write(traj.model_dump_json() + "\n")

        logger.info(
            "trajectory_flush",
            count=len(self._buffer),
            path=str(out_path),
        )
        self._buffer.clear()
        return out_path

    @property
    def buffered_count(self) -> int:
        return len(self._buffer)

    @property
    def trajectories(self) -> list[Trajectory]:
        """Read-only access to buffered trajectories."""
        return list(self._buffer)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_turn(agent_turn: AgentTurn) -> Turn:
        """Convert an AgentTurn (core model) into an RL Turn."""
        tool_calls: list[ToolCallRecord] = []
        code_execs: list[CodeExecRecord] = []

        if agent_turn.turn_type == TurnType.TOOL_CALL:
            # Parse tool name and args from parsed_action
            tool_name, args = _parse_tool_action(agent_turn.parsed_action)
            tool_calls.append(ToolCallRecord(
                tool_name=tool_name,
                arguments=args,
                result=agent_turn.execution_result,
                duration_ms=agent_turn.duration_ms,
                error=agent_turn.error,
            ))

        elif agent_turn.turn_type == TurnType.CODE_EXECUTION:
            code_execs.append(CodeExecRecord(
                code=agent_turn.parsed_action,
                output=agent_turn.execution_result,
                duration_ms=agent_turn.duration_ms,
                error=agent_turn.error,
            ))

        return Turn(
            turn_number=agent_turn.turn_number,
            role="assistant",
            content=agent_turn.raw_response,
            turn_type=str(agent_turn.turn_type),
            tool_calls=tool_calls,
            code_executions=code_execs,
            tokens_used=agent_turn.tokens_used,
            duration_ms=agent_turn.duration_ms,
            timestamp=agent_turn.timestamp,
        )

    @staticmethod
    def _extract_kg_mutations(result: AgentResult) -> list[KGMutationRecord]:
        """Extract KG mutations from an AgentResult."""
        mutations: list[KGMutationRecord] = []

        for node in result.nodes_added:
            mutations.append(KGMutationRecord(
                operation="add_node",
                entity_id=node.id,
                entity_type=str(node.type),
                details={"name": node.name, "confidence": node.confidence},
            ))

        for edge in result.edges_added:
            mutations.append(KGMutationRecord(
                operation="add_edge",
                entity_id=edge.id,
                entity_type=str(edge.relation),
                details={
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "confidence": edge.confidence.overall,
                },
            ))

        for node_id in result.nodes_updated:
            mutations.append(KGMutationRecord(
                operation="update_node",
                entity_id=node_id,
            ))

        for edge_id in result.edges_updated:
            mutations.append(KGMutationRecord(
                operation="update_edge",
                entity_id=edge_id,
            ))

        return mutations


def _parse_tool_action(parsed_action: str) -> tuple[str, dict[str, Any]]:
    """Parse 'tool_name:{"arg": "value"}' format into (name, args).

    Falls back gracefully if the format doesn't match.
    """
    if ":" in parsed_action:
        name, _, rest = parsed_action.partition(":")
        name = name.strip()
        try:
            args = json.loads(rest.strip())
            if isinstance(args, dict):
                return name, args
        except (json.JSONDecodeError, ValueError):
            pass
        return name, {"raw": rest.strip()}
    return parsed_action.strip(), {}
