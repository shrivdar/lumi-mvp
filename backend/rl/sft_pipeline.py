"""SFT data pipeline — filter successful trajectories and format for fine-tuning.

Supports:
- Filtering by reward threshold (default: reward >= 0.5)
- Quality filtering: skip crashed agents, empty answers, low-confidence results
- Instruction-response pair formatting for SFT training
- Rejection sampling: run N per instance, keep only successful ones
- Export as HuggingFace Dataset (optional, if `datasets` installed)
- Export as JSONL (both conversation and prompt/completion formats)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from rl.trajectory_format import Trajectory

logger = structlog.get_logger(__name__)


class SFTPipeline:
    """Filter and format trajectories for supervised fine-tuning."""

    def __init__(
        self,
        *,
        reward_threshold: float = 0.5,
        max_turns: int | None = None,
        include_thinking: bool = False,
        min_turns: int = 1,
        max_tokens: int | None = None,
    ) -> None:
        self.reward_threshold = reward_threshold
        self.max_turns = max_turns
        self.include_thinking = include_thinking
        self.min_turns = min_turns
        self.max_tokens = max_tokens

    def load_trajectories(self, path: str | Path) -> list[Trajectory]:
        """Load trajectories from a JSONL file."""
        trajectories: list[Trajectory] = []
        path = Path(path)
        if not path.exists():
            logger.warning("sft_load_missing", path=str(path))
            return trajectories

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                trajectories.append(Trajectory.model_validate_json(line))

        logger.info("sft_load", path=str(path), count=len(trajectories))
        return trajectories

    def load_directory(self, directory: str | Path) -> list[Trajectory]:
        """Load all .jsonl trajectory files from a directory."""
        directory = Path(directory)
        all_trajectories: list[Trajectory] = []
        for jsonl_file in sorted(directory.glob("*.jsonl")):
            all_trajectories.extend(self.load_trajectories(jsonl_file))
        logger.info("sft_load_dir", dir=str(directory), total=len(all_trajectories))
        return all_trajectories

    def filter(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """Keep only trajectories meeting the reward threshold."""
        filtered = [t for t in trajectories if t.reward >= self.reward_threshold]
        logger.info(
            "sft_filter",
            input=len(trajectories),
            output=len(filtered),
            threshold=self.reward_threshold,
        )
        return filtered

    def quality_filter(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """Apply quality filters beyond reward threshold.

        Removes trajectories where:
        - Agent crashed (no turns at all)
        - No final answer produced (empty or whitespace-only)
        - Fewer turns than min_turns
        - Exceeds max_tokens budget (if set)
        - Success flag is False (agent explicitly failed)
        """
        accepted: list[Trajectory] = []
        reasons: dict[str, int] = {
            "no_turns": 0,
            "no_answer": 0,
            "too_few_turns": 0,
            "too_many_tokens": 0,
            "not_successful": 0,
        }

        for t in trajectories:
            if not t.turns:
                reasons["no_turns"] += 1
                continue
            if not t.final_answer or not t.final_answer.strip():
                reasons["no_answer"] += 1
                continue
            if len(t.turns) < self.min_turns:
                reasons["too_few_turns"] += 1
                continue
            if self.max_tokens and t.total_tokens > self.max_tokens:
                reasons["too_many_tokens"] += 1
                continue
            if not t.success:
                reasons["not_successful"] += 1
                continue
            accepted.append(t)

        logger.info(
            "sft_quality_filter",
            input=len(trajectories),
            output=len(accepted),
            rejected_reasons=reasons,
        )
        return accepted

    def filter_and_prepare(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """Combined filter pipeline: reward threshold → quality filter.

        Convenience method that chains filter() and quality_filter().
        """
        step1 = self.filter(trajectories)
        return self.quality_filter(step1)

    def format_instruction_response(
        self,
        trajectory: Trajectory,
    ) -> dict[str, str]:
        """Format a trajectory as an instruction-response pair for SFT.

        Instruction = task + observation history (tool results, intermediate steps).
        Response = final agent action/answer.

        Returns:
            {"prompt": "...", "completion": "...", "metadata": {...}}
        """
        # Build instruction from task + observation history
        instruction_parts = [trajectory.instruction]

        # Add context if available
        if trajectory.context:
            ctx_str = json.dumps(trajectory.context, default=str)
            if len(ctx_str) < 2000:
                instruction_parts.append(f"\nContext: {ctx_str}")

        # Add observation history from prior turns (exclude the final answer)
        observation_turns = trajectory.turns[:-1] if len(trajectory.turns) > 1 else []
        for turn in observation_turns:
            if self.max_turns is not None and turn.turn_number > self.max_turns:
                break
            if turn.turn_type == "think" and not self.include_thinking:
                continue

            if turn.tool_calls:
                for tc in turn.tool_calls:
                    instruction_parts.append(
                        f"\n[Tool: {tc.tool_name}] {tc.result[:500]}" if tc.result else ""
                    )
            if turn.code_executions:
                for ce in turn.code_executions:
                    instruction_parts.append(
                        f"\n[Code Output] {ce.output[:500]}" if ce.output else ""
                    )
            if turn.content and turn.role != "assistant":
                instruction_parts.append(f"\n{turn.content[:500]}")

        prompt = "\n".join(p for p in instruction_parts if p)

        # Response is the final answer or last assistant turn
        completion = trajectory.final_answer
        if not completion and trajectory.turns:
            last_turn = trajectory.turns[-1]
            completion = last_turn.content or ""

        return {
            "prompt": prompt,
            "completion": completion,
            "metadata": {
                "trajectory_id": trajectory.trajectory_id,
                "task_id": trajectory.task_id,
                "agent_type": trajectory.agent_type,
                "reward": trajectory.reward,
                "total_tokens": trajectory.total_tokens,
                "num_turns": len(trajectory.turns),
            },
        }

    def format_all_instruction_response(
        self,
        trajectories: list[Trajectory],
    ) -> list[dict[str, str]]:
        """Format multiple trajectories as instruction-response pairs."""
        return [self.format_instruction_response(t) for t in trajectories]

    def export_sft_dataset(
        self,
        trajectories: list[Trajectory],
        output_path: str | Path,
    ) -> Path:
        """Export as prompt/completion JSONL ready for training scripts.

        This is the primary export format consumed by training/sft.py.
        Each line: {"prompt": "...", "completion": "...", "metadata": {...}}
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for traj in trajectories:
                record = self.format_instruction_response(traj)
                f.write(json.dumps(record) + "\n")

        logger.info("sft_export_dataset", path=str(output_path), count=len(trajectories))
        return output_path

    def rejection_sample(
        self,
        trajectories: list[Trajectory],
        n_per_task: int = 5,
    ) -> list[Trajectory]:
        """Rejection sampling: group by task_id, keep successful ones.

        Designed for benchmark runs where the same task is attempted N times.
        Returns at most one successful trajectory per unique task_id.
        """
        by_task: dict[str, list[Trajectory]] = {}
        for t in trajectories:
            by_task.setdefault(t.task_id, []).append(t)

        selected: list[Trajectory] = []
        for task_id, group in by_task.items():
            successes = [t for t in group if t.reward >= self.reward_threshold]
            if successes:
                # Pick the one with fewest turns (most efficient)
                best = min(successes, key=lambda t: len(t.turns))
                selected.append(best)

        logger.info(
            "sft_rejection_sample",
            tasks=len(by_task),
            selected=len(selected),
            n_per_task=n_per_task,
        )
        return selected

    def format_conversation(self, trajectory: Trajectory) -> list[dict[str, str]]:
        """Format a trajectory as a conversation (list of role/content dicts).

        Suitable for SFT training with chat-format models.
        """
        messages: list[dict[str, str]] = []

        # System message with task instruction
        messages.append({
            "role": "system",
            "content": (
                f"You are a {trajectory.agent_type} research agent. "
                f"Investigate the following task and provide a structured answer.\n\n"
                f"Task: {trajectory.instruction}"
            ),
        })

        for turn in trajectory.turns:
            if self.max_turns is not None and turn.turn_number > self.max_turns:
                break

            if turn.turn_type == "think" and not self.include_thinking:
                continue

            # Assistant message (the LLM's response)
            if turn.content:
                messages.append({
                    "role": "assistant",
                    "content": turn.content,
                })

            # Tool results as tool/user messages
            for tc in turn.tool_calls:
                if tc.result:
                    messages.append({
                        "role": "tool",
                        "content": f"[{tc.tool_name}] {tc.result}",
                    })

            for ce in turn.code_executions:
                if ce.output:
                    messages.append({
                        "role": "tool",
                        "content": f"[code_execution] {ce.output}",
                    })

        return messages

    def format_all(
        self,
        trajectories: list[Trajectory],
    ) -> list[list[dict[str, str]]]:
        """Format multiple trajectories as conversations."""
        return [self.format_conversation(t) for t in trajectories]

    def export_jsonl(
        self,
        trajectories: list[Trajectory],
        output_path: str | Path,
    ) -> Path:
        """Export formatted conversations to a JSONL file for SFT training."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for traj in trajectories:
                conversation = self.format_conversation(traj)
                record = {
                    "trajectory_id": traj.trajectory_id,
                    "task_id": traj.task_id,
                    "agent_type": traj.agent_type,
                    "messages": conversation,
                    "reward": traj.reward,
                    "total_tokens": traj.total_tokens,
                }
                f.write(json.dumps(record) + "\n")

        logger.info("sft_export_jsonl", path=str(output_path), count=len(trajectories))
        return output_path

    def export_huggingface(
        self,
        trajectories: list[Trajectory],
        output_dir: str | Path,
        dataset_name: str = "yohas_sft",
    ) -> Any:
        """Export as a HuggingFace Dataset (requires `datasets` library).

        Returns the Dataset object, or None if `datasets` is not installed.
        """
        try:
            from datasets import Dataset
        except ImportError:
            logger.warning(
                "sft_export_hf_skip",
                msg="datasets library not installed, skipping HuggingFace export",
            )
            return None

        records = []
        for traj in trajectories:
            conversation = self.format_conversation(traj)
            records.append({
                "trajectory_id": traj.trajectory_id,
                "task_id": traj.task_id,
                "agent_type": traj.agent_type,
                "messages": json.dumps(conversation),
                "reward": traj.reward,
                "total_tokens": traj.total_tokens,
                "wall_time_ms": traj.wall_time_ms,
                "num_turns": len(traj.turns),
                "num_kg_mutations": len(traj.kg_mutations),
            })

        dataset = Dataset.from_list(records)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_dir / dataset_name))

        logger.info(
            "sft_export_hf",
            path=str(output_dir / dataset_name),
            count=len(records),
        )
        return dataset
