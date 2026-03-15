"""SFT data pipeline — filter successful trajectories and format for fine-tuning.

Supports:
- Filtering by reward threshold (default: reward == 1.0)
- Formatting as conversation turns for SFT
- Rejection sampling: run N per instance, keep only successful ones
- Export as HuggingFace Dataset (optional, if `datasets` installed)
- Export as JSONL
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
        reward_threshold: float = 1.0,
        max_turns: int | None = None,
        include_thinking: bool = False,
    ) -> None:
        self.reward_threshold = reward_threshold
        self.max_turns = max_turns
        self.include_thinking = include_thinking

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
