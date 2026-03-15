"""Trajectory store — saves/loads agent trajectories for RL training."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.models import Trajectory

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "trajectories"


class TrajectoryStore:
    """Persist trajectories to JSONL files for downstream RL training."""

    def __init__(self, output_dir: Path | str | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, trajectories: list[Trajectory], tag: str = "") -> Path:
        """Save trajectories to a JSONL file. Returns the output path."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        suffix = f"_{tag}" if tag else ""
        filename = f"trajectories_{timestamp}{suffix}.jsonl"
        path = self.output_dir / filename

        with open(path, "w") as f:
            for t in trajectories:
                f.write(t.model_dump_json() + "\n")

        logger.info("Saved %d trajectories to %s", len(trajectories), path)
        return path

    def load(self, path: Path | str) -> list[Trajectory]:
        """Load trajectories from a JSONL file."""
        path = Path(path)
        trajectories: list[Trajectory] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    trajectories.append(Trajectory.model_validate_json(line))
        return trajectories

    def load_all(self, tag: str = "") -> list[Trajectory]:
        """Load all trajectory files from the output directory."""
        all_trajectories: list[Trajectory] = []
        pattern = f"trajectories_*{tag}*.jsonl" if tag else "trajectories_*.jsonl"
        for path in sorted(self.output_dir.glob(pattern)):
            all_trajectories.extend(self.load(path))
        return all_trajectories

    def summary(self) -> dict[str, Any]:
        """Summary of stored trajectories."""
        files = list(self.output_dir.glob("trajectories_*.jsonl"))
        total_trajectories = 0
        total_steps = 0
        for f in files:
            with open(f) as fh:
                for line in fh:
                    if line.strip():
                        data = json.loads(line)
                        total_trajectories += 1
                        total_steps += len(data.get("steps", []))
        return {
            "files": len(files),
            "total_trajectories": total_trajectories,
            "total_steps": total_steps,
            "output_dir": str(self.output_dir),
        }
