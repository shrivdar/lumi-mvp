"""Tests for the trajectory store."""

from __future__ import annotations

from benchmarks.models import (
    BenchmarkSuite,
    RunMode,
    Trajectory,
    TrajectoryStep,
)
from benchmarks.trajectory_store import TrajectoryStore


def _make_trajectory(instance_id: str = "test_001") -> Trajectory:
    return Trajectory(
        instance_id=instance_id,
        suite=BenchmarkSuite.BIOMNI_EVAL1,
        mode=RunMode.YOHAS_FULL,
        steps=[
            TrajectoryStep(step=0, action_type="think", action="Analyzing the question..."),
            TrajectoryStep(step=1, action_type="tool_call", action="pubmed_search('BRCA1')"),
            TrajectoryStep(step=2, action_type="answer", action="Tumor suppressor"),
        ],
        total_reward=1.0,
        correct=True,
    )


class TestTrajectoryStore:
    def test_save_and_load(self, trajectory_store):
        trajectories = [_make_trajectory(f"inst_{i}") for i in range(5)]
        path = trajectory_store.save(trajectories, tag="test")
        assert path.exists()

        loaded = trajectory_store.load(path)
        assert len(loaded) == 5
        assert loaded[0].instance_id == "inst_0"
        assert len(loaded[0].steps) == 3

    def test_save_creates_directory(self, tmp_path):
        new_dir = tmp_path / "new" / "dir"
        store = TrajectoryStore(output_dir=new_dir)
        trajectories = [_make_trajectory()]
        path = store.save(trajectories)
        assert path.exists()
        assert new_dir.exists()

    def test_load_all(self, trajectory_store):
        # Save two batches
        trajectory_store.save([_make_trajectory("a1")], tag="batch1")
        trajectory_store.save([_make_trajectory("a2"), _make_trajectory("a3")], tag="batch2")

        all_t = trajectory_store.load_all()
        assert len(all_t) == 3

    def test_load_all_with_tag_filter(self, trajectory_store):
        trajectory_store.save([_make_trajectory("a1")], tag="biomni")
        trajectory_store.save([_make_trajectory("a2")], tag="bixbench")

        biomni = trajectory_store.load_all(tag="biomni")
        assert len(biomni) == 1
        assert biomni[0].instance_id == "a1"

    def test_summary(self, trajectory_store):
        trajectory_store.save([_make_trajectory("a1"), _make_trajectory("a2")])
        summary = trajectory_store.summary()
        assert summary["files"] == 1
        assert summary["total_trajectories"] == 2
        assert summary["total_steps"] == 6  # 3 steps each

    def test_empty_summary(self, trajectory_store):
        summary = trajectory_store.summary()
        assert summary["files"] == 0
        assert summary["total_trajectories"] == 0

    def test_trajectory_roundtrip_preserves_data(self, trajectory_store):
        t = _make_trajectory()
        t.steps[1].observation = "Found 15 papers on BRCA1"
        t.steps[1].reward = 0.5
        t.steps[1].metadata = {"tool": "pubmed", "results": 15}

        path = trajectory_store.save([t])
        loaded = trajectory_store.load(path)
        step = loaded[0].steps[1]
        assert step.observation == "Found 15 papers on BRCA1"
        assert step.reward == 0.5
        assert step.metadata["tool"] == "pubmed"
