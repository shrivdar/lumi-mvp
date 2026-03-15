"""Tests for SFT, RL, and eval training scripts (dry-run mode)."""

from __future__ import annotations

import json
from pathlib import Path

from rl.training.config import EvalConfig, RLAlgorithm, RLConfig, SFTConfig
from rl.training.eval import run_eval
from rl.training.rl import run_rl
from rl.training.sft import run_sft


class TestSFTDryRun:
    def test_dry_run_creates_metadata(self, tmp_path: Path):
        config = SFTConfig(
            dry_run=True,
            output_dir=tmp_path / "sft_out",
            dataset_path=tmp_path / "nonexistent_dataset",
        )
        result_path = run_sft(config)
        assert result_path.exists()
        meta = json.loads((result_path / "dry_run_meta.json").read_text())
        assert meta["status"] == "dry_run"
        assert meta["base_model"] == config.base_model

    def test_dry_run_with_dataset(self, tmp_path: Path):
        # Create a minimal dataset
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        data = [
            {"prompt": "What is BRCA1?", "completion": "BRCA1 is a tumor suppressor gene."},
            {"prompt": "What is TP53?", "completion": "TP53 encodes the p53 protein."},
        ]
        with open(dataset_dir / "train.jsonl", "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

        config = SFTConfig(
            dry_run=True,
            output_dir=tmp_path / "sft_out",
            dataset_path=dataset_dir,
        )
        result_path = run_sft(config)
        meta = json.loads((result_path / "dry_run_meta.json").read_text())
        assert meta["dataset_samples"] == 2


class TestRLDryRun:
    def test_grpo_dry_run(self, tmp_path: Path):
        config = RLConfig(
            dry_run=True,
            algorithm=RLAlgorithm.GRPO,
            output_dir=tmp_path / "rl_out",
            trajectory_path=tmp_path / "nonexistent",
        )
        result_path = run_rl(config)
        assert result_path.exists()
        meta = json.loads((result_path / "dry_run_meta.json").read_text())
        assert meta["status"] == "dry_run"
        assert meta["algorithm"] == "grpo"

    def test_ppo_dry_run(self, tmp_path: Path):
        config = RLConfig(
            dry_run=True,
            algorithm=RLAlgorithm.PPO,
            output_dir=tmp_path / "rl_out",
            trajectory_path=tmp_path / "nonexistent",
        )
        result_path = run_rl(config)
        meta = json.loads((result_path / "dry_run_meta.json").read_text())
        assert meta["algorithm"] == "ppo"

    def test_dry_run_with_trajectories(self, tmp_path: Path):
        traj_dir = tmp_path / "trajectories"
        traj_dir.mkdir()
        trajectory = {
            "prompt": "Investigate BRCA1 in breast cancer",
            "agent_result": {
                "task_id": "t1",
                "agent_id": "a1",
                "agent_type": "literature_analyst",
                "summary": "BRCA1 is associated with breast cancer",
                "success": True,
            },
            "context": {
                "evaluator_score": 0.75,
                "hypothesis_info_gain": 1.5,
                "exploration_breadth": 2,
            },
        }
        with open(traj_dir / "batch_0.jsonl", "w") as f:
            f.write(json.dumps(trajectory) + "\n")

        config = RLConfig(
            dry_run=True,
            output_dir=tmp_path / "rl_out",
            trajectory_path=traj_dir,
        )
        result_path = run_rl(config)
        meta = json.loads((result_path / "dry_run_meta.json").read_text())
        assert meta["trajectories"] == 1
        assert meta["avg_reward"] > 0


class TestEvalDryRun:
    def test_dry_run_produces_report(self, tmp_path: Path):
        config = EvalConfig(
            dry_run=True,
            output_dir=tmp_path / "eval_out",
        )
        report = run_eval(config)
        assert report["status"] == "dry_run"
        assert report["metrics"]["total_samples"] > 0
        assert "avg_total_reward" in report["metrics"]

    def test_dry_run_with_custom_dataset(self, tmp_path: Path):
        eval_dir = tmp_path / "eval_data"
        eval_dir.mkdir()
        sample = {
            "prompt": "What is the role of PD-L1?",
            "reference_answer": "PD-L1 is a checkpoint protein involved in immune evasion.",
        }
        with open(eval_dir / "test.jsonl", "w") as f:
            f.write(json.dumps(sample) + "\n")

        config = EvalConfig(
            dry_run=True,
            eval_dataset_path=eval_dir,
            output_dir=tmp_path / "eval_out",
        )
        report = run_eval(config)
        assert report["metrics"]["total_samples"] == 1
