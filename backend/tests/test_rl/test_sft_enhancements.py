"""Tests for enhanced SFT pipeline: quality filtering, instruction-response pairs, config presets."""

from __future__ import annotations

import json
from pathlib import Path

from rl.sft_pipeline import SFTPipeline
from rl.trajectory_format import ToolCallRecord, Trajectory, Turn
from rl.training.config import (
    GRPOConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Quality filtering tests
# ---------------------------------------------------------------------------


class TestQualityFilter:
    def test_filters_no_turns(self):
        pipeline = SFTPipeline(reward_threshold=0.0)
        trajs = [
            Trajectory(task_id="t1", agent_type="lit", reward=1.0, success=True, turns=[]),
        ]
        result = pipeline.quality_filter(trajs)
        assert len(result) == 0

    def test_filters_no_answer(self):
        pipeline = SFTPipeline(reward_threshold=0.0)
        trajs = [
            Trajectory(
                task_id="t1", agent_type="lit", reward=1.0, success=True,
                final_answer="",
                turns=[Turn(turn_number=0, role="assistant", content="thinking")],
            ),
        ]
        result = pipeline.quality_filter(trajs)
        assert len(result) == 0

    def test_filters_not_successful(self):
        pipeline = SFTPipeline(reward_threshold=0.0)
        trajs = [
            Trajectory(
                task_id="t1", agent_type="lit", reward=0.5, success=False,
                final_answer="some answer",
                turns=[Turn(turn_number=0, role="assistant", content="x")],
            ),
        ]
        result = pipeline.quality_filter(trajs)
        assert len(result) == 0

    def test_filters_too_many_tokens(self):
        pipeline = SFTPipeline(reward_threshold=0.0, max_tokens=100)
        trajs = [
            Trajectory(
                task_id="t1", agent_type="lit", reward=1.0, success=True,
                final_answer="answer", total_tokens=200,
                turns=[Turn(turn_number=0, role="assistant", content="x")],
            ),
        ]
        result = pipeline.quality_filter(trajs)
        assert len(result) == 0

    def test_accepts_good_trajectory(self):
        pipeline = SFTPipeline(reward_threshold=0.0)
        trajs = [
            Trajectory(
                task_id="t1", agent_type="lit", reward=1.0, success=True,
                final_answer="BRCA1 is a tumor suppressor",
                turns=[Turn(turn_number=0, role="assistant", content="x")],
            ),
        ]
        result = pipeline.quality_filter(trajs)
        assert len(result) == 1

    def test_filter_and_prepare_chain(self):
        pipeline = SFTPipeline(reward_threshold=0.5)
        trajs = [
            # Good: reward=1.0, success, has answer
            Trajectory(
                task_id="t1", agent_type="lit", reward=1.0, success=True,
                final_answer="answer",
                turns=[Turn(turn_number=0, role="assistant", content="x")],
            ),
            # Bad: low reward
            Trajectory(
                task_id="t2", agent_type="lit", reward=0.1, success=True,
                final_answer="answer",
                turns=[Turn(turn_number=0, role="assistant", content="x")],
            ),
            # Bad: not successful
            Trajectory(
                task_id="t3", agent_type="lit", reward=0.8, success=False,
                final_answer="answer",
                turns=[Turn(turn_number=0, role="assistant", content="x")],
            ),
        ]
        result = pipeline.filter_and_prepare(trajs)
        assert len(result) == 1
        assert result[0].task_id == "t1"


# ---------------------------------------------------------------------------
# Instruction-response pair formatting
# ---------------------------------------------------------------------------


class TestInstructionResponseFormat:
    def test_basic_format(self):
        pipeline = SFTPipeline()
        traj = Trajectory(
            task_id="t1",
            agent_type="literature_analyst",
            instruction="Find BRCA1 info",
            final_answer="BRCA1 is a tumor suppressor gene.",
            turns=[
                Turn(turn_number=0, role="assistant", content="answer",
                     turn_type="answer"),
            ],
            reward=1.0,
        )

        pair = pipeline.format_instruction_response(traj)
        assert "prompt" in pair
        assert "completion" in pair
        assert "metadata" in pair
        assert "Find BRCA1 info" in pair["prompt"]
        assert pair["completion"] == "BRCA1 is a tumor suppressor gene."
        assert pair["metadata"]["reward"] == 1.0

    def test_format_with_tool_calls(self):
        pipeline = SFTPipeline()
        traj = Trajectory(
            task_id="t1",
            agent_type="lit",
            instruction="Search for EGFR",
            final_answer="EGFR drives lung cancer",
            turns=[
                Turn(
                    turn_number=0, role="assistant",
                    content="searching...", turn_type="tool_call",
                    tool_calls=[ToolCallRecord(
                        tool_name="pubmed_search",
                        arguments={"query": "EGFR"},
                        result="Found 100 papers",
                    )],
                ),
                Turn(turn_number=1, role="assistant",
                     content="EGFR drives lung cancer", turn_type="answer"),
            ],
            reward=1.0,
        )

        pair = pipeline.format_instruction_response(traj)
        assert "pubmed_search" in pair["prompt"]
        assert "Found 100 papers" in pair["prompt"]

    def test_export_sft_dataset(self, tmp_path: Path):
        pipeline = SFTPipeline()
        trajs = [
            Trajectory(
                task_id=f"t{i}", agent_type="lit",
                instruction=f"Task {i}", final_answer=f"Answer {i}",
                turns=[Turn(turn_number=0, role="assistant", content=f"a{i}")],
                reward=1.0,
            )
            for i in range(3)
        ]

        out = pipeline.export_sft_dataset(trajs, tmp_path / "sft.jsonl")
        assert out.exists()

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            assert "prompt" in record
            assert "completion" in record
            assert "metadata" in record

    def test_format_all_instruction_response(self):
        pipeline = SFTPipeline()
        trajs = [
            Trajectory(
                task_id=f"t{i}", agent_type="lit",
                instruction=f"Task {i}", final_answer=f"Answer {i}",
                turns=[Turn(turn_number=0, role="assistant", content=f"a{i}")],
                reward=1.0,
            )
            for i in range(2)
        ]
        pairs = pipeline.format_all_instruction_response(trajs)
        assert len(pairs) == 2
        assert pairs[0]["completion"] == "Answer 0"
        assert pairs[1]["completion"] == "Answer 1"


# ---------------------------------------------------------------------------
# Config presets and file I/O
# ---------------------------------------------------------------------------


class TestConfigPresets:
    def test_sft_8b_preset(self):
        cfg = TrainingConfig.for_sft_8b()
        assert "Llama" in cfg.sft.base_model
        assert cfg.sft.learning_rate == 2e-5
        assert cfg.sft.num_epochs == 3
        assert cfg.sft.per_device_batch_size == 4
        assert cfg.sft.gradient_accumulation_steps == 8

    def test_sft_32b_qlora_preset(self):
        cfg = TrainingConfig.for_sft_32b_qlora()
        assert "Qwen" in cfg.sft.base_model
        assert cfg.sft.learning_rate == 1e-4
        assert cfg.sft.lora.r == 64
        assert cfg.sft.lora.lora_alpha == 128

    def test_grpo_preset(self):
        cfg = TrainingConfig.for_grpo()
        assert cfg.rl.algorithm.value == "grpo"
        assert cfg.rl.grpo.kl_coeff == 0.1
        assert cfg.rl.grpo.clip_range == 0.2
        assert cfg.rl.grpo.reward_baseline == "moving_average"

    def test_grpo_config_defaults(self):
        grpo = GRPOConfig()
        assert grpo.kl_coeff == 0.1
        assert grpo.clip_range == 0.2
        assert grpo.reward_baseline == "moving_average"
        assert grpo.baseline_ema_decay == 0.99


class TestConfigFileIO:
    def test_save_and_load_json(self, tmp_path: Path):
        cfg = TrainingConfig.for_grpo()
        path = cfg.to_file(tmp_path / "config.json")
        assert path.exists()

        loaded = TrainingConfig.from_file(path)
        assert loaded.rl.grpo.kl_coeff == cfg.rl.grpo.kl_coeff
        assert loaded.sft.base_model == cfg.sft.base_model

    def test_load_default_config_file(self):
        """Test loading the shipped default config."""
        config_path = Path(__file__).resolve().parents[3] / "configs" / "train_default.json"
        if config_path.exists():
            cfg = TrainingConfig.from_file(config_path)
            assert cfg.rl.grpo.kl_coeff == 0.1
            assert cfg.rl.grpo.reward_baseline == "moving_average"
