"""Tests for RL training configuration."""

from __future__ import annotations

from pathlib import Path

from rl.training.config import (
    BaseModelChoice,
    EvalConfig,
    LoRAConfig,
    RewardWeights,
    RLAlgorithm,
    RLConfig,
    SFTConfig,
    TrainingConfig,
)


class TestRewardWeights:
    def test_default_weights_sum_to_one(self):
        w = RewardWeights()
        total = (
            w.answer_correctness
            + w.kg_quality
            + w.hypothesis_efficiency
            + w.falsification_quality
            + w.format_compliance
        )
        assert abs(total - 1.0) < 1e-9

    def test_custom_weights(self):
        w = RewardWeights(
            answer_correctness=0.4,
            kg_quality=0.3,
            hypothesis_efficiency=0.1,
            falsification_quality=0.1,
            format_compliance=0.1,
        )
        assert w.answer_correctness == 0.4
        assert w.kg_quality == 0.3


class TestLoRAConfig:
    def test_defaults(self):
        lora = LoRAConfig()
        assert lora.r == 64
        assert lora.lora_alpha == 128
        assert "q_proj" in lora.target_modules
        assert lora.task_type == "CAUSAL_LM"


class TestSFTConfig:
    def test_defaults(self):
        cfg = SFTConfig()
        assert cfg.base_model == BaseModelChoice.QWEN_32B
        assert cfg.bf16 is True
        assert cfg.gradient_checkpointing is True
        assert cfg.max_seq_length == 8192
        assert cfg.dry_run is False

    def test_output_dir_is_path(self):
        cfg = SFTConfig()
        assert isinstance(cfg.output_dir, Path)


class TestRLConfig:
    def test_defaults(self):
        cfg = RLConfig()
        assert cfg.algorithm == RLAlgorithm.GRPO
        assert cfg.kl_coef == 0.05
        assert cfg.grpo_num_generations == 4

    def test_ppo_algorithm(self):
        cfg = RLConfig(algorithm=RLAlgorithm.PPO)
        assert cfg.algorithm == RLAlgorithm.PPO


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.eval_dataset == "biomni-eval1"
        assert cfg.temperature == 0.1


class TestTrainingConfig:
    def test_default_aggregation(self):
        cfg = TrainingConfig()
        assert cfg.sft.base_model == BaseModelChoice.QWEN_32B
        assert cfg.rl.algorithm == RLAlgorithm.GRPO
        assert cfg.eval.eval_dataset == "biomni-eval1"

    def test_fast_iteration_preset(self):
        cfg = TrainingConfig.for_fast_iteration()
        assert cfg.sft.base_model == BaseModelChoice.LLAMA_8B
        assert cfg.rl.base_model == BaseModelChoice.LLAMA_8B
        assert cfg.eval.base_model == BaseModelChoice.LLAMA_8B
        assert cfg.sft.per_device_batch_size == 2

    def test_dry_run_preset(self):
        cfg = TrainingConfig.for_dry_run()
        assert cfg.sft.dry_run is True
        assert cfg.rl.dry_run is True
        assert cfg.eval.dry_run is True
        assert cfg.sft.num_epochs == 1
        assert cfg.rl.num_episodes == 2
