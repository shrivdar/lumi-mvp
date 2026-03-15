"""Training hyperparameters for SFT and RL fine-tuning on DGX Spark.

DGX Spark: Grace Blackwell GB10, 128 GB unified memory, 1 GPU.
Primary model: Qwen2.5-32B-Instruct (fits in FP16 with LoRA).
Fast iteration model: Llama-3.1-8B-Instruct.
"""

from __future__ import annotations

import enum
from pathlib import Path

from pydantic import BaseModel, Field


class BaseModelChoice(enum.StrEnum):
    """Supported base models for fine-tuning."""

    QWEN_32B = "Qwen/Qwen2.5-32B-Instruct"
    LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"


class RLAlgorithm(enum.StrEnum):
    """RL algorithm choices."""

    PPO = "ppo"
    GRPO = "grpo"


class RewardWeights(BaseModel):
    """Weights for the multi-dimensional reward function.

    Must sum to 1.0.  Defaults match the task specification.
    """

    answer_correctness: float = Field(default=0.50, ge=0.0, le=1.0, description="R1: evaluator correctness score")
    kg_quality: float = Field(default=0.20, ge=0.0, le=1.0, description="R2: KG confidence, provenance, contradictions")
    hypothesis_efficiency: float = Field(default=0.15, ge=0.0, le=1.0, description="R3: info-gain per iteration")
    falsification_quality: float = Field(default=0.10, ge=0.0, le=1.0, description="R4: real counter-evidence found")
    format_compliance: float = Field(default=0.05, ge=0.0, le=1.0, description="R5: output format adherence")


class LoRAConfig(BaseModel):
    """Low-Rank Adaptation parameters."""

    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    task_type: str = "CAUSAL_LM"


class SFTConfig(BaseModel):
    """Supervised fine-tuning configuration."""

    base_model: str = BaseModelChoice.QWEN_32B
    output_dir: Path = Path("checkpoints/sft")
    dataset_path: Path = Path("data/trajectories/sft")

    # Training
    num_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 8192

    # Memory optimization for DGX Spark
    bf16: bool = True
    gradient_checkpointing: bool = True
    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    # Logging
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3

    # Misc
    seed: int = 42
    dry_run: bool = False
    dry_run_steps: int = 2


class RLConfig(BaseModel):
    """Reinforcement learning configuration (PPO / GRPO)."""

    base_model: str = BaseModelChoice.QWEN_32B
    sft_checkpoint: Path | None = None
    output_dir: Path = Path("checkpoints/rl")
    trajectory_path: Path = Path("data/trajectories/rl")

    # Algorithm
    algorithm: RLAlgorithm = RLAlgorithm.GRPO
    reward_weights: RewardWeights = Field(default_factory=RewardWeights)

    # PPO-specific
    ppo_epochs: int = 4
    clip_range: float = 0.2
    vf_coef: float = 0.1
    kl_coef: float = 0.05
    kl_target: float | None = 0.01
    gamma: float = 1.0
    lam: float = 0.95

    # GRPO-specific
    grpo_num_generations: int = 4
    grpo_temperature: float = 0.7

    # Training
    num_episodes: int = 1000
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    max_seq_length: int = 8192
    max_response_length: int = 4096

    # Memory optimization for DGX Spark
    bf16: bool = True
    gradient_checkpointing: bool = True
    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    # Logging
    logging_steps: int = 5
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 5

    # Misc
    seed: int = 42
    dry_run: bool = False
    dry_run_steps: int = 2


class EvalConfig(BaseModel):
    """Evaluation configuration for checkpoint benchmarking."""

    checkpoint_path: Path = Path("checkpoints/rl/latest")
    base_model: str = BaseModelChoice.QWEN_32B
    eval_dataset: str = "biomni-eval1"
    eval_dataset_path: Path = Path("data/eval/biomni-eval1")
    output_dir: Path = Path("eval_results")

    # Generation
    temperature: float = 0.1
    max_new_tokens: int = 4096
    num_samples: int = 1

    # Reward
    reward_weights: RewardWeights = Field(default_factory=RewardWeights)

    # Hardware
    bf16: bool = True
    per_device_batch_size: int = 1

    dry_run: bool = False
    dry_run_samples: int = 5


class TrainingConfig(BaseModel):
    """Top-level training configuration aggregating all sub-configs."""

    sft: SFTConfig = Field(default_factory=SFTConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @classmethod
    def for_fast_iteration(cls) -> TrainingConfig:
        """Config preset using Llama-3.1-8B for quick experimentation."""
        return cls(
            sft=SFTConfig(
                base_model=BaseModelChoice.LLAMA_8B,
                per_device_batch_size=2,
                gradient_accumulation_steps=8,
                num_epochs=2,
            ),
            rl=RLConfig(
                base_model=BaseModelChoice.LLAMA_8B,
                per_device_batch_size=2,
                gradient_accumulation_steps=4,
                num_episodes=200,
            ),
            eval=EvalConfig(
                base_model=BaseModelChoice.LLAMA_8B,
                per_device_batch_size=2,
            ),
        )

    @classmethod
    def for_dry_run(cls) -> TrainingConfig:
        """Config preset for CI / smoke-testing without GPU."""
        return cls(
            sft=SFTConfig(dry_run=True, num_epochs=1),
            rl=RLConfig(dry_run=True, num_episodes=2),
            eval=EvalConfig(dry_run=True),
        )
