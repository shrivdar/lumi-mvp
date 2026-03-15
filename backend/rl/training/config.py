"""Training hyperparameters for SFT and RL fine-tuning on DGX Spark.

DGX Spark: Grace Blackwell GB10, 128 GB unified memory, 1 GPU.
Primary model: Qwen2.5-32B-Instruct (fits in FP16 with LoRA / QLoRA).
Fast iteration model: Llama-3.1-8B-Instruct.

Config can be loaded from / saved to YAML/JSON files:
    config = TrainingConfig.from_file("configs/train.yaml")
    config.to_file("configs/train.json")
"""

from __future__ import annotations

import enum
import json
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


class GRPOConfig(BaseModel):
    """GRPO-specific hyperparameters (Group Relative Policy Optimization)."""

    num_generations: int = 4
    temperature: float = 0.7
    kl_coeff: float = 0.1
    clip_range: float = 0.2
    reward_baseline: str = "moving_average"  # "moving_average" | "per_prompt_mean" | "none"
    baseline_ema_decay: float = 0.99


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

    # GRPO-specific (legacy fields kept for backward compat)
    grpo_num_generations: int = 4
    grpo_temperature: float = 0.7
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)

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
    def for_sft_8b(cls) -> TrainingConfig:
        """SFT 8B config — full fine-tuning friendly hyperparameters.

        learning_rate=2e-5, epochs=3, batch_size=4, gradient_accumulation=8.
        """
        return cls(
            sft=SFTConfig(
                base_model=BaseModelChoice.LLAMA_8B,
                learning_rate=2e-5,
                num_epochs=3,
                per_device_batch_size=4,
                gradient_accumulation_steps=8,
            ),
            rl=RLConfig(base_model=BaseModelChoice.LLAMA_8B),
            eval=EvalConfig(base_model=BaseModelChoice.LLAMA_8B),
        )

    @classmethod
    def for_sft_32b_qlora(cls) -> TrainingConfig:
        """SFT 32B (QLoRA) config — LoRA r=64, alpha=128, lr=1e-4."""
        return cls(
            sft=SFTConfig(
                base_model=BaseModelChoice.QWEN_32B,
                learning_rate=1e-4,
                num_epochs=3,
                per_device_batch_size=1,
                gradient_accumulation_steps=16,
                lora=LoRAConfig(r=64, lora_alpha=128),
            ),
            rl=RLConfig(base_model=BaseModelChoice.QWEN_32B),
            eval=EvalConfig(base_model=BaseModelChoice.QWEN_32B),
        )

    @classmethod
    def for_grpo(cls) -> TrainingConfig:
        """GRPO config — kl_coeff=0.1, clip_range=0.2, moving_average baseline."""
        return cls(
            sft=SFTConfig(),
            rl=RLConfig(
                algorithm=RLAlgorithm.GRPO,
                grpo=GRPOConfig(
                    kl_coeff=0.1,
                    clip_range=0.2,
                    reward_baseline="moving_average",
                ),
            ),
            eval=EvalConfig(),
        )

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

    # ------------------------------------------------------------------
    # Config file I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> TrainingConfig:
        """Load config from a YAML or JSON file.

        Supports .yaml/.yml (requires PyYAML) and .json files.
        """
        path = Path(path)
        text = path.read_text()

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:
                raise ImportError("PyYAML required: pip install pyyaml") from e
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)

        return cls.model_validate(data)

    def to_file(self, path: str | Path) -> Path:
        """Save config to a YAML or JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:
                raise ImportError("PyYAML required: pip install pyyaml") from e
            path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        else:
            path.write_text(json.dumps(data, indent=2, default=str))

        return path
