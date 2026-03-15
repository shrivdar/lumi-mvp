"""Supervised fine-tuning (SFT) script for YOHAS agent models.

Targets DGX Spark with LoRA + gradient checkpointing.
Uses HuggingFace Transformers + PEFT; compatible with veRL / SkyRL launchers.

Usage:
    python -m rl.training.sft                     # default Qwen2.5-32B
    python -m rl.training.sft --dry-run            # smoke test without GPU
    python -m rl.training.sft --base-model llama   # fast iteration with 8B
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rl.training.config import SFTConfig, TrainingConfig

logger = logging.getLogger(__name__)


def _load_dataset(dataset_path: Path, max_samples: int | None = None) -> list[dict]:
    """Load SFT dataset from JSONL files.

    Expected format per line:
        {"prompt": "...", "completion": "...", "metadata": {...}}
    """
    samples: list[dict] = []
    if not dataset_path.exists():
        logger.warning("Dataset path %s does not exist — using empty dataset for dry run", dataset_path)
        return samples

    for jsonl_file in sorted(dataset_path.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
                if max_samples and len(samples) >= max_samples:
                    return samples
    return samples


def run_sft(config: SFTConfig) -> Path:
    """Run supervised fine-tuning.

    Returns:
        Path to the saved checkpoint directory.
    """
    logger.info("Starting SFT training")
    logger.info("  Base model: %s", config.base_model)
    logger.info("  Output dir: %s", config.output_dir)
    logger.info("  LoRA r=%d, alpha=%d", config.lora.r, config.lora.lora_alpha)
    logger.info("  Dry run: %s", config.dry_run)

    # Load dataset
    max_samples = config.dry_run_steps * config.per_device_batch_size if config.dry_run else None
    dataset = _load_dataset(config.dataset_path, max_samples=max_samples)
    logger.info("  Dataset samples: %d", len(dataset))

    if config.dry_run:
        logger.info("[DRY RUN] Skipping actual training — validating config only")
        config.output_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "status": "dry_run",
            "base_model": config.base_model,
            "dataset_samples": len(dataset),
            "config": config.model_dump(mode="json"),
        }
        meta_path = config.output_dir / "dry_run_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        logger.info("[DRY RUN] Wrote metadata to %s", meta_path)
        return config.output_dir

    # ------------------------------------------------------------------
    # Real training path — requires torch + transformers + peft
    # ------------------------------------------------------------------
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as e:
        logger.error(
            "SFT training requires: torch, transformers, peft, trl. "
            "Install with: pip install torch transformers peft trl\n%s",
            e,
        )
        raise

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        task_type=config.lora.task_type,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Format dataset for SFTTrainer
    def _format_sample(sample: dict) -> str:
        prompt = sample.get("prompt", "")
        completion = sample.get("completion", "")
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"

    formatted = [{"text": _format_sample(s)} for s in dataset]

    from datasets import Dataset

    train_dataset = Dataset.from_list(formatted)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        max_seq_length=config.max_seq_length,
    )

    trainer.train()
    trainer.save_model(str(config.output_dir / "final"))
    tokenizer.save_pretrained(str(config.output_dir / "final"))

    logger.info("SFT training complete. Checkpoint at %s", config.output_dir / "final")
    return config.output_dir / "final"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="YOHAS SFT Training")
    parser.add_argument("--config", type=Path, default=None, help="Path to training config file (YAML/JSON)")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without GPU training")
    parser.add_argument(
        "--base-model",
        choices=["qwen", "llama"],
        default="qwen",
        help="Base model selection (default: qwen)",
    )
    parser.add_argument(
        "--preset",
        choices=["default", "sft-8b", "sft-32b-qlora", "fast"],
        default=None,
        help="Use a named config preset",
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Override dataset path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Config priority: --config file > --preset > --base-model fallback
    if args.config:
        config = TrainingConfig.from_file(args.config).sft
    elif args.preset == "sft-8b":
        config = TrainingConfig.for_sft_8b().sft
    elif args.preset == "sft-32b-qlora":
        config = TrainingConfig.for_sft_32b_qlora().sft
    elif args.preset == "fast":
        config = TrainingConfig.for_fast_iteration().sft
    elif args.base_model == "llama":
        config = TrainingConfig.for_fast_iteration().sft
    else:
        config = SFTConfig()

    if args.dry_run:
        config.dry_run = True
    if args.dataset:
        config.dataset_path = args.dataset
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.num_epochs = args.epochs

    run_sft(config)


if __name__ == "__main__":
    main()
