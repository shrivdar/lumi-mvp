"""RL training script for YOHAS agent models (PPO / GRPO).

Uses the multi-dimensional reward function from reward.py.
Compatible with veRL / SkyRL launchers for distributed training.
Targets DGX Spark with LoRA + gradient checkpointing.

Usage:
    python -m rl.training.rl                          # default GRPO + Qwen2.5-32B
    python -m rl.training.rl --algorithm ppo           # PPO variant
    python -m rl.training.rl --dry-run                 # smoke test without GPU
    python -m rl.training.rl --base-model llama        # fast iteration
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from core.models import AgentResult, AgentType
from rl.training.config import RLAlgorithm, RLConfig, TrainingConfig
from rl.training.reward import RewardBreakdown, TrajectoryContext, compute_reward

logger = logging.getLogger(__name__)


def _load_trajectories(
    trajectory_path: Path,
    max_samples: int | None = None,
) -> list[dict]:
    """Load RL trajectories from JSONL.

    Expected format per line:
        {
            "prompt": "...",
            "agent_result": {...},   # serialized AgentResult
            "context": {...}         # serialized TrajectoryContext
        }
    """
    trajectories: list[dict] = []
    if not trajectory_path.exists():
        logger.warning("Trajectory path %s does not exist — using empty set for dry run", trajectory_path)
        return trajectories

    for jsonl_file in sorted(trajectory_path.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                trajectories.append(json.loads(line))
                if max_samples and len(trajectories) >= max_samples:
                    return trajectories
    return trajectories


def _compute_trajectory_reward(trajectory: dict, config: RLConfig) -> RewardBreakdown:
    """Parse trajectory dict and compute reward."""
    agent_result_data = trajectory.get("agent_result", {})
    context_data = trajectory.get("context", {})

    result = AgentResult(
        task_id=agent_result_data.get("task_id", ""),
        agent_id=agent_result_data.get("agent_id", ""),
        agent_type=agent_result_data.get("agent_type", AgentType.LITERATURE_ANALYST),
        **{k: v for k, v in agent_result_data.items() if k not in ("task_id", "agent_id", "agent_type")},
    )

    context = TrajectoryContext(
        evaluator_score=context_data.get("evaluator_score", 0.0),
        hypothesis_info_gain=context_data.get("hypothesis_info_gain", 0.0),
        exploration_breadth=context_data.get("exploration_breadth", 0),
        total_iterations=context_data.get("total_iterations", 1),
        expected_format=context_data.get("expected_format", "json"),
    )

    return compute_reward(result, context, config.reward_weights)


def run_rl(config: RLConfig) -> Path:
    """Run RL training (PPO or GRPO).

    Returns:
        Path to the saved checkpoint directory.
    """
    logger.info("Starting RL training")
    logger.info("  Algorithm: %s", config.algorithm)
    logger.info("  Base model: %s", config.base_model)
    logger.info("  SFT checkpoint: %s", config.sft_checkpoint)
    logger.info("  Output dir: %s", config.output_dir)
    logger.info("  Dry run: %s", config.dry_run)

    # Load trajectories
    max_samples = config.dry_run_steps * config.per_device_batch_size if config.dry_run else None
    trajectories = _load_trajectories(config.trajectory_path, max_samples=max_samples)
    logger.info("  Trajectories loaded: %d", len(trajectories))

    # Compute rewards for all trajectories (for logging / validation)
    rewards = [_compute_trajectory_reward(t, config) for t in trajectories]
    if rewards:
        avg_reward = sum(r.total for r in rewards) / len(rewards)
        logger.info("  Avg trajectory reward: %.4f", avg_reward)

    if config.dry_run:
        logger.info("[DRY RUN] Skipping actual training — validating config and rewards")
        config.output_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "status": "dry_run",
            "algorithm": config.algorithm,
            "base_model": config.base_model,
            "sft_checkpoint": str(config.sft_checkpoint) if config.sft_checkpoint else None,
            "trajectories": len(trajectories),
            "reward_weights": config.reward_weights.model_dump(),
            "avg_reward": sum(r.total for r in rewards) / len(rewards) if rewards else 0.0,
            "sample_rewards": [r.as_dict() for r in rewards[:3]],
            "config": config.model_dump(mode="json"),
        }
        meta_path = config.output_dir / "dry_run_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        logger.info("[DRY RUN] Wrote metadata to %s", meta_path)
        return config.output_dir

    # ------------------------------------------------------------------
    # Real training path — requires torch + transformers + peft + trl
    # ------------------------------------------------------------------
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import PPOConfig, PPOTrainer  # noqa: F401
    except ImportError as e:
        logger.error(
            "RL training requires: torch, transformers, peft, trl. "
            "Install with: pip install torch transformers peft trl\n%s",
            e,
        )
        raise

    model_path = str(config.sft_checkpoint) if config.sft_checkpoint else config.base_model

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )

    # Apply LoRA if starting from base model (SFT checkpoint already has adapters)
    if not config.sft_checkpoint:
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=config.lora.target_modules,
            task_type=config.lora.task_type,
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    if config.algorithm == RLAlgorithm.GRPO:
        _run_grpo(model, tokenizer, trajectories, rewards, config)
    else:
        _run_ppo(model, tokenizer, trajectories, rewards, config)

    # Save final checkpoint
    final_path = config.output_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    logger.info("RL training complete. Checkpoint at %s", final_path)
    return final_path


def _run_grpo(model, tokenizer, trajectories, rewards, config: RLConfig) -> None:  # noqa: ANN001
    """GRPO training loop: generate N completions, rank by reward, update policy."""
    import torch
    from torch.optim import AdamW

    logger.info("Running GRPO with %d generations per prompt", config.grpo_num_generations)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    model.train()

    for step, (traj, reward) in enumerate(zip(trajectories, rewards)):
        if step >= config.num_episodes:
            break

        prompt = traj.get("prompt", "")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_seq_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate N completions (used for ranking in full GRPO; simplified here)
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=config.max_response_length,
                num_return_sequences=config.grpo_num_generations,
                temperature=config.grpo_temperature,
                do_sample=True,
            )

        # Use reward as advantage signal (simplified GRPO)
        advantage = reward.total - 0.5  # center around 0.5 baseline

        # Policy gradient step
        logits = model(**inputs).logits
        loss = -advantage * logits.mean()  # simplified — real impl uses per-token log-probs
        loss.backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % config.logging_steps == 0:
            logger.info(
                "  GRPO step %d/%d  reward=%.4f  loss=%.4f",
                step, config.num_episodes, reward.total, loss.item(),
            )


def _run_ppo(model, tokenizer, trajectories, rewards, config: RLConfig) -> None:  # noqa: ANN001
    """PPO training loop using TRL's PPOTrainer."""
    from trl import PPOConfig, PPOTrainer

    ppo_config = PPOConfig(
        model_name=config.base_model,
        learning_rate=config.learning_rate,
        batch_size=config.per_device_batch_size,
        mini_batch_size=1,
        ppo_epochs=config.ppo_epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        seed=config.seed,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    for step, (traj, reward) in enumerate(zip(trajectories, rewards)):
        if step >= config.num_episodes:
            break

        prompt = traj.get("prompt", "")
        query_tensors = tokenizer(prompt, return_tensors="pt").input_ids[0]

        response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=config.max_response_length)
        response_tensors = response_tensors.squeeze()

        import torch

        reward_tensor = [torch.tensor(reward.total)]
        stats = ppo_trainer.step([query_tensors], [response_tensors], reward_tensor)

        if step % config.logging_steps == 0:
            logger.info(
                "  PPO step %d/%d  reward=%.4f  kl=%.4f",
                step,
                config.num_episodes,
                reward.total,
                stats.get("objective/kl", 0.0),
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="YOHAS RL Training (PPO/GRPO)")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without GPU training")
    parser.add_argument("--algorithm", choices=["ppo", "grpo"], default="grpo", help="RL algorithm")
    parser.add_argument(
        "--base-model",
        choices=["qwen", "llama"],
        default="qwen",
        help="Base model selection",
    )
    parser.add_argument("--sft-checkpoint", type=Path, default=None, help="Path to SFT checkpoint")
    parser.add_argument("--trajectories", type=Path, default=None, help="Override trajectory path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of episodes")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.base_model == "llama":
        config = TrainingConfig.for_fast_iteration().rl
    else:
        config = RLConfig()

    config.algorithm = RLAlgorithm(args.algorithm)
    if args.dry_run:
        config.dry_run = True
    if args.sft_checkpoint:
        config.sft_checkpoint = args.sft_checkpoint
    if args.trajectories:
        config.trajectory_path = args.trajectories
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.episodes:
        config.num_episodes = args.episodes

    run_rl(config)


if __name__ == "__main__":
    main()
