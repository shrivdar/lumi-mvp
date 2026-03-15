"""Model serving configuration — serve trained models via SGLang or vLLM.

Usage:
    python -m rl.training.serve --checkpoint checkpoints/rl/final
    python -m rl.training.serve --engine vllm --port 8080
    python -m rl.training.serve --dry-run
"""

from __future__ import annotations

import argparse
import logging
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ServingEngine(StrEnum):
    SGLANG = "sglang"
    VLLM = "vllm"


class ServeConfig(BaseModel):
    """Configuration for serving a trained YOHAS model."""

    checkpoint_path: Path = Path("checkpoints/rl/final")
    engine: ServingEngine = ServingEngine.SGLANG
    host: str = "0.0.0.0"
    port: int = 8080

    # Model loading
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    trust_remote_code: bool = True

    # Serving
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 16384

    # LoRA adapter (if serving a LoRA checkpoint on top of base model)
    base_model: str | None = None
    lora_path: Path | None = None

    # Misc
    api_key: str | None = None
    log_level: str = "info"

    def build_sglang_cmd(self) -> list[str]:
        """Build the SGLang server launch command."""
        model_path = str(self.checkpoint_path)
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--tp-size", str(self.tensor_parallel_size),
            "--max-total-tokens", str(self.max_model_len),
            "--mem-fraction-static", str(self.gpu_memory_utilization),
        ]
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.api_key:
            cmd.extend(["--api-key", self.api_key])
        return cmd

    def build_vllm_cmd(self) -> list[str]:
        """Build the vLLM server launch command."""
        model_path = str(self.checkpoint_path)
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-num-seqs", str(self.max_num_seqs),
            "--max-num-batched-tokens", str(self.max_num_batched_tokens),
        ]
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.lora_path:
            cmd.extend(["--enable-lora", "--lora-modules", f"yohas={self.lora_path}"])
        if self.api_key:
            cmd.extend(["--api-key", self.api_key])
        return cmd

    def build_cmd(self) -> list[str]:
        """Build the launch command for the configured engine."""
        if self.engine == ServingEngine.SGLANG:
            return self.build_sglang_cmd()
        return self.build_vllm_cmd()

    def to_docker_env(self) -> dict[str, str]:
        """Generate environment variables for docker-compose.serve.yml."""
        env = {
            "MODEL_PATH": str(self.checkpoint_path),
            "SERVE_ENGINE": self.engine.value,
            "SERVE_HOST": self.host,
            "SERVE_PORT": str(self.port),
            "SERVE_DTYPE": self.dtype,
            "SERVE_TP_SIZE": str(self.tensor_parallel_size),
            "SERVE_MAX_MODEL_LEN": str(self.max_model_len),
            "SERVE_GPU_MEM_UTIL": str(self.gpu_memory_utilization),
            "SERVE_MAX_NUM_SEQS": str(self.max_num_seqs),
        }
        if self.base_model:
            env["BASE_MODEL"] = self.base_model
        if self.lora_path:
            env["LORA_PATH"] = str(self.lora_path)
        if self.api_key:
            env["SERVE_API_KEY"] = self.api_key
        return env


def run_serve(config: ServeConfig) -> None:
    """Launch the model serving process."""
    cmd = config.build_cmd()
    logger.info("Serving model: %s", config.checkpoint_path)
    logger.info("Engine: %s", config.engine.value)
    logger.info("Endpoint: http://%s:%d", config.host, config.port)
    logger.info("Command: %s", " ".join(cmd))

    if not config.checkpoint_path.exists():
        logger.warning(
            "Checkpoint path %s does not exist. "
            "Run SFT/RL training first or provide a valid --checkpoint.",
            config.checkpoint_path,
        )

    import subprocess
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="YOHAS Model Serving (SGLang / vLLM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to model checkpoint")
    parser.add_argument(
        "--engine",
        choices=["sglang", "vllm"],
        default="sglang",
        help="Serving engine",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Max sequence length")
    parser.add_argument("--base-model", type=str, default=None, help="Base model (for LoRA serving)")
    parser.add_argument("--lora-path", type=Path, default=None, help="LoRA adapter path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and command without launching",
    )
    parser.add_argument(
        "--print-docker-env",
        action="store_true",
        help="Print docker environment variables and exit",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    config = ServeConfig(
        engine=ServingEngine(args.engine),
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
    )
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if args.base_model:
        config.base_model = args.base_model
    if args.lora_path:
        config.lora_path = args.lora_path

    if args.print_docker_env:
        for k, v in config.to_docker_env().items():
            print(f"{k}={v}")
        return

    if args.dry_run:
        logger.info("[DRY RUN] Config: %s", config.model_dump_json(indent=2))
        logger.info("[DRY RUN] Command: %s", " ".join(config.build_cmd()))
        return

    run_serve(config)


if __name__ == "__main__":
    main()
