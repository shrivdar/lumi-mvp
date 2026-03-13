"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings

from core.models import MCPServerConfig


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 4096

    # ── Database ─────────────────────────────────────
    database_url: str = "postgresql+asyncpg://yohas:yohas@postgres:5432/yohas"

    # ── Redis ────────────────────────────────────────
    redis_url: str = "redis://redis:6379/0"

    # ── Yami / ESM ──────────────────────────────────
    yami_backend: str = "huggingface"
    hf_api_token: str = ""
    esm_model: str = "facebook/esm2_t33_650M_UR50D"

    # ── External APIs ───────────────────────────────
    ncbi_api_key: str = ""
    s2_api_key: str = ""

    # ── Slack HITL ──────────────────────────────────
    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    slack_default_channel: str = "yohas-hitl"

    # ── Research defaults ───────────────────────────
    max_hypothesis_depth: int = 2
    max_mcts_iterations: int = 15
    confidence_threshold: float = 0.7
    hitl_uncertainty_threshold: float = 0.6
    hitl_timeout_seconds: int = 600

    # ── MCP Server Config ───────────────────────────
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    mcp_auto_discover: bool = True

    # ── Container Runtime ───────────────────────────
    container_runtime: str = "docker"  # "docker" or "podman"
    container_default_image: str = "yohas-tool-sandbox:latest"
    container_memory_limit: str = "512m"
    container_cpu_limit: str = "1.0"
    container_network_mode: str = "none"
    container_timeout_seconds: int = 120

    # ── App ──────────────────────────────────────────
    api_key: str = "dev-api-key-change-me"
    log_level: str = "INFO"
    environment: str = "development"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
