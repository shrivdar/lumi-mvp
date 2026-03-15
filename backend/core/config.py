"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings

from core.models import MCPServerConfig


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────
    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-4-6"
    llm_fast_model: str = "claude-sonnet-4-20250514"
    llm_cheap_model: str = "claude-haiku-4-5-20251001"
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

    # ── Biosecurity Screening ────────────────────────
    biosecurity_screening_model: str = "claude-haiku-4-5-20251001"

    # ── Research defaults ───────────────────────────
    max_hypothesis_depth: int = 2
    max_mcts_iterations: int = 30
    confidence_threshold: float = 0.7
    hitl_uncertainty_threshold: float = 0.6
    hitl_timeout_seconds: int = 600

    # ── MCP Server Config ───────────────────────────
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    mcp_auto_discover: bool = True
    mcp_health_check_interval: int = 60  # seconds between health checks, 0 to disable
    mcp_rate_limit_rps: float = 5.0
    mcp_request_timeout: int = 30
    mcp_max_retries: int = 3

    # ── Container Runtime ───────────────────────────
    container_runtime: str = "docker"  # "docker" or "podman"
    container_default_image: str = "yohas-tool-sandbox:latest"
    container_memory_limit: str = "512m"
    container_cpu_limit: str = "1.0"
    container_network_mode: str = "none"
    container_timeout_seconds: int = 120

    # ── Python REPL ──────────────────────────────────
    repl_image: str = "yohas-repl:latest"
    repl_memory_limit: str = "1g"
    repl_cpu_limit: str = "2.0"
    repl_timeout_seconds: int = 120
    repl_max_output_chars: int = 10_000
    data_lake_path: str = "/data"  # host path mounted read-only into REPL containers

    # ── App ──────────────────────────────────────────
    api_key: str = "dev-api-key-change-me"
    log_level: str = "INFO"
    environment: str = "development"
    frontend_url: str = "http://localhost:3000"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
