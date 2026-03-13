"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 4096

    # Database
    database_url: str = "postgresql+asyncpg://yohas:yohas@postgres:5432/yohas"

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Yami / ESM
    yami_backend: str = "huggingface"
    hf_api_token: str = ""
    esm_model: str = "facebook/esm2_t33_650M_UR50D"

    # External APIs
    ncbi_api_key: str = ""
    s2_api_key: str = ""

    # Slack
    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    slack_default_channel: str = "yohas-hitl"

    # Research defaults
    max_hypothesis_depth: int = 2
    max_mcts_iterations: int = 15
    confidence_threshold: float = 0.7
    hitl_uncertainty_threshold: float = 0.6
    hitl_timeout_seconds: int = 600

    # App
    api_key: str = "dev-api-key-change-me"
    log_level: str = "INFO"
    environment: str = "development"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
