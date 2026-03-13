"""Root test configuration and shared fixtures."""

import os

# Ensure tests don't accidentally use real API keys
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("NCBI_API_KEY", "")
os.environ.setdefault("S2_API_KEY", "")
os.environ.setdefault("HF_API_TOKEN", "")
os.environ.setdefault("SLACK_BOT_TOKEN", "")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///test.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
