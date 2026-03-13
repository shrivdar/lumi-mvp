"""Health and readiness endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter

from agents.templates import AGENT_TEMPLATES
from core.config import settings

router = APIRouter(tags=["health"])

_START_TIME = time.monotonic()


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": "0.1.0",
        "uptime_seconds": int(time.monotonic() - _START_TIME),
    }


@router.get("/health/ready")
async def readiness() -> dict:
    # Best-effort checks — we don't block startup if services are down
    redis_ok = True
    postgres_ok = True
    celery_ok = True

    try:
        import redis as _redis

        r = _redis.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
    except Exception:
        redis_ok = False

    return {
        "redis": redis_ok,
        "postgres": postgres_ok,
        "celery": celery_ok,
    }


@router.get("/templates")
async def list_templates() -> list[dict]:
    return [
        {
            "agent_type": str(t.agent_type),
            "display_name": t.display_name,
            "description": t.description,
            "tools": t.tools,
            "requires_yami": t.requires_yami,
        }
        for t in AGENT_TEMPLATES.values()
    ]
