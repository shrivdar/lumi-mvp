"""FastAPI application factory — registers routers, middleware, lifespan."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import APIKeyAuthMiddleware, RequestIDMiddleware, TimingMiddleware
from api.routes import agents, graph, health, hypothesis, mcp, monitoring, research
from api.websocket import router as ws_router
from core.audit import configure_audit_logging
from core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle."""
    configure_audit_logging(log_level=settings.log_level)
    yield


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="YOHAS 3.0",
        description="Your Own Hypothesis-driven Agentic Scientist — autonomous biomedical research platform",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware (applied bottom-to-top) ──────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            settings.api_key if settings.environment == "production" else "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(APIKeyAuthMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # ── Routers ────────────────────────────────────────────────────
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(research.router, prefix="/api/v1")
    app.include_router(graph.router, prefix="/api/v1")
    app.include_router(agents.router, prefix="/api/v1")
    app.include_router(hypothesis.router, prefix="/api/v1")
    app.include_router(monitoring.router, prefix="/api/v1")
    app.include_router(mcp.router, prefix="/api/v1")
    app.include_router(ws_router, prefix="/api/v1")

    return app


# Module-level app for uvicorn: `uvicorn api.main:app`
app = create_app()
