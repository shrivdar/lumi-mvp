"""SQLAlchemy ORM models for YOHAS persistent storage."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# research_sessions
# ---------------------------------------------------------------------------

class ResearchSessionRow(Base):
    __tablename__ = "research_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="INITIALIZING")
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    swarm_composition: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    current_iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_nodes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_edges: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_hypotheses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_agents_spawned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    celery_task_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    report_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_research_sessions_status", "status"),
        Index("ix_research_sessions_created_at", "created_at"),
    )


# ---------------------------------------------------------------------------
# kg_snapshots
# ---------------------------------------------------------------------------

class KGSnapshotRow(Base):
    __tablename__ = "kg_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_kg_snapshots_session_id", "session_id"),
    )


# ---------------------------------------------------------------------------
# audit_logs
# ---------------------------------------------------------------------------

class AuditLogRow(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    research_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    event: Mapped[str] = mapped_column(String(128), nullable=False)
    data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_audit_logs_research_id", "research_id"),
        Index("ix_audit_logs_event", "event"),
        Index("ix_audit_logs_created_at", "created_at"),
    )


# ---------------------------------------------------------------------------
# benchmark_runs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# session_checkpoints — MCTS checkpoint/resume state
# ---------------------------------------------------------------------------

class SessionCheckpointRow(Base):
    __tablename__ = "session_checkpoints"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    hypothesis_tree: Mapped[dict] = mapped_column(JSON, nullable=False)
    kg_snapshot_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    agent_results: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    session_tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_agents_spawned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_session_checkpoints_session_id", "session_id"),
        Index("ix_session_checkpoints_iteration", "session_id", "iteration"),
    )


# ---------------------------------------------------------------------------
# hitl_requests — HITL request/response tracking
# ---------------------------------------------------------------------------

class HITLRequestRow(Base):
    __tablename__ = "hitl_requests"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    hypothesis_id: Mapped[str] = mapped_column(String(64), nullable=False)
    uncertainty_composite: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    channel: Mapped[str] = mapped_column(String(128), nullable=True)
    thread_ts: Mapped[str | None] = mapped_column(String(64), nullable=True)
    response: Mapped[str | None] = mapped_column(Text, nullable=True)
    responded: Mapped[bool] = mapped_column(nullable=False, default=False)
    timed_out: Mapped[bool] = mapped_column(nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    responded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_hitl_requests_session_id", "session_id"),
    )


# ---------------------------------------------------------------------------
# benchmark_runs
# ---------------------------------------------------------------------------

class BenchmarkRunRow(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    benchmark_name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    results: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    baseline_comparison: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    total_tasks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    correct_tasks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    run_config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_benchmark_runs_name", "benchmark_name"),
        Index("ix_benchmark_runs_started_at", "started_at"),
    )
