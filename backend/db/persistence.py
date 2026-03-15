"""Persistence layer — save/load research sessions, KG snapshots, checkpoints to PostgreSQL."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.tables import (
    HITLRequestRow,
    KGSnapshotRow,
    ResearchSessionRow,
    SessionCheckpointRow,
)

logger = structlog.get_logger(__name__)


class SessionPersistence:
    """Async persistence for research sessions and related data."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Research sessions
    # ------------------------------------------------------------------

    async def create_session(
        self,
        session_id: str,
        query: str,
        config: dict[str, Any],
        celery_task_id: str | None = None,
    ) -> ResearchSessionRow:
        """Insert a new research session row."""
        row = ResearchSessionRow(
            id=uuid.UUID(session_id),
            query=query,
            status="INITIALIZING",
            config=config,
            swarm_composition=[],
            celery_task_id=celery_task_id,
        )
        self.db.add(row)
        await self.db.flush()
        logger.info("session_persisted", session_id=session_id)
        return row

    async def update_session_status(
        self,
        session_id: str,
        status: str,
        *,
        current_iteration: int | None = None,
        total_nodes: int | None = None,
        total_edges: int | None = None,
        total_hypotheses: int | None = None,
        total_tokens_used: int | None = None,
        total_agents_spawned: int | None = None,
    ) -> None:
        """Update session status and stats."""
        row = await self._get_session_row(session_id)
        if row is None:
            return
        row.status = status
        if current_iteration is not None:
            row.current_iteration = current_iteration
        if total_nodes is not None:
            row.total_nodes = total_nodes
        if total_edges is not None:
            row.total_edges = total_edges
        if total_hypotheses is not None:
            row.total_hypotheses = total_hypotheses
        if total_tokens_used is not None:
            row.total_tokens_used = total_tokens_used
        if total_agents_spawned is not None:
            row.total_agents_spawned = total_agents_spawned
        await self.db.flush()

    async def complete_session(
        self,
        session_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        report_markdown: str | None = None,
    ) -> None:
        """Mark session as completed/failed with result."""
        row = await self._get_session_row(session_id)
        if row is None:
            return
        row.status = status
        row.result = result
        row.report_markdown = report_markdown
        row.completed_at = datetime.now(UTC)
        await self.db.flush()

    async def get_session(self, session_id: str) -> ResearchSessionRow | None:
        return await self._get_session_row(session_id)

    async def list_sessions(
        self,
        status: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[ResearchSessionRow], int]:
        """List sessions with optional status filter."""
        stmt = select(ResearchSessionRow)
        if status:
            stmt = stmt.where(ResearchSessionRow.status == status)
        count_stmt = select(ResearchSessionRow.id)
        if status:
            count_stmt = count_stmt.where(ResearchSessionRow.status == status)

        count_result = await self.db.execute(count_stmt)
        total = len(count_result.all())

        stmt = stmt.order_by(ResearchSessionRow.created_at.desc())
        stmt = stmt.offset(offset).limit(limit)
        result = await self.db.execute(stmt)
        rows = list(result.scalars().all())
        return rows, total

    async def _get_session_row(self, session_id: str) -> ResearchSessionRow | None:
        stmt = select(ResearchSessionRow).where(
            ResearchSessionRow.id == uuid.UUID(session_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # KG snapshots
    # ------------------------------------------------------------------

    async def save_kg_snapshot(
        self,
        session_id: str,
        snapshot: dict[str, Any],
    ) -> str:
        """Save a KG snapshot and return its ID."""
        snapshot_id = str(uuid.uuid4())
        row = KGSnapshotRow(
            id=uuid.UUID(snapshot_id),
            session_id=uuid.UUID(session_id),
            snapshot=snapshot,
        )
        self.db.add(row)
        await self.db.flush()
        return snapshot_id

    async def get_latest_kg_snapshot(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Get the most recent KG snapshot for a session."""
        stmt = (
            select(KGSnapshotRow)
            .where(KGSnapshotRow.session_id == uuid.UUID(session_id))
            .order_by(KGSnapshotRow.created_at.desc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        row = result.scalar_one_or_none()
        return row.snapshot if row else None

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    async def save_checkpoint(
        self,
        session_id: str,
        iteration: int,
        hypothesis_tree: dict[str, Any],
        kg_snapshot_id: str | None,
        agent_results: list[dict[str, Any]],
        session_tokens_used: int,
        total_agents_spawned: int,
    ) -> str:
        """Save an MCTS checkpoint."""
        checkpoint_id = str(uuid.uuid4())
        row = SessionCheckpointRow(
            id=uuid.UUID(checkpoint_id),
            session_id=uuid.UUID(session_id),
            iteration=iteration,
            hypothesis_tree=hypothesis_tree,
            kg_snapshot_id=uuid.UUID(kg_snapshot_id) if kg_snapshot_id else None,
            agent_results=agent_results,
            session_tokens_used=session_tokens_used,
            total_agents_spawned=total_agents_spawned,
        )
        self.db.add(row)
        await self.db.flush()
        logger.info(
            "checkpoint_saved",
            session_id=session_id,
            iteration=iteration,
            checkpoint_id=checkpoint_id,
        )
        return checkpoint_id

    async def get_latest_checkpoint(
        self,
        session_id: str,
    ) -> SessionCheckpointRow | None:
        """Get the latest checkpoint for a session."""
        stmt = (
            select(SessionCheckpointRow)
            .where(SessionCheckpointRow.session_id == uuid.UUID(session_id))
            .order_by(SessionCheckpointRow.iteration.desc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # HITL requests
    # ------------------------------------------------------------------

    async def save_hitl_request(
        self,
        session_id: str,
        hypothesis_id: str,
        uncertainty_composite: float,
        reason: str,
        message: str,
        channel: str | None = None,
    ) -> str:
        """Record a HITL request."""
        hitl_id = str(uuid.uuid4())
        row = HITLRequestRow(
            id=uuid.UUID(hitl_id),
            session_id=uuid.UUID(session_id),
            hypothesis_id=hypothesis_id,
            uncertainty_composite=uncertainty_composite,
            reason=reason,
            message=message,
            channel=channel,
        )
        self.db.add(row)
        await self.db.flush()
        return hitl_id

    async def update_hitl_response(
        self,
        hitl_id: str,
        response: str | None,
        timed_out: bool = False,
    ) -> None:
        """Record a HITL response."""
        stmt = select(HITLRequestRow).where(
            HITLRequestRow.id == uuid.UUID(hitl_id)
        )
        result = await self.db.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return
        row.response = response
        row.responded = response is not None
        row.timed_out = timed_out
        row.responded_at = datetime.now(UTC) if response else None
        await self.db.flush()

    async def get_hitl_requests(
        self,
        session_id: str,
    ) -> list[HITLRequestRow]:
        """Get all HITL requests for a session."""
        stmt = (
            select(HITLRequestRow)
            .where(HITLRequestRow.session_id == uuid.UUID(session_id))
            .order_by(HITLRequestRow.created_at.desc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
