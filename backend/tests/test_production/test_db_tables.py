"""Tests for DB table definitions and Alembic migration consistency."""

from __future__ import annotations

from db.tables import (
    Base,
    HITLRequestRow,
    ResearchSessionRow,
    SessionCheckpointRow,
)


class TestTableDefinitions:
    def test_research_session_has_production_columns(self):
        """Ensure production columns exist on ResearchSessionRow."""
        columns = {c.name for c in ResearchSessionRow.__table__.columns}
        assert "current_iteration" in columns
        assert "total_nodes" in columns
        assert "total_edges" in columns
        assert "total_hypotheses" in columns
        assert "total_tokens_used" in columns
        assert "total_agents_spawned" in columns
        assert "celery_task_id" in columns
        assert "report_markdown" in columns

    def test_session_checkpoint_table_exists(self):
        assert SessionCheckpointRow.__tablename__ == "session_checkpoints"
        columns = {c.name for c in SessionCheckpointRow.__table__.columns}
        assert "session_id" in columns
        assert "iteration" in columns
        assert "hypothesis_tree" in columns
        assert "kg_snapshot_id" in columns
        assert "session_tokens_used" in columns
        assert "total_agents_spawned" in columns

    def test_hitl_request_table_exists(self):
        assert HITLRequestRow.__tablename__ == "hitl_requests"
        columns = {c.name for c in HITLRequestRow.__table__.columns}
        assert "session_id" in columns
        assert "hypothesis_id" in columns
        assert "uncertainty_composite" in columns
        assert "reason" in columns
        assert "message" in columns
        assert "response" in columns
        assert "responded" in columns
        assert "timed_out" in columns

    def test_all_tables_in_metadata(self):
        """All tables should be registered in the SQLAlchemy metadata."""
        table_names = set(Base.metadata.tables.keys())
        assert "research_sessions" in table_names
        assert "kg_snapshots" in table_names
        assert "audit_logs" in table_names
        assert "benchmark_runs" in table_names
        assert "session_checkpoints" in table_names
        assert "hitl_requests" in table_names

    def test_checkpoint_indexes(self):
        """Verify session_checkpoints has the expected indexes."""
        indexes = {idx.name for idx in SessionCheckpointRow.__table__.indexes}
        assert "ix_session_checkpoints_session_id" in indexes
        assert "ix_session_checkpoints_iteration" in indexes

    def test_hitl_request_indexes(self):
        indexes = {idx.name for idx in HITLRequestRow.__table__.indexes}
        assert "ix_hitl_requests_session_id" in indexes
