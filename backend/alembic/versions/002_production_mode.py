"""Production mode: checkpoints, HITL tracking, session stats columns.

Revision ID: 002
Revises: 001
Create Date: 2026-03-15
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add production columns to research_sessions
    op.add_column("research_sessions", sa.Column("current_iteration", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("research_sessions", sa.Column("total_nodes", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("research_sessions", sa.Column("total_edges", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("research_sessions", sa.Column("total_hypotheses", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("research_sessions", sa.Column("total_tokens_used", sa.Integer(), nullable=False, server_default="0"))
    op.add_column(
        "research_sessions",
        sa.Column("total_agents_spawned", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("research_sessions", sa.Column("celery_task_id", sa.String(128), nullable=True))
    op.add_column("research_sessions", sa.Column("report_markdown", sa.Text(), nullable=True))

    # Session checkpoints
    op.create_table(
        "session_checkpoints",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("hypothesis_tree", postgresql.JSON(), nullable=False),
        sa.Column("kg_snapshot_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("agent_results", postgresql.JSON(), nullable=False, server_default="[]"),
        sa.Column("session_tokens_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_agents_spawned", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_session_checkpoints_session_id", "session_checkpoints", ["session_id"])
    op.create_index("ix_session_checkpoints_iteration", "session_checkpoints", ["session_id", "iteration"])

    # HITL requests
    op.create_table(
        "hitl_requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("hypothesis_id", sa.String(64), nullable=False),
        sa.Column("uncertainty_composite", sa.Float(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("channel", sa.String(128), nullable=True),
        sa.Column("thread_ts", sa.String(64), nullable=True),
        sa.Column("response", sa.Text(), nullable=True),
        sa.Column("responded", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("timed_out", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("responded_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_hitl_requests_session_id", "hitl_requests", ["session_id"])


def downgrade() -> None:
    op.drop_table("hitl_requests")
    op.drop_table("session_checkpoints")

    op.drop_column("research_sessions", "report_markdown")
    op.drop_column("research_sessions", "celery_task_id")
    op.drop_column("research_sessions", "total_agents_spawned")
    op.drop_column("research_sessions", "total_tokens_used")
    op.drop_column("research_sessions", "total_hypotheses")
    op.drop_column("research_sessions", "total_edges")
    op.drop_column("research_sessions", "total_nodes")
    op.drop_column("research_sessions", "current_iteration")
