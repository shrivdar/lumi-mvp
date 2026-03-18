"""Add dynamic_tools and strategy_templates tables.

Revision ID: 003
Revises: 002
Create Date: 2026-03-18
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # strategy_templates
    op.create_table(
        "strategy_templates",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(256), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("strategy_type", sa.String(64), nullable=False, server_default="general"),
        sa.Column("template_data", postgresql.JSON(), nullable=False, server_default="{}"),
        sa.Column("success_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failure_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("avg_info_gain", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("tags", postgresql.JSON(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_strategy_templates_name", "strategy_templates", ["name"])
    op.create_index("ix_strategy_templates_type", "strategy_templates", ["strategy_type"])

    # dynamic_tools
    op.create_table(
        "dynamic_tools",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("api_base_url", sa.String(512), nullable=False, server_default=""),
        sa.Column("wrapper_code", sa.Text(), nullable=False),
        sa.Column("test_code", sa.Text(), nullable=False, server_default=""),
        sa.Column("category", sa.String(64), nullable=False, server_default="dynamic"),
        sa.Column("capabilities", postgresql.JSON(), nullable=False, server_default="[]"),
        sa.Column("example_tasks", postgresql.JSON(), nullable=False, server_default="[]"),
        sa.Column("parameters", postgresql.JSON(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(32), nullable=False, server_default="REGISTERED"),
        sa.Column("created_by", sa.String(128), nullable=False, server_default=""),
        sa.Column("usage_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("success_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("success_rate", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_dynamic_tools_name", "dynamic_tools", ["name"])
    op.create_index("ix_dynamic_tools_category", "dynamic_tools", ["category"])
    op.create_index("ix_dynamic_tools_success_rate", "dynamic_tools", ["success_rate"])


def downgrade() -> None:
    op.drop_table("dynamic_tools")
    op.drop_table("strategy_templates")
