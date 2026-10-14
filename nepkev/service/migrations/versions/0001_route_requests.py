"""route_requests: one row per answered POST /route

Revision ID: 0001
Revises:
Create Date: 2026-10-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "route_requests",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("request_id", sa.String(32), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("text_hash", sa.String(64), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("script", sa.String(16), nullable=False),
        sa.Column("channel", sa.String(16), nullable=False),
        sa.Column("intent", sa.String(64), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("needs_clarification", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=True, unique=True),
        sa.Column("request_hash", sa.String(64), nullable=True),
        sa.Column("response", postgresql.JSONB(), nullable=False),
        sa.CheckConstraint("script IN ('devanagari', 'romanized', 'mixed')", name="ck_route_requests_script"),
        sa.CheckConstraint("confidence BETWEEN 0 AND 1", name="ck_route_requests_confidence"),
    )
    op.create_index("ix_route_requests_created_at", "route_requests", ["created_at"])


def downgrade() -> None:
    op.drop_table("route_requests")
