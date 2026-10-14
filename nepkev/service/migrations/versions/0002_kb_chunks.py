"""kb_chunks: knowledge-base chunks with pgvector embeddings and an HNSW cosine index

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DIM = 384   # intfloat/multilingual-e5-small; nepkev.rag.store.EMBED_DIM


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "kb_chunks",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("doc_id", sa.String(96), nullable=False),
        sa.Column("domain", sa.String(16), nullable=False),
        sa.Column("lang", sa.String(8), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(DIM), nullable=False),
        sa.Column("embedding_model", sa.String(128), nullable=False),
        sa.Column("updated_at", sa.Date(), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("domain IN ('shop', 'wallet', 'bank')", name="ck_kb_chunks_domain"),
    )
    op.create_index("ix_kb_chunks_doc_id", "kb_chunks", ["doc_id"])
    op.create_index("ix_kb_chunks_domain", "kb_chunks", ["domain"])
    op.execute("CREATE INDEX ix_kb_chunks_embedding ON kb_chunks USING hnsw (embedding vector_cosine_ops)")


def downgrade() -> None:
    op.drop_table("kb_chunks")
