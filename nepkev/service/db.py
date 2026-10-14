"""Postgres request log: the SQLAlchemy model, engine and session factory. The schema itself is owned by Alembic."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Identity, Index, String, Text, create_engine, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class RouteRequest(Base):
    """One answered POST /route. Replays of an idempotent request are not logged again."""
    __tablename__ = "route_requests"
    __table_args__ = (Index("ix_route_requests_created_at", "created_at"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(32), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    text_hash: Mapped[str] = mapped_column(String(64))            # sha256 hex of the message as received
    text: Mapped[str | None] = mapped_column(Text)               # only with NEPKEV_LOG_TEXT=true
    script: Mapped[str] = mapped_column(String(16))              # devanagari | romanized | mixed
    channel: Mapped[str] = mapped_column(String(16))
    intent: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    needs_clarification: Mapped[bool] = mapped_column(Boolean)
    latency_ms: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(128))
    idempotency_key: Mapped[str | None] = mapped_column(String(255), unique=True)
    request_hash: Mapped[str | None] = mapped_column(String(64))   # sha256 of the canonical body, for key reuse checks
    response: Mapped[dict] = mapped_column(JSONB)                # the body returned, replayed for a repeated key


def make_engine(url: str, connect_timeout: int = 5) -> Engine:
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": connect_timeout})


def make_sessionmaker(engine: Engine) -> sessionmaker:
    return sessionmaker(engine, expire_on_commit=False)


def ping(engine: Engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
