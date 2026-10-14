"""kb_chunks in Postgres with pgvector: the table model, top-k search within a domain, and idempotent sync from the KB.

Similarity is cosine (vectors are L2-normalised, so 1 - cosine distance). The HNSW index answers the nearest-neighbour
query; a domain filter applied after an approximate index scan can return fewer than k rows, so the search turns on
pgvector's iterative index scan, which keeps scanning until k rows pass the filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, DateTime, Integer, String, Text, delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Mapped, Session, mapped_column

from ..service.db import Base
from .embed import Embedder
from .kb import Chunk

EMBED_DIM = 384   # the column's size (migration 0002); must match the configured embedding model


class KbChunk(Base):
    __tablename__ = "kb_chunks"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(96), index=True)
    domain: Mapped[str] = mapped_column(String(16), index=True)
    lang: Mapped[str] = mapped_column(String(8))
    title: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    position: Mapped[int] = mapped_column(Integer)
    embedding = mapped_column(Vector(EMBED_DIM), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128))
    updated_at: Mapped[date] = mapped_column(Date)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    domain: str
    lang: str
    title: str
    text: str
    score: float


def search(db: Session, query: np.ndarray, domain: str, k: int) -> list[Hit]:
    """The k chunks of `domain` most similar to `query`, most similar first."""
    db.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
    dist = KbChunk.embedding.cosine_distance(query.tolist())
    rows = db.execute(select(KbChunk.id, KbChunk.doc_id, KbChunk.domain, KbChunk.lang, KbChunk.title, KbChunk.text,
                             (1 - dist).label("score"))
                      .where(KbChunk.domain == domain).order_by(dist).limit(k)).all()
    return [Hit(r.id, r.doc_id, r.domain, r.lang, r.title, r.text, float(r.score)) for r in rows]


@dataclass
class SyncReport:
    embedded: list[str] = field(default_factory=list)   # new or changed chunks, or embedded by another model
    deleted: list[str] = field(default_factory=list)    # chunks no longer in the KB
    unchanged: int = 0

    def __str__(self) -> str:
        return f"{len(self.embedded)} embedded, {len(self.deleted)} deleted, {self.unchanged} unchanged"


def sync(db: Session, chunks: list[Chunk], embedder: Embedder) -> SyncReport:
    """Make kb_chunks hold exactly `chunks`, embedding only those not already stored with this embedder."""
    if embedder.dim != EMBED_DIM:
        raise ValueError(f"{embedder.name} makes {embedder.dim}-d vectors; kb_chunks stores {EMBED_DIM}-d")
    stored = {r.id: r for r in db.execute(select(KbChunk.id, KbChunk.embedding_model, KbChunk.updated_at,
                                                 KbChunk.position)).all()}
    wanted = {c.id: c for c in chunks}
    report = SyncReport()
    report.deleted = sorted(set(stored) - set(wanted))
    if report.deleted:
        db.execute(delete(KbChunk).where(KbChunk.id.in_(report.deleted)))
    todo = [c for c in chunks if c.id not in stored or stored[c.id].embedding_model != embedder.name]
    report.unchanged = len(chunks) - len(todo)
    if todo:
        vectors = embedder.embed_passages([c.embed_text for c in todo])
        rows = [{"id": c.id, "doc_id": c.doc_id, "domain": c.domain, "lang": c.lang, "title": c.title, "text": c.text,
                 "position": c.position, "embedding": v.tolist(), "embedding_model": embedder.name,
                 "updated_at": c.updated_at} for c, v in zip(todo, vectors)]
        stmt = insert(KbChunk).values(rows)
        db.execute(stmt.on_conflict_do_update(index_elements=[KbChunk.id], set_={
            k: stmt.excluded[k] for k in ("embedding", "embedding_model", "updated_at", "position")} | {
            "ingested_at": func.now()}))
        report.embedded = [c.id for c in todo]
    # the date and position are not part of the id, so they can change while the text stays the same
    for c in chunks:
        old = stored.get(c.id)
        if c.id not in report.embedded and (old.updated_at, old.position) != (c.updated_at, c.position):
            db.execute(update(KbChunk).where(KbChunk.id == c.id).values(updated_at=c.updated_at, position=c.position))
    db.commit()
    return report
