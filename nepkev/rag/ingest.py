"""Load the knowledge base into kb_chunks: python -m nepkev.rag.ingest [--kb kb] [--dry-run]

Idempotent: chunks already stored with the configured embedding model are kept as they are; new or edited chunks are
embedded, and chunks whose text left the knowledge base are deleted. Running it twice embeds nothing the second time.
The database is NEPKEV_DATABASE_URL; the schema must be migrated first (alembic upgrade head).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sqlalchemy import select

from ..service.db import make_engine, make_sessionmaker
from ..service.settings import Settings
from .config import load_rag_config
from .embed import HFEmbedder
from .kb import KB_DIR, kb_chunks
from .store import KbChunk, sync


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kb", type=Path, default=KB_DIR)
    ap.add_argument("--dry-run", action="store_true", help="report what would change, embed nothing")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    settings, cfg = Settings(), load_rag_config()
    model = settings.rag_embedding_model or cfg.embedding_model
    chunks = kb_chunks(args.kb)
    engine = make_engine(settings.database_url)
    with make_sessionmaker(engine)() as db:
        if args.dry_run:
            stored = dict(db.execute(select(KbChunk.id, KbChunk.embedding_model)).all())
            todo = sum(stored.get(c.id) != model for c in chunks)
            gone = len(set(stored) - {c.id for c in chunks})
            print(f"{len(chunks)} chunks in {args.kb}: {todo} to embed, {gone} to delete")
            return
        report = sync(db, chunks, HFEmbedder(model))   # loads the model only if something needs embedding
    engine.dispose()
    print(f"{len(chunks)} chunks from {args.kb} with {model}: {report}")


if __name__ == "__main__":
    main()
