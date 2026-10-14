"""API and database test setup: a stub router, a hashing embedder and a scripted answer model (no models, no
downloads, no API calls) and a real Postgres migrated with Alembic.

Set NEPKEV_TEST_DATABASE_URL (e.g. postgresql+psycopg://nepkev:nepkev@localhost:5432/nepkev_test) to run the tests
that need a database; they are skipped without it. The database's tables are dropped and re-created.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="needs the service extra (uv sync --extra service)")

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import Engine, text  # noqa: E402

import numpy as np  # noqa: E402

from nepkev import ROOT  # noqa: E402
from nepkev.rag.llm import Draft  # noqa: E402
from nepkev.rag.store import EMBED_DIM, Hit  # noqa: E402
from nepkev.router import Routing  # noqa: E402
from nepkev.service.app import create_app  # noqa: E402
from nepkev.service.db import make_engine  # noqa: E402
from nepkev.service.settings import Settings  # noqa: E402
from nepkev.taxonomy import load_taxonomy  # noqa: E402

TEST_DB_ENV = "NEPKEV_TEST_DATABASE_URL"
# Nothing listens on port 1, so connecting fails at once.
UNREACHABLE_DB = "postgresql+psycopg://nepkev:nepkev@127.0.0.1:1/nepkev"


class StubRouter:
    """Answers from a table of canned routings; anything else goes to the domain's first category at 0.9."""
    version = "stub@0000000"

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.canned: dict[str, tuple[dict[str, float], float]] = {}
        self.before_return = None   # hook run inside route(), to simulate work happening concurrently

    def route(self, message: str, domain: str) -> Routing:
        self.calls.append((message, domain))
        cats = load_taxonomy().categories(domain)
        if message in self.canned:
            top, clarification = self.canned[message]
        else:
            top, clarification = {cats[0]: 0.9}, 0.02
        rest = (1 - sum(top.values())) / (len(cats) - len(top))
        if self.before_return is not None:
            self.before_return()
        return Routing(domain=domain, issue={c: top.get(c, rest) for c in cats}, clarification=clarification)


class HashEmbedder:
    """Bag of lowercased words hashed into EMBED_DIM buckets, L2-normalised: shared words mean high similarity."""
    name, dim = "test/hash-embedder", EMBED_DIM

    def __init__(self):
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self.calls += len(texts)
        return np.stack([self._vec(t) for t in texts])


class ScriptedModel:
    """Answers with the first sentence of the best chunk, quoted; `reply` overrides the next draft."""
    name, prompt_version = "test/scripted", "answer_v1"

    def __init__(self):
        self.calls: list[tuple[str, list[Hit], str]] = []
        self.reply: Draft | None = None

    def draft(self, question: str, hits: list[Hit], service: str) -> Draft:
        self.calls.append((question, hits, service))
        if self.reply is not None:
            return self.reply
        first = re.split(r"(?<=[.।])\s", hits[0].text)[0]
        return Draft(answer=f"Answer: {first}", citations=[(hits[0].chunk_id, first)], refused=False)


def migrate(url: str, revision: str = "head") -> None:
    cfg = Config(str(Path(ROOT) / "alembic.ini"), attributes={"configure_logger": False})
    cfg.set_main_option("sqlalchemy.url", url)
    if revision == "base":
        command.downgrade(cfg, "base")
    else:
        command.upgrade(cfg, revision)


@pytest.fixture(scope="session")
def db_url() -> str:
    url = os.environ.get(TEST_DB_ENV)
    if not url:
        pytest.skip(f"{TEST_DB_ENV} is not set")
    engine = make_engine(url)
    with engine.begin() as conn:   # start from an empty schema, whatever an earlier run left
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    engine.dispose()
    migrate(url)
    return url


@pytest.fixture
def engine(db_url) -> Iterator[Engine]:
    eng = make_engine(db_url)
    with eng.begin() as conn:
        conn.execute(text("TRUNCATE route_requests, kb_chunks RESTART IDENTITY"))
    yield eng
    eng.dispose()


@pytest.fixture
def stub() -> StubRouter:
    return StubRouter()


@pytest.fixture
def settings(db_url) -> Settings:
    return Settings(_env_file=None, database_url=db_url)


@pytest.fixture
def client(settings, stub, engine) -> Iterator[TestClient]:
    with TestClient(create_app(settings, router=stub)) as c:
        yield c


def rows(engine: Engine, sql: str = "SELECT * FROM route_requests ORDER BY id") -> list[dict]:
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(sql))]
