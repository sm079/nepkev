"""FastAPI app: POST /route, GET /health, GET /ready, GET /stats.

    uvicorn nepkev.service.app:create_app --factory --host 0.0.0.0 --port 8000

The model loads once, in a background thread started by the lifespan, so the process answers /health (liveness) at
once and /ready (readiness: model loaded and database reachable) only when it can serve. /route answers 503 until then.
Every answered /route is logged to Postgres (route_requests); the raw text only when NEPKEV_LOG_TEXT is on.

Idempotency follows the IETF Idempotency-Key draft: a repeated key with the same body returns the stored response
(marked with an `Idempotent-Replayed: true` header) without running the model; the same key with a different body is
rejected with 422.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import Float, Integer, cast, func, literal, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from ..rag.answer import Answerer
from ..rag.config import load_rag_config
from ..rag.llm import AnthropicAnswerModel
from ..router import KevRouter, Router
from ..taxonomy import load_taxonomy
from .clarify import decide, script_of
from .db import RouteRequest, make_engine, make_sessionmaker, ping
from .schemas import (
    AnswerIn,
    AnswerOut,
    CitationOut,
    IntentCount,
    LatencyStats,
    ReadyOut,
    RouteIn,
    RouteOut,
    ScoreOut,
    StatsOut,
)
from .settings import Settings, get_settings

log = logging.getLogger("nepkev.service")

REPLAY_HEADER = "Idempotent-Replayed"


class Unavailable(Exception):
    """A component that is switched off or not configured (reported as `disabled`, not `failed`)."""


class Slot:
    """A component loaded in the background (the router, the answerer) and how loading went."""

    def __init__(self, name: str, value=None, status: str | None = None, detail: str | None = None):
        self.name, self.value, self.detail = name, value, detail
        self.status = status or ("ready" if value is not None else "loading")

    def load(self, factory: Callable[[], object]) -> None:
        started = time.perf_counter()
        try:
            value = factory()
        except Unavailable as e:
            self.status, self.detail = "disabled", str(e)
            log.warning("%s disabled: %s", self.name, e)
            return
        except Exception as e:
            self.status, self.detail = "failed", f"{type(e).__name__}: {e}"
            log.exception("%s failed to load", self.name)
            return
        self.value, self.status = value, "ready"
        log.info("%s ready after %.1f s", self.name, time.perf_counter() - started)

    def get(self):
        if self.value is None:
            raise HTTPException(503, f"{self.name} {self.status}" + (f": {self.detail}" if self.detail else ""),
                                headers={"Retry-After": "30"} if self.status == "loading" else None)
        return self.value


def kev_router(settings: Settings) -> Router:
    router = KevRouter(settings.model, revision=settings.model_revision, device=settings.device, dtype=settings.dtype,
                       threads=settings.torch_threads)
    router.load()
    return router


def rag_answerer(settings: Settings) -> Answerer:
    if not settings.rag_enabled:
        raise Unavailable("NEPKEV_RAG_ENABLED is off")
    if settings.llm_provider == "none":
        raise Unavailable("no answer model configured (NEPKEV_LLM_PROVIDER=none)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise Unavailable("ANTHROPIC_API_KEY is not set")
    cfg = load_rag_config()
    threshold = settings.rag_min_similarity if settings.rag_min_similarity is not None else cfg.min_similarity
    if threshold is None:
        raise Unavailable("no refusal threshold: run python -m nepkev.rag.eval --select-threshold")
    from ..rag.embed import HFEmbedder

    embedder = HFEmbedder(settings.rag_embedding_model or cfg.embedding_model).load()
    model = AnthropicAnswerModel(settings.llm_model, effort=settings.llm_effort, prompt_version=cfg.prompt)
    return Answerer(embedder, model, top_k=settings.rag_top_k or cfg.top_k, min_similarity=threshold)


def require_json(request: Request) -> None:
    """415 for a body sent as anything but JSON (curl -d sends application/x-www-form-urlencoded by default)."""
    kind = request.headers.get("content-type", "application/json").split(";")[0].strip().lower()
    if kind != "application/json" and not kind.endswith("+json"):
        raise HTTPException(415, "send the body as JSON with Content-Type: application/json")


def session(request: Request):
    with request.app.state.sessions() as s:
        yield s


DB = Annotated[Session, Depends(session)]


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def create_app(settings: Settings | None = None, router: Router | None = None,
               answerer: Answerer | None = None) -> FastAPI:
    """The app. With `router` (and optionally `answerer`) those serve at once (tests); otherwise the configured
    checkpoint, and then the answerer, load in a background thread."""
    settings = settings or get_settings()
    tax = load_taxonomy()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings.database_url)
        app.state.engine, app.state.sessions = engine, make_sessionmaker(engine)
        app.state.model = models = Slot("model", router)
        given = router is not None
        app.state.answerer = answers = Slot("answerer", answerer, "disabled" if given and answerer is None else None,
                                            "not configured" if given and answerer is None else None)

        def load_all() -> None:
            models.load(lambda: kev_router(settings))
            answers.load(lambda: rag_answerer(settings))

        loading = None if given else asyncio.create_task(asyncio.to_thread(load_all))
        yield
        if loading is not None and not loading.done():
            log.warning("shutting down while the model is still loading")
        engine.dispose()

    app = FastAPI(title="nepkev router", version="1.0", lifespan=lifespan,
                  description="Routes Nepali support messages (Devanagari or Romanized) for a demo shop, wallet and bank.")

    @app.exception_handler(OperationalError)
    async def database_down(request: Request, exc: OperationalError) -> JSONResponse:
        log.error("database error on %s: %s", request.url.path, exc.orig)
        return JSONResponse({"detail": "database unavailable"}, status_code=503)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadyOut, responses={503: {"model": ReadyOut}})
    def ready(request: Request):
        model = request.app.state.model.status
        try:
            ping(request.app.state.engine)
            database = "ok"
        except Exception as e:
            log.warning("readiness: database unreachable: %s", e)
            database = "unreachable"
        body = ReadyOut(ready=model == "ready" and database == "ok", model=model, database=database,
                        answer=request.app.state.answerer.status)
        return JSONResponse(body.model_dump(), status_code=200 if body.ready else 503)

    @app.post("/route", response_model=RouteOut, dependencies=[Depends(require_json)])
    def route(body: RouteIn, request: Request, response: Response, db: DB,
              idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", min_length=1, max_length=255)] = None):
        body_hash = sha256(json.dumps(body.model_dump(), ensure_ascii=False, sort_keys=True))
        if idempotency_key is not None and (stored := _stored(db, idempotency_key, body_hash)) is not None:
            return _replay(response, stored)

        router_: Router = request.app.state.model.get()
        channel = body.channel or settings.default_channel
        started = time.perf_counter()
        routing = router_.route(body.text, channel)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        decision = decide(routing, body.text, tax, settings.clarify_threshold, settings.min_confidence)

        out = RouteOut(request_id=uuid.uuid4().hex, channel=channel, intent=routing.intent,
                       confidence=round(routing.confidence, 4), needs_clarification=decision.needs_clarification,
                       clarification_reason=decision.reason, clarifying_question=decision.question,
                       model_version=router_.version, latency_ms=latency_ms)
        db.add(RouteRequest(request_id=out.request_id, text_hash=sha256(body.text), text=body.text if settings.log_text else None,
                            script=script_of(body.text), channel=channel, intent=out.intent, confidence=out.confidence,
                            needs_clarification=out.needs_clarification, latency_ms=latency_ms,
                            model_version=out.model_version, idempotency_key=idempotency_key,
                            request_hash=body_hash if idempotency_key else None, response=out.model_dump(mode="json")))
        try:
            db.commit()
        except IntegrityError:
            # a concurrent request with the same key committed first: answer as its replay
            db.rollback()
            if idempotency_key is None or (stored := _stored(db, idempotency_key, body_hash)) is None:
                raise
            return _replay(response, stored)
        response.headers["X-Request-ID"] = out.request_id
        return out

    @app.post("/answer", response_model=AnswerOut, dependencies=[Depends(require_json)])
    def answer(body: AnswerIn, request: Request, response: Response, db: DB):
        """Route the question, retrieve from the channel's articles, and answer only from them, with citations."""
        router_: Router = request.app.state.model.get()
        answerer_: Answerer = request.app.state.answerer.get()
        channel = body.channel or settings.default_channel
        t0 = time.perf_counter()
        routing = router_.route(body.text, channel)
        route_ms = round((time.perf_counter() - t0) * 1000, 1)
        result = answerer_.answer(db, body.text, channel, service=tax.domain(channel).name)
        out = AnswerOut(
            request_id=uuid.uuid4().hex, channel=channel, intent=routing.intent, answer=result.answer,
            citations=[CitationOut(chunk_id=c.chunk_id, doc_id=c.doc_id, quote=c.quote) for c in result.citations],
            refused=result.refused, refusal_reason=result.refusal_reason,
            retrieval_scores=[ScoreOut(chunk_id=h.chunk_id, doc_id=h.doc_id, score=round(h.score, 4))
                              for h in result.hits],
            prompt_version=answerer_.model.prompt_version, answer_model=answerer_.model.name,
            latency_ms={"route": route_ms, **result.timings_ms, "total": round((time.perf_counter() - t0) * 1000, 1)})
        response.headers["X-Request-ID"] = out.request_id
        return out

    @app.get("/stats", response_model=StatsOut)
    def stats(db: DB, hours: Annotated[int, Query(ge=1, le=24 * 30)] = 24) -> StatsOut:
        """Requests in the last `hours`: count per channel and intent, clarification rate, p50/p95 latency."""
        recent = RouteRequest.created_at >= func.now() - literal(timedelta(hours=hours))
        n, p50, p95, clar = db.execute(select(
            func.count(),
            func.percentile_cont(0.5).within_group(RouteRequest.latency_ms),
            func.percentile_cont(0.95).within_group(RouteRequest.latency_ms),
            func.avg(cast(cast(RouteRequest.needs_clarification, Integer), Float)),
        ).where(recent)).one()
        n_intent = func.count().label("n")
        rows = db.execute(select(RouteRequest.channel, RouteRequest.intent, n_intent).where(recent)
                          .group_by(RouteRequest.channel, RouteRequest.intent)
                          .order_by(n_intent.desc(), RouteRequest.channel, RouteRequest.intent)).all()
        return StatsOut(window_hours=hours, requests=n, clarification_rate=clar,
                        latency_ms=LatencyStats(p50=p50, p95=p95),
                        intents=[IntentCount(channel=c, intent=i, count=k) for c, i, k in rows])

    return app


def _replay(response: Response, stored: dict) -> dict:
    response.headers[REPLAY_HEADER] = "true"
    response.headers["X-Request-ID"] = stored["request_id"]
    return stored


def _stored(db: Session, key: str, body_hash: str) -> dict | None:
    """The response stored under an idempotency key; 422 if the key was used with a different body."""
    row = db.execute(select(RouteRequest.request_hash, RouteRequest.response)
                     .where(RouteRequest.idempotency_key == key)).one_or_none()
    if row is None:
        return None
    if row.request_hash != body_hash:
        raise HTTPException(422, "Idempotency-Key was already used with a different request body")
    return row.response
