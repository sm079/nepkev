import itertools
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert

from nepkev.service import app as app_module
from nepkev.service.app import create_app
from nepkev.service.db import RouteRequest

from .conftest import UNREACHABLE_DB, StubRouter

_ids = itertools.count()


def percentile_cont(xs: list[float], q: float) -> float:
    """Postgres percentile_cont: linear interpolation between the closest ranks."""
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def add(engine, intent, latency, channel="wallet", age=timedelta(minutes=5), clarify=False):
    with engine.begin() as conn:
        conn.execute(insert(RouteRequest).values(
            request_id=f"{next(_ids):032d}", created_at=datetime.now(timezone.utc) - age, text_hash="0" * 64,
            script="romanized", channel=channel, intent=intent, confidence=0.9, needs_clarification=clarify,
            latency_ms=latency, model_version="stub", response={}))


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)


def test_stats_counts_and_latency_percentiles_over_the_window(client, engine):
    recent = [120.0, 80.5, 300.0, 95.0, 101.0, 2400.0, 99.9]
    for i, ms in enumerate(recent):
        add(engine, "login_otp" if i < 4 else "card_issue", ms, channel="wallet" if i < 4 else "bank", clarify=i == 0)
    add(engine, "login_otp", 9999.0, age=timedelta(hours=25))   # outside the default 24 h window

    s = client.get("/stats").json()
    assert s["window_hours"] == 24 and s["requests"] == len(recent)
    assert s["latency_ms"]["p50"] == pytest.approx(percentile_cont(recent, 0.5))
    assert s["latency_ms"]["p95"] == pytest.approx(percentile_cont(recent, 0.95))
    assert s["clarification_rate"] == pytest.approx(1 / len(recent))
    assert s["intents"] == [{"channel": "wallet", "intent": "login_otp", "count": 4},
                            {"channel": "bank", "intent": "card_issue", "count": 3}]

    wider = client.get("/stats", params={"hours": 48}).json()
    assert wider["requests"] == len(recent) + 1 and wider["intents"][0]["count"] == 5
    assert wider["latency_ms"]["p95"] == pytest.approx(percentile_cont([*recent, 9999.0], 0.95))


def test_stats_on_an_empty_window(client):
    s = client.get("/stats").json()
    assert s == {"window_hours": 24, "requests": 0, "clarification_rate": None,
                 "latency_ms": {"p50": None, "p95": None}, "intents": []}


@pytest.mark.parametrize("hours", [0, 721, "x"])
def test_stats_window_is_validated(client, hours):
    assert client.get("/stats", params={"hours": hours}).status_code == 422


def test_stats_counts_routed_requests(client):
    for text in ("a", "b", "c"):
        client.post("/route", json={"text": text, "channel": "bank"})
    s = client.get("/stats").json()
    assert s["requests"] == 3 and s["intents"] == [{"channel": "bank", "intent": "account_access", "count": 3}]


def test_ready_when_model_and_database_are_up(client):
    r = client.get("/ready")
    assert r.status_code == 200 and r.json() == {"ready": True, "model": "ready", "database": "ok", "answer": "disabled"}


def test_not_ready_when_the_database_is_down(settings):
    with TestClient(create_app(settings.model_copy(update={"database_url": UNREACHABLE_DB}), router=StubRouter())) as c:
        r = c.get("/ready")
        assert c.get("/health").status_code == 200   # still alive
    assert r.status_code == 503 and r.json() == {"ready": False, "model": "ready", "database": "unreachable", "answer": "disabled"}


def no_answerer(_settings):
    raise app_module.Unavailable("off in this test")


def test_model_loads_in_the_background_and_ready_follows_it(settings, engine, monkeypatch):
    release, stub = threading.Event(), StubRouter()

    def slow_load(_settings):
        release.wait(10)
        return stub

    monkeypatch.setattr(app_module, "kev_router", slow_load)
    monkeypatch.setattr(app_module, "rag_answerer", no_answerer)
    with TestClient(create_app(settings)) as c:
        loading = c.get("/ready")
        busy = c.post("/route", json={"text": "hello"})
        release.set()
        wait_for(lambda: c.app.state.model.status == "ready")
        done = c.get("/ready")
        routed = c.post("/route", json={"text": "hello"})
    assert loading.status_code == 503 and loading.json()["model"] == "loading"
    assert busy.status_code == 503 and busy.headers["Retry-After"] == "30"
    assert done.status_code == 200 and routed.status_code == 200
    assert stub.calls == [("hello", settings.default_channel)]


def test_failed_model_load_is_reported(settings, engine, monkeypatch):
    def broken(_settings):
        raise OSError("weights not found")

    monkeypatch.setattr(app_module, "kev_router", broken)
    monkeypatch.setattr(app_module, "rag_answerer", no_answerer)
    with TestClient(create_app(settings)) as c:
        wait_for(lambda: c.app.state.model.status != "loading")
        r = c.get("/ready")
        routed = c.post("/route", json={"text": "hello"})
    assert r.status_code == 503 and r.json()["model"] == "failed"
    assert routed.status_code == 503 and routed.json()["detail"] == "model failed: OSError: weights not found"
