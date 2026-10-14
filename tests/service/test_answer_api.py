import pytest
from fastapi.testclient import TestClient

from nepkev.rag import ingest as ingest_module
from nepkev.rag.answer import REFUSAL_EN, Answerer
from nepkev.rag.kb import kb_chunks
from nepkev.rag.store import sync
from nepkev.service import app as app_module
from nepkev.service.app import Unavailable, create_app, rag_answerer
from nepkev.service.db import make_sessionmaker

from .conftest import HashEmbedder, ScriptedModel

QUESTION = "How long do card refunds take after a return is approved?"


@pytest.fixture
def model():
    return ScriptedModel()


@pytest.fixture
def rag_client(settings, stub, engine, model):
    emb = HashEmbedder()
    with make_sessionmaker(engine)() as db:
        sync(db, kb_chunks(), emb)
    with TestClient(create_app(settings, router=stub, answerer=Answerer(emb, model, 5, min_similarity=0.2))) as c:
        yield c


def test_answer_returns_cited_text_from_the_channel(rag_client, stub, model):
    r = rag_client.post("/answer", json={"text": QUESTION, "channel": "shop"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False and body["refusal_reason"] is None and body["channel"] == "shop"
    assert body["intent"] == "order_status"   # the stub routes to the domain's first category
    [c] = body["citations"]
    assert set(c) == {"chunk_id", "doc_id", "quote"} and c["chunk_id"].startswith(f"{c['doc_id']}#")
    assert body["answer"] == f"Answer: {c['quote']}"
    scores = [s["score"] for s in body["retrieval_scores"]]
    assert len(scores) == 5 and scores == sorted(scores, reverse=True)
    assert all(s["doc_id"].startswith("shop-") for s in body["retrieval_scores"])
    assert set(body["latency_ms"]) == {"route", "retrieve", "generate", "total"}
    assert (body["prompt_version"], body["answer_model"]) == ("answer_v1", model.name)
    assert r.headers["X-Request-ID"] == body["request_id"] and stub.calls == [(QUESTION, "shop")]


def test_unrelated_question_is_refused_without_a_model_call(rag_client, model):
    body = rag_client.post("/answer", json={"text": "Bitcoin price prediction tomorrow?", "channel": "wallet"}).json()
    assert body["refused"] is True and body["refusal_reason"] == "low_similarity"
    assert body["answer"] == REFUSAL_EN and body["citations"] == [] and model.calls == []
    assert "generate" not in body["latency_ms"]


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "x" * 2001}, {"text": "hi", "lang": "en"},
                                     {"text": "hi", "channel": "casino"}, {}])
def test_answer_validates_its_body(rag_client, model, payload):
    assert rag_client.post("/answer", json=payload).status_code == 422 and model.calls == []


def test_answer_needs_a_json_body(rag_client):
    r = rag_client.post("/answer", content=b'{"text": "hi"}', headers={"Content-Type": "text/plain"})
    assert r.status_code == 415


def test_answer_is_503_without_an_answerer(client):
    r = client.post("/answer", json={"text": QUESTION})
    assert r.status_code == 503 and r.json()["detail"] == "answerer disabled: not configured"
    assert client.get("/ready").json()["answer"] == "disabled"   # /route still serves
    assert client.get("/ready").status_code == 200


def test_ready_reports_the_answerer(rag_client):
    assert rag_client.get("/ready").json() == {"ready": True, "model": "ready", "database": "ok", "answer": "ready"}


def test_answerer_is_disabled_until_configured(settings, monkeypatch):
    def reason(**changes):
        with pytest.raises(Unavailable) as e:
            rag_answerer(settings.model_copy(update=changes))
        return str(e.value)

    assert "RAG_ENABLED" in reason(rag_enabled=False)
    assert "LLM_PROVIDER=none" in reason(llm_provider="none")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "ANTHROPIC_API_KEY" in reason()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(app_module, "load_rag_config", lambda: _config(None))
    assert "threshold" in reason()


def test_configured_answerer_uses_the_rag_config(settings, monkeypatch):
    import nepkev.rag.embed

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(app_module, "load_rag_config", lambda: _config(0.8))
    monkeypatch.setattr(nepkev.rag.embed, "HFEmbedder", _Loadable)
    a = rag_answerer(settings.model_copy(update={"rag_top_k": 3}))
    assert (a.top_k, a.min_similarity, a.embedder.name) == (3, 0.8, "intfloat/multilingual-e5-small")
    assert (a.model.name, a.model.prompt_version) == (settings.llm_model, "answer_v1")
    assert rag_answerer(settings.model_copy(update={"rag_min_similarity": 0.5})).min_similarity == 0.5


def _config(min_similarity):
    from nepkev.rag.config import RagConfig

    return RagConfig(embedding_model="intfloat/multilingual-e5-small", top_k=5, min_similarity=min_similarity,
                     prompt="answer_v1")


def test_ingest_command_is_idempotent(settings, engine, monkeypatch, capsys):
    monkeypatch.setattr(ingest_module, "Settings", lambda: settings.model_copy(update={"rag_embedding_model": "test/hash"}))
    monkeypatch.setattr(ingest_module, "HFEmbedder", lambda name: _Loadable(name))
    n = len(kb_chunks())
    ingest_module.main([])
    assert f"{n} embedded, 0 deleted, 0 unchanged" in capsys.readouterr().out
    ingest_module.main(["--dry-run"])
    assert "0 to embed, 0 to delete" in capsys.readouterr().out
    ingest_module.main([])
    assert f"0 embedded, 0 deleted, {n} unchanged" in capsys.readouterr().out


class _Loadable(HashEmbedder):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def load(self):
        return self
