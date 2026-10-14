import json
from dataclasses import replace

import pytest

from nepkev.rag import eval as rag_eval
from nepkev.rag.config import load_rag_config
from nepkev.rag.kb import kb_chunks
from nepkev.rag.store import sync
from nepkev.service.db import make_sessionmaker

from .conftest import HashEmbedder, ScriptedModel


class NamedEmbedder(HashEmbedder):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def load(self):
        return self


@pytest.fixture
def setup(settings, engine, tmp_path, monkeypatch):
    """The real eval set and KB, a hashing embedder stored under the configured model's name, outputs in tmp_path."""
    cfg = replace(load_rag_config(), min_similarity=0.1)
    with make_sessionmaker(engine)() as db:
        sync(db, kb_chunks(), NamedEmbedder(cfg.embedding_model))
    monkeypatch.setattr(rag_eval, "Settings", lambda: settings)
    monkeypatch.setattr(rag_eval, "load_rag_config", lambda: cfg)
    monkeypatch.setattr(rag_eval, "HFEmbedder", NamedEmbedder)
    monkeypatch.setattr(rag_eval, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(rag_eval, "load_dotenv", lambda: None)
    return tmp_path


def test_retrieval_report_and_quality_gate(setup):
    out, base = setup / "report.md", setup / "baseline.json"
    rag_eval.main(["--retrieval-only", "--out", str(out), "--write-baseline", str(base)])
    rep = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert rep["n"]["dev"] + rep["n"]["test"] == 126 and rep["threshold"] == 0.1
    assert 0 <= rep["retrieval"]["recall_at_5"] <= 1 and set(rep["retrieval_by_script"]) == {"en", "deva", "roman"}
    lo, hi = rep["gate_ci95"]["refusal_precision"]
    assert 0 <= lo <= hi <= 1
    assert "## Retrieval (test)" in out.read_text(encoding="utf-8") and "## Answers" not in out.read_text(encoding="utf-8")

    rag_eval.main(["--retrieval-only", "--check", str(base)])   # passes against its own numbers
    base.write_text(json.dumps({"recall_at_5": 1.01, "gate_refusal_recall": 0.0}), encoding="utf-8")
    with pytest.raises(SystemExit, match="quality gate failed: recall_at_5"):
        rag_eval.main(["--retrieval-only", "--check", str(base)])


def test_eval_refuses_a_stale_knowledge_base(setup, engine):
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM kb_chunks WHERE domain = 'bank'"))
    with pytest.raises(SystemExit, match="run python -m nepkev.rag.ingest"):
        rag_eval.main(["--retrieval-only", "--out", str(setup / "r.md")])


def test_answer_eval_caches_model_outputs(setup, monkeypatch):
    models: list[ScriptedModel] = []

    def make_model(*args, **kwargs):
        models.append(ScriptedModel())
        return models[-1]

    monkeypatch.setattr(rag_eval, "AnthropicAnswerModel", make_model)
    monkeypatch.setattr(rag_eval, "judge", lambda client, q, res: {"verdict": "supported", "unsupported_statements": []})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    out = setup / "report.md"

    with pytest.raises(SystemExit, match="--yes"):
        rag_eval.main(["--out", str(out)])   # nothing cached yet, and no permission to call the model

    rag_eval.main(["--yes", "--out", str(out)])
    first = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert first["new_model_calls"] == len(models[-1].calls) > 0   # models[0] never ran: no --yes
    answers = [json.loads(x) for x in (setup / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(answers) == first["new_model_calls"]
    a = first["answers"]
    assert a["citation_verbatim"] == 1.0 and a["judge_supported"] == 1.0 and a["answered"] > 0
    assert sum(a["refusal_reasons"].values()) + a["answered"] == first["n"]["test"]

    judged = {json.loads(x)["id"] for x in (setup / "judge.jsonl").read_text(encoding="utf-8").splitlines()}
    sample = sorted(judged)[:2]
    (setup / "judge_check.jsonl").write_text("".join(json.dumps({"id": i, "verdict": v}) + "\n"
                                                     for i, v in zip(sample, ["supported", "unsupported"])),
                                             encoding="utf-8")
    rag_eval.main(["--out", str(out)])   # everything cached: no --yes needed, no calls
    second = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert second["new_model_calls"] == 0 and models[-1].calls == []
    assert second["judge_check"] == {"n": 2, "agree": 1}
    assert second["answers"] == first["answers"]
    text = out.read_text(encoding="utf-8")
    assert "## Answers (test" in text and "Hand check: 1 of 2" in text
