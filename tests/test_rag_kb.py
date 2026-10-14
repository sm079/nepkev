import json
from collections import Counter
from datetime import date
from types import SimpleNamespace

import pytest

from nepkev.rag.config import load_rag_config, write_min_similarity
from nepkev.rag.evalset import (auc, best_threshold, bootstrap, first_gold_rank, load_eval_set, mrr, percentile,
                                recall_at)
from nepkev.rag.kb import Doc, chunk_doc, kb_chunks, load_kb, normalize, parse_doc

DOC = """---
id: wallet-test-en
domain: wallet
lang: en
title: A test article
updated_at: 2026-10-01
---

First paragraph about top-ups.

Limits are:

- one
- two

Last paragraph.
"""


# ---------------------------------------------------------------- knowledge base

def test_every_topic_has_an_english_and_a_nepali_article():
    docs = load_kb()
    assert 40 <= len(docs) <= 80
    topics = Counter((d.id.rsplit("-", 1)[0], d.domain) for d in docs)
    assert set(topics.values()) == {2}
    for d in docs:
        assert d.id.startswith(d.domain) and d.id.endswith(f"-{d.lang}")


def test_chunk_ids_are_unique_and_derived_from_the_text():
    chunks = kb_chunks()
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.id.startswith(f"{c.doc_id}#") and len(c.text) <= 400 for c in chunks)


def test_parse_doc_reads_front_matter():
    d = parse_doc(DOC)
    assert (d.id, d.domain, d.lang, d.title, d.updated_at) == ("wallet-test-en", "wallet", "en", "A test article",
                                                               date(2026, 10, 1))
    assert d.body.startswith("First paragraph")


@pytest.mark.parametrize("bad", ["no front matter", DOC.replace("domain: wallet", "domain: casino"),
                                 DOC.replace("title: A test article\n", "")])
def test_parse_doc_rejects_bad_articles(bad):
    with pytest.raises(ValueError):
        parse_doc(bad)


def test_a_list_stays_with_its_lead_in_and_edits_change_only_their_chunk():
    doc = parse_doc(DOC)
    one = chunk_doc(doc, max_chars=30)
    assert [c.text for c in one] == ["First paragraph about top-ups.", "Limits are: - one - two", "Last paragraph."]
    edited = chunk_doc(Doc(**{**doc.__dict__, "body": doc.body.replace("Last paragraph.", "Last paragraph, edited.")}),
                       max_chars=30)
    assert [c.id for c in one][:2] == [c.id for c in edited][:2] and one[2].id != edited[2].id
    assert chunk_doc(doc, max_chars=30)[0].id == one[0].id   # deterministic
    retitled = chunk_doc(Doc(**{**doc.__dict__, "title": "Another title"}), max_chars=30)
    assert not {c.id for c in one} & {c.id for c in retitled}   # the title is embedded with the text


def test_normalize_collapses_whitespace_and_composes_unicode():
    assert normalize("  a\n\tb  ") == "a b"
    assert normalize("é") == "é"   # composed, so a quote matches however its accents were typed


# ---------------------------------------------------------------- evaluation set

def test_eval_set_points_at_articles_of_its_own_domain():
    rows, docs = load_eval_set(), {d.id: d for d in load_kb()}
    answerable = [r for r in rows if not r.must_refuse]
    assert len(answerable) >= 100 and sum(r.must_refuse for r in rows) >= 20
    assert {r.script for r in rows} == {"en", "deva", "roman"}
    for r in answerable:
        assert r.gold_doc_ids and all(docs[g].domain == r.domain for g in r.gold_doc_ids)
    assert len({r.id for r in rows}) == len(rows)


def test_split_is_fixed_and_about_a_third_dev():
    rows = load_eval_set()
    dev = [r.id for r in rows if r.split == "dev"]
    assert dev == [r.id for r in load_eval_set() if r.split == "dev"]
    assert 0.2 < len(dev) / len(rows) < 0.45


def test_retrieval_metrics():
    assert first_gold_rank(["a", "b", "c"], ["c", "x"]) == 3 and first_gold_rank(["a"], ["b"]) is None
    ranks = [1, 2, None, 6]
    assert recall_at(ranks, 1) == 0.25 and recall_at(ranks, 5) == 0.5
    assert mrr(ranks) == pytest.approx((1 + 0.5 + 1 / 6) / 4)
    assert auc([0.9, 0.8], [0.1, 0.8]) == pytest.approx(0.875)
    assert percentile([1, 2, 3, 4], 0.5) == 2.5 and percentile([5], 0.95) == 5


def test_best_threshold_keeps_every_answerable_question_and_sits_in_the_widest_gap():
    assert best_threshold([0.80, 0.85, 0.90], [0.60, 0.70]) == pytest.approx(0.75)
    # overlapping scores: no cut-off may drop the 0.7 answerable question, so only 0.6 is refused
    assert best_threshold([0.7, 0.8, 0.9], [0.75, 0.6]) == pytest.approx(0.65)
    # nothing can be refused without losing an answerable question: the cut-off sits below every score
    assert best_threshold([0.5, 0.9], [0.7]) < 0.5
    # a looser target trades answerable questions for refusals
    assert best_threshold([0.7, 0.8, 0.9], [0.75, 0.6], min_kept=0.6) == pytest.approx(0.775)


def test_bootstrap_resamples_whole_units():
    items = [("a", 1.0), ("a", 1.0), ("b", 0.0)]
    ci = bootstrap(items, unit=lambda x: x[0], metrics={"mean": lambda s: sum(v for _, v in s) / len(s)}, samples=200)
    lo, hi = ci["mean"]
    assert 0.0 <= lo <= 2 / 3 <= hi <= 1.0
    assert bootstrap(items, lambda x: x[0], {"mean": lambda s: 0.5}, samples=10) == {"mean": [0.5, 0.5]}


# ---------------------------------------------------------------- config and prompt

def test_threshold_is_written_without_touching_the_rest(tmp_path):
    path = tmp_path / "rag.yaml"
    path.write_text("# comment\nembedding_model: m\ntop_k: 5\nmin_similarity: null\nprompt: answer_v1\n", encoding="utf-8")
    write_min_similarity(0.812345, path)
    assert path.read_text(encoding="utf-8").startswith("# comment\n")
    assert load_rag_config(path).min_similarity == 0.812345


def test_answer_model_sends_excerpts_and_parses_structured_output():
    pytest.importorskip("pgvector")
    from nepkev.rag.llm import AnthropicAnswerModel, user_message
    from nepkev.rag.store import Hit

    hits = [Hit("w-en#1", "w-en", "wallet", "en", "Top-ups", "Reversed within 24 hours.", 0.9)]
    payload = {"answer": "Within 24 hours.", "citations": [{"chunk_id": "w-en#1", "quote": "Reversed within 24 hours."}],
               "refused": False}
    sent = {}

    def create(**kw):
        sent.update(kw)
        return SimpleNamespace(stop_reason="end_turn", model=kw["model"], usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                               content=[SimpleNamespace(type="text", text=json.dumps(payload))])

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    model = AnthropicAnswerModel("claude-opus-5-5", client=client)
    draft = model.draft("top-up failed?", hits, "digital wallet")
    assert draft.answer == "Within 24 hours." and draft.citations == [("w-en#1", "Reversed within 24 hours.")]
    assert not draft.refused and draft.usage["input_tokens"] == 10
    assert "digital wallet" in sent["system"] and "{service}" not in sent["system"]
    assert sent["output_config"]["format"]["type"] == "json_schema" and sent["fallbacks"] == "default"
    msg = sent["messages"][0]["content"]
    assert msg == user_message("top-up failed?", hits) and '<excerpt id="w-en#1">' in msg

    def refusing(**kw):
        return SimpleNamespace(stop_reason="refusal", model=kw["model"], content=[],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=0))
    client.beta.messages.create = refusing
    assert model.draft("q", hits, "bank").refused

    def truncated(**kw):
        return SimpleNamespace(stop_reason="max_tokens", model=kw["model"], content=[],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=8000))
    client.beta.messages.create = truncated
    with pytest.raises(RuntimeError, match="max_tokens"):
        model.draft("q", hits, "bank")


@pytest.mark.slow
def test_real_embedder_matches_a_nepali_question_to_its_article():
    """Downloads intfloat/multilingual-e5-small (~0.5 GB): uv run pytest -m slow"""
    pytest.importorskip("transformers", reason="needs uv sync --extra rag")
    from nepkev.rag.embed import HFEmbedder

    emb = HFEmbedder(load_rag_config().embedding_model).load()
    chunks = [c for c in kb_chunks() if c.domain == "bank"]
    p = emb.embed_passages([c.embed_text for c in chunks])
    q = emb.embed_queries(["मेरो डेबिट कार्ड हरायो, कसरी ब्लक गर्ने?", "atm le paisa diyena tara khata bata katyo"])
    assert p.shape == (len(chunks), emb.dim) and abs(float((p[0] ** 2).sum()) - 1) < 1e-4
    best = [chunks[int(i)].doc_id for i in (q @ p.T).argmax(1)]
    assert best[0].startswith("bank-card-lost-block") and best[1].startswith("bank-atm-cash-not-dispensed")
