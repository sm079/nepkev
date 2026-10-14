from dataclasses import replace

import pytest
from sqlalchemy import select

from nepkev.rag.answer import REFUSAL_EN, REFUSAL_NE, REFUSAL_ROMAN, Answerer, question_language, valid_citations
from nepkev.rag.kb import kb_chunks
from nepkev.rag.llm import Draft
from nepkev.rag.store import EMBED_DIM, Hit, KbChunk, search, sync
from nepkev.service.db import make_sessionmaker

from .conftest import HashEmbedder, ScriptedModel


@pytest.fixture
def db(engine):
    with make_sessionmaker(engine)() as s:
        yield s


@pytest.fixture
def ingested(db):
    emb = HashEmbedder()
    sync(db, kb_chunks(), emb)
    return emb


def stored(db) -> dict[str, str]:
    return dict(db.execute(select(KbChunk.id, KbChunk.embedding_model)).all())


# ---------------------------------------------------------------- ingest

def test_sync_is_idempotent(db):
    chunks, emb = kb_chunks(), HashEmbedder()
    first = sync(db, chunks, emb)
    assert len(first.embedded) == len(chunks) and first.deleted == [] and emb.calls == len(chunks)
    second = sync(db, chunks, emb)
    assert (second.embedded, second.deleted, second.unchanged) == ([], [], len(chunks))
    assert emb.calls == len(chunks) and set(stored(db)) == {c.id for c in chunks}


def test_an_edited_chunk_is_re_embedded_and_its_old_version_removed(db, ingested):
    chunks = kb_chunks()
    edited = replace(chunks[0], id=chunks[0].id.split("#")[0] + "#edited", text=chunks[0].text + " Updated.")
    report = sync(db, [edited, *chunks[1:]], ingested)
    assert report.embedded == [edited.id] and report.deleted == [chunks[0].id]
    assert len(stored(db)) == len(chunks)


def test_removed_articles_are_deleted_and_a_new_embedder_re_embeds_everything(db, ingested):
    chunks = kb_chunks()
    kept = [c for c in chunks if c.domain != "shop"]
    assert len(sync(db, kept, ingested).deleted) == len(chunks) - len(kept)

    other = HashEmbedder()
    other.name = "test/another-embedder"
    report = sync(db, kept, other)
    assert len(report.embedded) == len(kept) and set(stored(db).values()) == {other.name}


def test_metadata_changes_are_saved_without_re_embedding(db, ingested):
    chunks = kb_chunks()
    moved = replace(chunks[0], updated_at=chunks[0].updated_at.replace(year=2027))
    report = sync(db, [moved, *chunks[1:]], ingested)
    assert report.embedded == []
    assert db.get(KbChunk, moved.id).updated_at.year == 2027


def test_sync_refuses_an_embedder_of_another_size(db):
    emb = HashEmbedder()
    emb.dim = EMBED_DIM * 2
    with pytest.raises(ValueError, match="kb_chunks stores"):
        sync(db, kb_chunks(), emb)


# ---------------------------------------------------------------- search

def test_search_stays_in_the_domain_and_ranks_by_similarity(db, ingested):
    chunk = next(c for c in kb_chunks() if c.domain == "bank")
    hits = search(db, ingested.embed_queries([chunk.embed_text])[0], "bank", 5)
    assert len(hits) == 5 and {h.domain for h in hits} == {"bank"}
    assert hits[0].chunk_id == chunk.id and hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert search(db, ingested.embed_queries([chunk.embed_text])[0], "shop", 3)[0].domain == "shop"


# ---------------------------------------------------------------- answerer

QUESTION = "How long do card refunds take after a return is approved?"


def answerer(emb, model=None, min_similarity=0.2):
    return Answerer(emb, model or ScriptedModel(), top_k=4, min_similarity=min_similarity)


def test_answer_cites_the_retrieved_chunk_verbatim(db, ingested):
    model = ScriptedModel()
    res = answerer(ingested, model).answer(db, QUESTION, "shop", "online shop")
    assert not res.refused and res.refusal_reason is None
    [c] = res.citations
    hit = next(h for h in res.hits if h.chunk_id == c.chunk_id)
    assert c.quote in hit.text and c.doc_id == hit.doc_id and hit.domain == "shop"
    assert len(res.hits) == 4 and model.calls[0][2] == "online shop"
    assert set(res.timings_ms) == {"retrieve", "generate"}


def test_low_similarity_refuses_without_calling_the_model(db, ingested):
    model = ScriptedModel()
    res = answerer(ingested, model, min_similarity=0.99).answer(db, "kati din lagcha?", "wallet", "digital wallet")
    assert res.refused and res.refusal_reason == "low_similarity" and model.calls == []
    assert res.answer == REFUSAL_ROMAN and res.citations == [] and res.hits


def test_a_declining_model_gives_the_fixed_refusal(db, ingested):
    model = ScriptedModel()
    model.reply = Draft(answer="Sure, here is a poem", citations=[], refused=True)
    res = answerer(ingested, model).answer(db, QUESTION, "shop", "online shop")
    assert res.refused and res.refusal_reason == "model_declined" and res.answer == REFUSAL_EN


def test_answers_without_a_verbatim_citation_are_refused(db, ingested):
    model = ScriptedModel()
    hits = answerer(ingested).retrieve(db, QUESTION, "shop")
    model.reply = Draft(answer="Ten days.", citations=[(hits[0].chunk_id, "a sentence that is not there"),
                                                       ("shop-other#000000000000", hits[0].text[:20])], refused=False)
    res = answerer(ingested, model).answer(db, QUESTION, "shop", "online shop")
    assert res.refused and res.refusal_reason == "no_valid_citation" and res.draft is model.reply


def test_only_valid_citations_are_kept():
    hits = [Hit("a#1", "a", "shop", "en", "T", "Refunds take 3 working days. Cards take longer.", 0.9),
            Hit("b#1", "b", "shop", "ne", "T", "रिफन्ड ३ कार्य दिनमा आउँछ।", 0.8)]
    draft = Draft("x", [("a#1", "Refunds take  3 working\ndays."), ("a#1", "Refunds take 3 working days."),
                        ("b#1", "रिफन्ड ३ कार्य दिनमा"), ("a#1", ""), ("a#1", "Cards are instant."), ("zzz", "Refunds")],
                  refused=False)
    kept = valid_citations(draft, hits)
    assert [(c.chunk_id, c.quote) for c in kept] == [("a#1", "Refunds take 3 working days."), ("b#1", "रिफन्ड ३ कार्य दिनमा")]


@pytest.mark.parametrize("text,lang,message", [
    ("How long does KYC take?", "en", REFUSAL_EN),
    ("kyc kati din ma huncha?", "roman", REFUSAL_ROMAN),
    ("केवाईसी कति दिनमा हुन्छ?", "deva", REFUSAL_NE),
])
def test_refusals_follow_the_question_language(text, lang, message):
    from nepkev.rag.answer import refusal_message

    assert question_language(text) == lang and refusal_message(text) == message
