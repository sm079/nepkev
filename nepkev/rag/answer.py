"""Retrieve, gate, generate, check: one cited answer or a refusal.

1. Embed the question and take the top-k chunks of the channel's domain from pgvector.
2. If the best chunk is less similar than min_similarity (chosen on the eval set's dev split), refuse without calling
   the model: the knowledge base has nothing close enough.
3. Otherwise ask the answer model, which may itself decline.
4. Keep only citations that name a retrieved chunk and quote it verbatim (after whitespace and Unicode normalisation).
   An answer left with no valid citation is refused: it is not grounded in anything we can show.

Refusals return a fixed message in the question's language, so a refusal never contains generated text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ..validate import detect_script, nepali_hindi_scores
from .embed import Embedder
from .kb import normalize
from .llm import AnswerModel, Draft
from .store import Hit, search

REFUSAL_EN = "Sorry, I can't answer that from our help articles. Please contact our support team."
REFUSAL_NE = "माफ गर्नुहोस्, हाम्रा सहायता लेखहरूमा यसको जवाफ छैन। कृपया हाम्रो सपोर्ट टोलीलाई सम्पर्क गर्नुहोस्।"
REFUSAL_ROMAN = "Maaf garnuhola, hamro help article haru ma yasko jawaf chhaina. Kripaya hamro support team lai sampark garnuhos."


def question_language(text: str) -> str:
    """'deva', 'roman' (Romanized Nepali) or 'en'. Latin text counts as Nepali when it has Nepali function words."""
    script = detect_script(text)
    if script != "roman":
        return "deva"
    return "roman" if nepali_hindi_scores(text, "roman")[0] > 0 else "en"


def refusal_message(question: str) -> str:
    lang = question_language(question)
    return {"en": REFUSAL_EN, "deva": REFUSAL_NE, "roman": REFUSAL_ROMAN}[lang]


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    doc_id: str
    quote: str


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    refused: bool
    refusal_reason: str | None        # low_similarity | model_declined | no_valid_citation
    hits: list[Hit]
    draft: Draft | None = None        # the model's raw output, kept for evaluation
    timings_ms: dict[str, float] = field(default_factory=dict)


def valid_citations(draft: Draft, hits: list[Hit]) -> list[Citation]:
    by_id = {h.chunk_id: h for h in hits}
    out: list[Citation] = []
    for chunk_id, quote in draft.citations:
        q, hit = normalize(quote), by_id.get(chunk_id)
        c = Citation(chunk_id, hit.doc_id, q) if hit is not None else None
        if c is not None and q and q in hit.text and c not in out:
            out.append(c)
    return out


class Answerer:
    def __init__(self, embedder: Embedder, model: AnswerModel, top_k: int, min_similarity: float):
        self.embedder, self.model, self.top_k, self.min_similarity = embedder, model, top_k, min_similarity

    def retrieve(self, db: Session, question: str, domain: str, k: int | None = None) -> list[Hit]:
        return search(db, self.embedder.embed_queries([question])[0], domain, k or self.top_k)

    def answer(self, db: Session, question: str, domain: str, service: str) -> AnswerResult:
        t0 = time.perf_counter()
        hits = self.retrieve(db, question, domain)
        return self.respond(question, hits, service, {"retrieve": round((time.perf_counter() - t0) * 1000, 1)})

    def respond(self, question: str, hits: list[Hit], service: str, timings: dict | None = None) -> AnswerResult:
        """Steps 2-4 on chunks already retrieved, best first."""
        timings = dict(timings or {})
        if not hits or hits[0].score < self.min_similarity:
            return self._refuse(question, "low_similarity", hits, None, timings)
        t1 = time.perf_counter()
        draft = self.model.draft(question, hits, service)
        timings["generate"] = round((time.perf_counter() - t1) * 1000, 1)
        if draft.refused:
            return self._refuse(question, "model_declined", hits, draft, timings)
        citations = valid_citations(draft, hits)
        if not citations or not draft.answer:
            return self._refuse(question, "no_valid_citation", hits, draft, timings)
        return AnswerResult(draft.answer, citations, False, None, hits, draft, timings)

    @staticmethod
    def _refuse(question: str, reason: str, hits: list[Hit], draft: Draft | None, timings: dict) -> AnswerResult:
        return AnswerResult(refusal_message(question), [], True, reason, hits, draft, timings)
