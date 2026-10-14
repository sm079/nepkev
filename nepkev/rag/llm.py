"""Answer generation behind a small interface: the model gets the question and the retrieved excerpts and returns an
answer with citations (excerpt id + verbatim quote) or a refusal, as JSON constrained by a schema.

The system prompt is a versioned file, configs/prompts/<version>.md; its version is reported with every answer.
Provider: Anthropic (NEPKEV_LLM_PROVIDER=anthropic, key from ANTHROPIC_API_KEY). Tests use a fake with the same
interface. Validation of what the model returns (do the quotes exist?) happens in nepkev.rag.answer, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from .. import PROMPTS
from .store import Hit

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {
            "type": "object",
            "properties": {"chunk_id": {"type": "string"}, "quote": {"type": "string"}},
            "required": ["chunk_id", "quote"], "additionalProperties": False}},
        "refused": {"type": "boolean"},
    },
    "required": ["answer", "citations", "refused"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Draft:
    """What the model returned, before validation."""
    answer: str
    citations: list[tuple[str, str]]    # (chunk id, quote)
    refused: bool
    usage: dict = field(default_factory=dict)


class AnswerModel(Protocol):
    name: str
    prompt_version: str

    def draft(self, question: str, hits: list[Hit], service: str) -> Draft: ...


def load_prompt(version: str) -> str:
    return (PROMPTS / f"{version}.md").read_text(encoding="utf-8")


def user_message(question: str, hits: list[Hit]) -> str:
    excerpts = "\n".join(f'<excerpt id="{h.chunk_id}">\n{h.title}\n\n{h.text}\n</excerpt>' for h in hits)
    return f"<excerpts>\n{excerpts}\n</excerpts>\n\n<question>\n{question}\n</question>"


def parse_draft(payload: str, usage: dict | None = None) -> Draft:
    data = json.loads(payload)
    return Draft(answer=data["answer"].strip(), refused=bool(data["refused"]), usage=usage or {},
                 citations=[(c["chunk_id"], c["quote"]) for c in data["citations"]])


class AnthropicAnswerModel:
    """Claude with structured output. Server-side fallbacks are on, so a safety decline is retried on the model
    Anthropic recommends for its category instead of failing the request; a final decline counts as a refusal."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", prompt_version: str = "answer_v1",
                 client=None, timeout: float = 60.0):
        if client is None:
            import anthropic
            client = anthropic.Anthropic(timeout=timeout, max_retries=2)
        self.client, self.name, self.effort = client, model, effort
        self.prompt_version, self._system = prompt_version, load_prompt(prompt_version)

    def draft(self, question: str, hits: list[Hit], service: str) -> Draft:
        resp = self.client.beta.messages.create(
            model=self.name, max_tokens=8000, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            system=self._system.format(service=service),
            messages=[{"role": "user", "content": user_message(question, hits)}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
        )
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens, "model": resp.model}
        if resp.stop_reason == "refusal":
            return Draft(answer="", citations=[], refused=True, usage=usage)
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("answer model hit max_tokens before finishing its JSON")
        return parse_draft(next(b.text for b in resp.content if b.type == "text"), usage)
