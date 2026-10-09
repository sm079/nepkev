"""Structured-output providers.

- `anthropic`: the official Anthropic SDK, JSON-schema structured outputs, a cached system prompt, server-side refusal
  fallback on models that support it, and optional Message Batches (half price, no fallback).
- `dry_run`: deterministic, network-free renderings for development and CI. Its output is a mock and is labelled as
  such in every record it produces.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Protocol

from ..config import ModelCall

FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class RefusalError(RuntimeError):
    """The provider declined the request; retrying the same request will not help."""


class OutputError(RuntimeError):
    """Truncated or unparseable output; worth one retry."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, o: "Usage") -> "Usage":
        return Usage(self.input_tokens + o.input_tokens, self.output_tokens + o.output_tokens,
                     self.cache_read_tokens + o.cache_read_tokens, self.cache_write_tokens + o.cache_write_tokens)


@dataclass
class Result:
    data: dict
    usage: Usage
    model: str                      # the model that actually answered (a fallback may differ from the request)
    request_id: str | None = None
    extra: dict = field(default_factory=dict)


class Provider(Protocol):
    name: str
    model: str

    async def complete_json(self, system: str, user: str, schema: dict) -> Result: ...


# ---------------------------------------------------------------- Anthropic

class AnthropicProvider:
    name = "anthropic"

    def __init__(self, call: ModelCall):
        import anthropic   # imported lazily so dry runs and CI need no API package or key

        self.call, self.model = call, call.model
        self.client = anthropic.AsyncAnthropic(max_retries=call.max_retries)
        # batch calls are synchronous; one long-lived client, because a client that is garbage-collected while
        # its results stream is still being read closes the socket under the stream
        self.sync_client = anthropic.Anthropic(max_retries=call.max_retries)

    def params(self, system: str, user: str, schema: dict) -> dict:
        output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
        if self.call.effort:
            output_config["effort"] = self.call.effort
        return {
            "model": self.model,
            "max_tokens": self.call.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }

    @staticmethod
    def parse(message) -> dict:
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise RefusalError(f"refused: {getattr(details, 'category', None)}")
        if message.stop_reason == "max_tokens":
            raise OutputError("hit max_tokens before the JSON was complete")
        text = next((b.text for b in message.content if b.type == "text"), None)
        if text is None:
            raise OutputError("no text block in the response")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise OutputError(f"invalid JSON: {e}") from e

    @staticmethod
    def usage(message) -> Usage:
        u = message.usage
        return Usage(u.input_tokens, u.output_tokens, getattr(u, "cache_read_input_tokens", 0) or 0,
                     getattr(u, "cache_creation_input_tokens", 0) or 0)

    async def complete_json(self, system: str, user: str, schema: dict) -> Result:
        params = self.params(system, user, schema)
        if self.model in FALLBACK_MODELS:
            # on a safety decline the API re-runs the request on a suitable fallback model in the same call
            message = await self.client.beta.messages.create(**params, betas=[FALLBACK_BETA],
                                                             extra_body={"fallbacks": "default"})
        else:
            message = await self.client.messages.create(**params)
        return Result(self.parse(message), self.usage(message), message.model, getattr(message, "_request_id", None))

    # -------- Message Batches: asynchronous, half price, no server-side fallback

    def submit_batch(self, requests: list[tuple[str, str, str, dict]]) -> str:
        """requests: [(custom_id, system, user, schema)] -> batch id."""
        batch = self.sync_client.messages.batches.create(requests=[
            {"custom_id": cid, "params": self.params(system, user, schema)} for cid, system, user, schema in requests])
        return batch.id

    def batch_status(self, batch_id: str) -> tuple[str, dict]:
        b = self.sync_client.messages.batches.retrieve(batch_id)
        return b.processing_status, b.request_counts.model_dump() if hasattr(b.request_counts, "model_dump") else dict(b.request_counts)

    def batch_results(self, batch_id: str):
        """Yields (custom_id, Result | Exception). Results arrive in any order."""
        for r in self.sync_client.messages.batches.results(batch_id):
            if r.result.type != "succeeded":
                yield r.custom_id, RuntimeError(f"batch item {r.result.type}")
                continue
            msg = r.result.message
            try:
                yield r.custom_id, Result(self.parse(msg), self.usage(msg), msg.model)
            except (RefusalError, OutputError) as e:
                yield r.custom_id, e


# ---------------------------------------------------------------- dry run

class DryRunProvider:
    """Deterministic mock renderings built from the request itself. Never a benchmark input."""

    name = "dry_run"

    def __init__(self, call: ModelCall):
        self.call, self.model = call, "dry-run"

    async def complete_json(self, system: str, user: str, schema: dict) -> Result:
        payload = json.loads(user[user.index("["):])
        if "message" in schema["properties"]["items"]["items"]["properties"]:
            return Result({"items": [mock_rendering(s) for s in payload]}, Usage(), self.model)
        return Result({"items": [{"id": m["id"], "issue": None, "clarification_needed": False, "issues_mentioned": [],
                                  "language_ok": True, "_mock": True} for m in payload]}, Usage(), self.model)


_WORDS = ["हिजो", "आज", "बिहान", "बेलुका", "घर", "अफिस", "बजार", "साथी", "भाइ", "दिदी", "फोन", "पसल", "बाटो", "काम",
          "पानी", "खाना", "गाडी", "बस", "स्कुल", "किताब", "चिया", "दाल", "भात", "मान्छे", "सहर", "गाउँ", "दिन", "रात"]
_FILL = ["हेरिदिनुस् न", "छिटो मिलाइदिनु", "के गर्ने अब?", "प्लिज हेर्नुस्", "धेरै दिन भयो", "रिस उठ्यो"]


def mock_rendering(spec: dict) -> dict:
    """A Devanagari placeholder that carries the situation text's key facts. Mock data, never training material."""
    rng = random.Random(spec["scenario_id"])
    facts = spec["visible_facts"]
    amt = facts.get("amount_npr")
    oid = facts.get("order_id")
    situation = spec["situation"]
    extra = " ".join(x for x in [f"रु {amt}" if amt else "", f"अर्डर {oid}" if oid else ""] if x)
    if situation.startswith("The customer only says"):
        msg = "मेरो समस्या भयो, सहयोग गर्नुस् न"
    else:
        filler = " ".join(rng.sample(_WORDS, 5))     # keeps mock messages distinct for the duplicate checks
        msg = f"मेरो समस्या भयो {extra} {filler}".replace("  ", " ")
    if "negation" in situation:
        msg += ", अरू केही समस्या छैन"
    msg = f"{msg}। {_FILL[rng.randrange(len(_FILL))]} #{rng.randint(100, 999)}"
    return {"scenario_id": spec["scenario_id"], "message": msg, "evidence": "dry-run mock rendering"}


def make_provider(call: ModelCall) -> Provider:
    return {"anthropic": AnthropicProvider, "dry_run": DryRunProvider}[call.provider](call)
