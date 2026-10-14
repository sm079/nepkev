"""Request and response bodies."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .settings import Channel

MAX_TEXT_CHARS = 2000

Message = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS)]


class RouteIn(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"text": "OTP aaudaina, login garna sakdina", "channel": "wallet"}]})

    text: Message = Field(description="The customer's message, Devanagari or Romanized Nepali (1-2,000 characters)")
    channel: Channel | None = Field(None, description="Support inbox the message came from; the service default if omitted")


class RouteOut(BaseModel):
    request_id: str
    channel: Channel
    intent: str = Field(description="Category id from the channel's taxonomy (configs/taxonomy/demo-1.yaml)")
    confidence: float = Field(description="Calibrated probability of `intent`")
    needs_clarification: bool
    clarification_reason: Literal["unclear_message", "low_confidence"] | None = None
    clarifying_question: str | None = Field(None, description="A question to ask the customer, in their script")
    model_version: str
    latency_ms: float = Field(description="Model inference time")


class LatencyStats(BaseModel):
    p50: float | None
    p95: float | None


class IntentCount(BaseModel):
    channel: str
    intent: str
    count: int


class StatsOut(BaseModel):
    window_hours: int
    requests: int
    clarification_rate: float | None
    latency_ms: LatencyStats
    intents: list[IntentCount]


class ReadyOut(BaseModel):
    ready: bool = Field(description="Model loaded and database reachable: /route can serve")
    model: Literal["loading", "ready", "failed"]
    database: Literal["ok", "unreachable"]
    answer: Literal["loading", "ready", "failed", "disabled"] = Field(description="Whether /answer can serve")


class AnswerIn(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"text": "KYC submit gareko kati din ma verify huncha?", "channel": "wallet"}]})

    text: Message = Field(description="The customer's question (1-2,000 characters)")
    channel: Channel | None = Field(None, description="Support inbox; retrieval only reads this domain's articles")


class CitationOut(BaseModel):
    chunk_id: str
    doc_id: str
    quote: str = Field(description="Verbatim text of the cited chunk")


class ScoreOut(BaseModel):
    chunk_id: str
    doc_id: str
    score: float = Field(description="Cosine similarity to the question")


class AnswerOut(BaseModel):
    request_id: str
    channel: Channel
    intent: str
    answer: str = Field(description="The answer, or a fixed refusal message when refused")
    citations: list[CitationOut]
    refused: bool
    refusal_reason: Literal["low_similarity", "model_declined", "no_valid_citation"] | None
    retrieval_scores: list[ScoreOut]
    prompt_version: str
    answer_model: str
    latency_ms: dict[str, float] = Field(description="route, retrieve, generate (when the model ran) and total")
