"""Typed generation configuration (configs/generation/*.yaml), with environment overrides for the model calls."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from . import CONFIGS

ProviderName = Literal["anthropic", "dry_run"]

FAMILIES = ("single", "contrast", "negation", "corrected_statement", "irrelevant_background", "missing_details",
            "out_of_scope", "vague", "multi_issue:no_priority", "multi_issue:stated_priority")


def _distribution(d: dict[str, float], name: str) -> dict[str, float]:
    if not d or abs(sum(d.values()) - 1) > 1e-6 or min(d.values()) < 0:
        raise ValueError(f"{name} must be a probability distribution, got {d}")
    return d


class Style(BaseModel):
    length: dict[str, float]
    tone: dict[str, float]
    formality: dict[str, float]

    @model_validator(mode="after")
    def _dists(self) -> "Style":
        for k in ("length", "tone", "formality"):
            _distribution(getattr(self, k), f"style.{k}")
        return self


class ModelCall(BaseModel):
    provider: ProviderName
    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    max_tokens: int = 16000
    concurrency: int = Field(1, ge=1, le=32)
    max_retries: int = Field(5, ge=0)
    prompt: str


class Generator(ModelCall):
    scenarios_per_request: int = Field(5, ge=1, le=20)


class Verifier(ModelCall):
    messages_per_request: int = Field(20, ge=1, le=60)


class Price(BaseModel):
    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


class Estimate(BaseModel):
    system_tokens: int
    user_tokens_per_scenario: int
    output_tokens_per_scenario: int
    thinking_tokens_per_request: int
    verify_user_tokens_per_message: int
    verify_output_tokens_per_message: int


class RealStyle(BaseModel):
    """Real-review-style rendering of already accepted scenarios (nepkev realstyle-generate)."""
    deva_share: float = Field(ge=0, le=1)     # share written in Devanagari, the rest Romanized
    style_examples: str                       # JSON list of example reviews, relative to the repository root


class GenConfig(BaseModel):
    name: str                                 # prefix of every scenario id
    taxonomy_version: str
    seed: int
    scenarios_per_domain: int = Field(ge=1)
    family_mix: dict[str, float]
    style: Style
    roman_share: float = Field(ge=0, le=1)   # share of accepted Devanagari records that also get a Romanized variant
    generator: Generator
    verifier: Verifier
    adjudicator: Verifier | None = None     # second opinion on messages the verifier disputes
    realstyle: RealStyle | None = None
    prices: dict[str, Price] = {}
    estimate: Estimate

    @model_validator(mode="after")
    def _families(self) -> "GenConfig":
        if bad := set(self.family_mix) - set(FAMILIES):
            raise ValueError(f"unknown families {sorted(bad)}; known: {FAMILIES}")
        _distribution(self.family_mix, "family_mix")
        return self


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _raw(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists() and not p.suffix:
        p = CONFIGS / "generation" / f"{p.name}.yaml"
    with p.open(encoding="utf-8") as f:
        d = yaml.safe_load(f)
    # `extends: <config>` starts from another config and overrides only the keys given here
    return _merge(_raw(d.pop("extends")), d) if "extends" in d else d


def load_genconfig(path: str | Path) -> GenConfig:
    cfg = GenConfig.model_validate(_raw(path))
    # swap a model without editing the checked-in config
    for role, prefix in (("generator", "NEPKEV_GEN"), ("verifier", "NEPKEV_VERIFY")):
        call = getattr(cfg, role)
        if v := os.environ.get(f"{prefix}_PROVIDER"):
            call.provider = v
        if v := os.environ.get(f"{prefix}_MODEL"):
            call.model = v
    return cfg
