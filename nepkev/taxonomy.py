"""Versioned routing taxonomy: typed loading, integrity checks and the model-facing question text."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from . import CONFIGS

DEFAULT_VERSION = "demo-1"


class Category(BaseModel):
    description: str = Field(min_length=10)
    ne: str
    include: list[str] = []
    exclude: list[str] = []
    neighbors: list[str] = []
    facts: list[str] = []


class Domain(BaseModel):
    name: str
    ne: str
    categories: dict[str, Category]
    contrasts: list[tuple[str, str, str]] = []

    @model_validator(mode="after")
    def _refs_exist(self) -> "Domain":
        ids = set(self.categories)
        for cid, c in self.categories.items():
            if bad := set(c.neighbors) - ids:
                raise ValueError(f"{cid}: unknown neighbors {sorted(bad)}")
        for a, b, _ in self.contrasts:
            if a not in ids or b not in ids:
                raise ValueError(f"contrast {a}/{b}: unknown category")
        return self


class Clarification(BaseModel):
    instructions: list[str] = Field(min_length=1)
    criteria: dict[str, str]

    @model_validator(mode="after")
    def _true_false(self) -> "Clarification":
        if set(self.criteria) != {"true", "false"}:
            raise ValueError("clarification criteria must have exactly 'true' and 'false'")
        return self


class Issue(BaseModel):
    instructions: list[str] = Field(min_length=1)


class Questions(BaseModel):
    issue: Issue
    clarification: Clarification


class Taxonomy(BaseModel):
    version: str
    questions: Questions
    fallback: str
    domains: dict[str, Domain]

    @model_validator(mode="after")
    def _fallback_everywhere(self) -> "Taxonomy":
        for name, d in self.domains.items():
            if self.fallback not in d.categories:
                raise ValueError(f"domain {name} lacks the fallback category {self.fallback!r}")
        for q in self.questions.issue.instructions + self.questions.clarification.instructions:
            if "{domain}" not in q:
                raise ValueError(f"question wording must name the domain: {q!r}")
        return self

    def domain(self, name: str) -> Domain:
        if name not in self.domains:
            raise KeyError(f"unknown domain {name!r}; known: {sorted(self.domains)}")
        return self.domains[name]

    def categories(self, domain: str) -> list[str]:
        return list(self.domain(domain).categories)

    def criteria(self, domain: str) -> dict[str, str]:
        """Choice criteria in canonical order: {category id: English description}."""
        return {cid: c.description for cid, c in self.domain(domain).categories.items()}

    def issue_instructions(self, domain: str, variant: int = 0) -> str:
        qs = self.questions.issue.instructions
        return qs[variant % len(qs)].format(domain=self.domain(domain).name)

    def clarification_instructions(self, domain: str, variant: int = 0) -> str:
        qs = self.questions.clarification.instructions
        return qs[variant % len(qs)].format(domain=self.domain(domain).name)


def taxonomy_path(version: str = DEFAULT_VERSION) -> Path:
    return CONFIGS / "taxonomy" / f"{version}.yaml"


@lru_cache
def load_taxonomy(version: str = DEFAULT_VERSION) -> Taxonomy:
    with taxonomy_path(version).open(encoding="utf-8") as f:
        tax = Taxonomy.model_validate(yaml.safe_load(f))
    if tax.version != version:
        raise ValueError(f"{taxonomy_path(version)} declares version {tax.version!r}")
    return tax
