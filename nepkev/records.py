"""Canonical record schema and JSONL helpers.

A record is one customer message plus everything known about it. Only `message` ever reaches the model; every other
field is metadata or gold.

Origins (train/eval slices use them):
- `synthetic`: written by the generator LLM in clean, casual Devanagari;
- `translated`: derived from a synthetic record by code (Romanized, English mixed in);
- `realstyle`: written by the generator LLM in the style of app-store reviews (style tag `realstyle`);
- `external`: labelled records supplied from outside the pipeline (no script label); training data only.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Script = Literal["deva", "roman"]
Origin = Literal["synthetic", "translated", "realstyle", "external"]
Status = Literal["pending", "accepted", "rejected", "queued"]
REALSTYLE = "realstyle"           # style tag of real-review-style records
EXTERNAL = "external"             # Source.generator of external records


class Source(BaseModel):
    generator: str                      # llm | romanize | dry-run | external
    provider: str | None = None
    model: str | None = None
    revision: str | None = None
    prompt_version: str | None = None
    seed: int | None = None


class Labels(BaseModel):
    issue: str | None                   # None: not identifiable from the text
    issue_target: dict[str, float] | None = None   # soft target used when issue is None
    clarification_needed: bool

    @model_validator(mode="after")
    def _consistent(self) -> "Labels":
        if self.issue is not None and self.issue_target is not None:
            raise ValueError("a record has either a hard issue label or a soft target, not both")
        if self.issue_target is not None:
            total = sum(self.issue_target.values())
            if not self.issue_target or abs(total - 1) > 1e-3 or min(self.issue_target.values()) < 0:
                raise ValueError(f"issue_target must be a distribution, got sum {total}")
        return self


class Validation(BaseModel):
    status: Status = "pending"
    rejection_reason: str | None = None
    checks: dict[str, Any] = {}


class Derivation(BaseModel):
    """Lineage of a record derived by code (the Romanized variant of a Devanagari message)."""
    parent_record_id: str
    ops: list[str]
    seed: int


class Record(BaseModel):
    record_id: str
    scenario_id: str
    scenario_group_id: str              # siblings (Romanized variant, counterfactual pairs, realstyle twin) share a group
    domain: str
    taxonomy_version: str
    scenario_facts: dict[str, Any] = {}
    visible_facts: list[str] = []
    withheld_facts: list[str] = []
    message: str = Field(min_length=1)
    script: Script | None               # None exactly for external records
    style_tags: list[str] = []
    difficulty_family: str
    labels: Labels
    issues_all: list[str] = []
    stated_priority: str | None = None
    source: Source
    derivation: Derivation | None = None
    validation: Validation = Validation()
    content_hash: str = ""
    note: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Record":
        if (self.script is None) != (self.source.generator == EXTERNAL):
            raise ValueError("external records carry no script label; every generated record has exactly one")
        if not self.content_hash:
            self.content_hash = content_hash(self.message)
        return self

    @property
    def origin(self) -> Origin:
        if self.source.generator == EXTERNAL:
            return "external"
        if self.derivation is not None:
            return "translated"
        return "realstyle" if REALSTYLE in self.style_tags else "synthetic"


DEVA_TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize(text: str) -> str:
    """Comparison form: NFC, lowercase, digits unified to ASCII, punctuation and extra spaces removed."""
    t = unicodedata.normalize("NFC", text).lower().translate(DEVA_TO_ASCII)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{n}: {e}") from e


def write_jsonl(path: str | Path, rows: Iterable[dict | BaseModel], append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a" if append else "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            d = r.model_dump(mode="json") if isinstance(r, BaseModel) else r
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_records(path: str | Path) -> list[Record]:
    return [Record.model_validate(d) for d in read_jsonl(path)]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
