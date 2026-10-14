"""configs/rag.yaml: embedding model, top-k, the refusal threshold chosen on dev, and the prompt version."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .. import CONFIGS

RAG_CONFIG = CONFIGS / "rag.yaml"


@dataclass(frozen=True)
class RagConfig:
    embedding_model: str
    top_k: int
    min_similarity: float | None
    prompt: str


def load_rag_config(path: Path = RAG_CONFIG) -> RagConfig:
    return RagConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))


def write_min_similarity(value: float, path: Path = RAG_CONFIG) -> None:
    """Rewrite only the min_similarity line, so the file's comments stay."""
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(r"^min_similarity:.*$", f"min_similarity: {value:.6f}", text, flags=re.M)
    if n != 1:
        raise ValueError(f"{path}: expected one min_similarity line")
    path.write_text(new, encoding="utf-8", newline="\n")
