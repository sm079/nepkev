"""Sentence embeddings for retrieval: open multilingual models run locally with transformers, L2-normalised.

Each supported model is described by its pooling and the prefixes its model card asks for (E5 was trained with
"query: " / "passage: "; BGE-M3's dense vectors are its [CLS] state and take no prefix). torch and transformers are
imported on first use, so code that only stores or compares vectors does not need them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str
    dim: int
    pooling: str            # "mean" | "cls"
    query_prefix: str = ""
    passage_prefix: str = ""
    max_tokens: int = 512


MODELS = {
    "intfloat/multilingual-e5-small": ModelSpec("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3",
                                                384, "mean", "query: ", "passage: "),
    "intfloat/multilingual-e5-base": ModelSpec("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a",
                                               768, "mean", "query: ", "passage: "),
    "BAAI/bge-m3": ModelSpec("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", 1024, "cls", max_tokens=1024),
}


class HFEmbedder:
    """One of MODELS, pinned to a revision, on CPU by default. Calls are serialised (one model, shared)."""

    def __init__(self, name: str, device: str = "cpu", batch_size: int = 32):
        if name not in MODELS:
            raise ValueError(f"unknown embedding model {name!r}; known: {sorted(MODELS)}")
        self.spec, self.name, self.dim = MODELS[name], name, MODELS[name].dim
        self.device, self.batch_size = device, batch_size
        self._lock = threading.Lock()
        self._model = self._tok = None

    def load(self) -> HFEmbedder:
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.spec.repo, revision=self.spec.revision)
        self._model = AutoModel.from_pretrained(self.spec.repo, revision=self.spec.revision).to(self.device).eval()
        return self

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.spec.query_prefix + t for t in texts])

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.spec.passage_prefix + t for t in texts])

    def _embed(self, texts: list[str]) -> np.ndarray:
        import torch

        if self._model is None:
            self.load()
        out = []
        with self._lock, torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = self._tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                                  max_length=self.spec.max_tokens, return_tensors="pt").to(self.device)
                h = self._model(**batch).last_hidden_state
                if self.spec.pooling == "cls":
                    v = h[:, 0]
                else:
                    mask = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
                    v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)
