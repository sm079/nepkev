"""The knowledge base: markdown articles with front matter, and their split into chunks with stable ids.

A chunk id is `<doc id>#<12 hex chars>`: a hash of the text that is embedded (title and chunk text). An unchanged
chunk keeps its id across runs and edits elsewhere in its article, so ingestion re-embeds only what changed, and a
citation always names the exact text that was retrieved.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from .. import ROOT

KB_DIR = ROOT / "kb"
DOMAINS = ("shop", "wallet", "bank")
LANGS = ("en", "ne")
MAX_CHUNK_CHARS = 400

_FRONT = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.S)


@dataclass(frozen=True)
class Doc:
    id: str
    domain: str
    lang: str
    title: str
    updated_at: date
    body: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    domain: str
    lang: str
    title: str
    text: str
    position: int
    updated_at: date

    @property
    def embed_text(self) -> str:
        return f"{self.title}\n\n{self.text}"


def normalize(text: str) -> str:
    """NFC and collapsed whitespace: the form chunk text is stored in and quotes are compared in."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def parse_doc(text: str, source: str = "<doc>") -> Doc:
    m = _FRONT.match(text.replace("\r\n", "\n"))
    if not m:
        raise ValueError(f"{source}: missing front matter")
    meta = yaml.safe_load(m.group(1)) or {}
    missing = {"id", "domain", "lang", "title", "updated_at"} - set(meta)
    if missing:
        raise ValueError(f"{source}: front matter lacks {sorted(missing)}")
    if meta["domain"] not in DOMAINS or meta["lang"] not in LANGS:
        raise ValueError(f"{source}: domain must be one of {DOMAINS} and lang one of {LANGS}")
    updated = meta["updated_at"]
    if not isinstance(updated, date):
        updated = date.fromisoformat(str(updated))
    return Doc(id=str(meta["id"]), domain=meta["domain"], lang=meta["lang"], title=str(meta["title"]),
               updated_at=updated, body=m.group(2).strip())


def load_kb(kb_dir: Path = KB_DIR) -> list[Doc]:
    docs = [parse_doc(p.read_text(encoding="utf-8"), str(p)) for p in sorted(kb_dir.glob("*/*.md"))]
    ids = [d.id for d in docs]
    if dupes := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"duplicate document ids {dupes}")
    return docs


def chunk_doc(doc: Doc, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Paragraphs (a list stays with the paragraph before it) packed greedily into chunks of at most max_chars."""
    blocks: list[str] = []
    for para in re.split(r"\n\s*\n", doc.body):
        para = para.strip()
        if not para:
            continue
        if blocks and para.lstrip().startswith("- ") and blocks[-1].endswith(":"):
            blocks[-1] = f"{blocks[-1]}\n{para}"
        else:
            blocks.append(para)
    groups: list[str] = []
    for b in blocks:
        if groups and len(groups[-1]) + len(b) + 2 <= max_chars:
            groups[-1] = f"{groups[-1]}\n\n{b}"
        else:
            groups.append(b)
    chunks, seen = [], set()
    for i, g in enumerate(groups):
        text = normalize(g)
        digest = hashlib.sha256(f"{doc.title}\n{text}".encode()).hexdigest()[:12]
        cid = f"{doc.id}#{digest}"
        if cid in seen:   # the same text twice in one article: keep one copy
            continue
        seen.add(cid)
        chunks.append(Chunk(id=cid, doc_id=doc.id, domain=doc.domain, lang=doc.lang, title=doc.title, text=text,
                            position=i, updated_at=doc.updated_at))
    return chunks


def kb_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    return [c for d in load_kb(kb_dir) for c in chunk_doc(d)]
