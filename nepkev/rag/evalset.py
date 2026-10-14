"""The RAG evaluation set (eval/rag_set.jsonl), its dev/test split, and the metrics shared by the eval scripts.

Each row is a question in English, Devanagari or Romanized Nepali with the ids of the articles that answer it (an
English and a Nepali version of one topic), or no ids and must_refuse for questions the knowledge base does not cover.
The split is fixed by a hash of the row id: about a third dev (threshold and model choice), the rest test (reported).

Confidence intervals are 95% percentile bootstraps over topics: every question about one topic (its paraphrases in the
three scripts) is resampled together, and each must-refuse question is its own unit.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import ROOT

EVAL_SET = ROOT / "eval" / "rag_set.jsonl"
SCRIPTS = ("en", "deva", "roman")


@dataclass(frozen=True)
class EvalRow:
    id: str
    domain: str
    script: str
    question: str
    gold_doc_ids: tuple[str, ...]
    must_refuse: bool

    @property
    def split(self) -> str:
        return "dev" if int(hashlib.sha256(self.id.encode()).hexdigest(), 16) % 3 == 0 else "test"

    @property
    def unit(self) -> str:
        """Bootstrap unit: the topic for answerable questions, the question itself otherwise."""
        return self.gold_doc_ids[0].rsplit("-", 1)[0] if self.gold_doc_ids else self.id


def load_eval_set(path: Path = EVAL_SET) -> list[EvalRow]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            rows.append(EvalRow(d["id"], d["domain"], d["script"], d["question"], tuple(d["gold_doc_ids"]),
                                d["must_refuse"]))
    for r in rows:
        if r.must_refuse == bool(r.gold_doc_ids) or r.script not in SCRIPTS:
            raise ValueError(f"{r.id}: needs gold documents xor must_refuse, and a script in {SCRIPTS}")
    return rows


def first_gold_rank(doc_ids: Sequence[str], gold: Sequence[str]) -> int | None:
    """1-based rank of the first retrieved chunk from a gold article, or None."""
    return next((i for i, d in enumerate(doc_ids, 1) if d in gold), None)


def recall_at(ranks: Sequence[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks) if ranks else float("nan")


def mrr(ranks: Sequence[int | None]) -> float:
    return sum(1 / r for r in ranks if r is not None) / len(ranks) if ranks else float("nan")


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """P(a random positive scores above a random negative); ties count half."""
    if not positives or not negatives:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default, Postgres percentile_cont)."""
    xs = sorted(xs)
    if not xs:
        return float("nan")
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def bootstrap(items: Sequence, unit: Callable[[object], str], metrics: dict[str, Callable[[list], float]],
              samples: int = 1000, seed: int = 0) -> dict[str, list[float]]:
    """95% percentile intervals of each metric, resampling whole units (see the module docstring)."""
    groups: dict[str, list] = defaultdict(list)
    for x in items:
        groups[unit(x)].append(x)
    keys, rng = list(groups), random.Random(seed)
    stats: dict[str, list[float]] = {name: [] for name in metrics}
    for _ in range(samples):
        sample = [x for k in rng.choices(keys, k=len(keys)) for x in groups[k]]
        for name, fn in metrics.items():
            stats[name].append(fn(sample))

    def ci(v: list[float]) -> list[float]:
        v = sorted(x for x in v if not math.isnan(x))
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]] if v else [float("nan")] * 2
    return {name: ci(v) for name, v in stats.items()}


def best_threshold(scores_answerable: Sequence[float], scores_refuse: Sequence[float], min_kept: float = 1.0) -> float:
    """The similarity cut-off for the refusal gate, from dev scores.

    The gate's two mistakes cost differently: an answerable question it refuses is lost, while an out-of-scope question
    it lets through still meets the answer model, which can decline it. So the cut-off must keep at least `min_kept` of
    the answerable questions, and among those cut-offs it refuses the most must-refuse questions. Ties go to the middle
    of the widest gap between scores, so the cut-off does not sit on a dev point.
    """
    cands = sorted(set(scores_answerable) | set(scores_refuse))
    mids = [cands[0] - 1e-3] + [(a + b) / 2 for a, b in zip(cands, cands[1:])]

    def kept(t: float) -> float:
        return sum(s >= t for s in scores_answerable) / len(scores_answerable)

    def refused(t: float) -> float:
        return sum(s < t for s in scores_refuse) / len(scores_refuse)

    feasible = [t for t in mids if kept(t) >= min_kept]
    best = max(refused(t) for t in feasible)
    ties = [t for t in feasible if refused(t) == best]
    gaps = []
    for t in ties:
        below = max((c for c in cands if c < t), default=t)
        above = min((c for c in cands if c > t), default=t)
        gaps.append((above - below, t))
    return max(gaps)[1]
