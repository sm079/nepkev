"""Group-level train/dev/calib/test split with leakage checks, and the training mix built on it.

Everything that shares a scenario group (a message, its Romanized twin, its app-review-style twin, counterfactual
pairs) lands in one split. Groups that contain near-identical messages are merged first, so a paraphrase in train can
never leak a test item. Test never tunes anything: checkpoint selection uses `dev`, temperature and thresholds use
`calib`.

The mix (`build_splits`): clean records (synthetic + translated) are assigned by group; app-review-style records join
the split their scenario group already has, so adding them never moves a clean record; external labelled records, if
any, are appended to the splits they come in (train, dev or calib only: they are never scored).
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .records import Record, content_hash
from .validate import near_duplicate_pairs

SPLITS = ("train", "dev", "calib", "test")
DEFAULT_FRACTIONS = {"train": 0.80, "dev": 0.05, "calib": 0.05, "test": 0.10}


class _UnionFind:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


@dataclass
class SplitResult:
    assignment: dict[str, str]                       # record_id -> split
    merged_groups: dict[str, str]                    # scenario_group_id -> leakage cluster id
    merges: list[tuple[str, str, float]] = field(default_factory=list)


def _bucket(cluster: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{cluster}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign(recs: list[Record], fractions: dict[str, float] = DEFAULT_FRACTIONS, seed: int = 0,
           near_dup_threshold: float = 0.85) -> SplitResult:
    if abs(sum(fractions.values()) - 1) > 1e-9 or set(fractions) != set(SPLITS):
        raise ValueError(f"fractions must cover {SPLITS} and sum to 1")
    uf = _UnionFind()
    for r in recs:
        uf.find(r.scenario_group_id)
    by_hash: dict[str, str] = {}
    for r in recs:                                   # identical normalized text -> same cluster
        uf.union(by_hash.setdefault(r.content_hash, r.scenario_group_id), r.scenario_group_id)
    groups = {r.record_id: r.scenario_group_id for r in recs}
    merges = near_duplicate_pairs({r.record_id: r.message for r in recs}, groups, near_dup_threshold)
    for a, b, _ in merges:
        uf.union(groups[a], groups[b])
    cluster = {g: uf.find(g) for g in set(groups.values())}
    edges, acc = [], 0.0
    for s in SPLITS:
        acc += fractions[s]
        edges.append((acc, s))
    def split_of(c: str) -> str:
        u = _bucket(c, seed)
        return next(s for e, s in edges if u < e) if u < edges[-1][0] else SPLITS[-1]
    return SplitResult({r.record_id: split_of(cluster[r.scenario_group_id]) for r in recs}, cluster, merges)


def leakage_report(recs: list[Record], assignment: dict[str, str]) -> dict:
    """Proves isolation: no scenario group and no normalized message text appears in two splits."""
    group_splits: dict[str, set[str]] = {}
    text_splits: dict[str, set[str]] = {}
    for r in recs:
        s = assignment[r.record_id]
        group_splits.setdefault(r.scenario_group_id, set()).add(s)
        text_splits.setdefault(content_hash(r.message), set()).add(s)
    bad_groups = sorted(g for g, s in group_splits.items() if len(s) > 1)
    bad_texts = sorted(h for h, s in text_splits.items() if len(s) > 1)
    return {"groups_in_multiple_splits": bad_groups, "texts_in_multiple_splits": bad_texts,
            "ok": not bad_groups and not bad_texts}


def summary(recs: list[Record], assignment: dict[str, str]) -> dict:
    out = {}
    for s in SPLITS:
        rs = [r for r in recs if assignment[r.record_id] == s]
        out[s] = {"records": len(rs), "groups": len({r.scenario_group_id for r in rs}),
                  "canonical_scenarios": len({r.scenario_id for r in rs}),
                  "by_script": dict(Counter(r.script for r in rs)),
                  "by_domain": dict(Counter(r.domain for r in rs)),
                  "clarification_true": sum(r.labels.clarification_needed for r in rs)}
    return out


def build_splits(clean: list[Record], realstyle: list[Record] = (), external: dict[str, list[Record]] | None = None,
                 seed: int = 0, near_dup_threshold: float = 0.85) -> tuple[dict[str, list[Record]], dict]:
    """The training mix. Within a split: clean records in input order, then external, then realstyle."""
    external = external or {}
    if bad := set(external) - {"train", "dev", "calib"}:
        raise ValueError(f"external records only go to train, dev or calib, not {sorted(bad)}")
    res = assign(clean, seed=seed, near_dup_threshold=near_dup_threshold)
    out: dict[str, list[Record]] = {s: [] for s in SPLITS}
    for r in clean:
        out[res.assignment[r.record_id]].append(r)
    for s, rs in external.items():
        out[s] += rs
    group_split = {r.scenario_group_id: res.assignment[r.record_id] for r in clean}
    unplaced = [r.record_id for r in realstyle if r.scenario_group_id not in group_split]
    for r in realstyle:
        if r.scenario_group_id in group_split:
            out[group_split[r.scenario_group_id]].append(r)
    where = {r.record_id: s for s in SPLITS for r in out[s]}
    stats = {"splits": {s: {"records": len(out[s]), "by_origin": dict(Counter(r.origin for r in out[s]))} for s in SPLITS},
             "near_duplicate_merges": len(res.merges), "realstyle_unplaced": unplaced,
             "leakage": leakage_report([r for s in SPLITS for r in out[s]], where),
             "train_label_shares": label_shares(out["train"])}
    return out, stats


def label_shares(recs: list[Record]) -> dict[str, dict[str, float]]:
    """Per domain, each hard label's share of the domain's hard-labelled records (the balance check)."""
    c: dict[str, Counter] = defaultdict(Counter)
    for r in recs:
        if r.labels.issue is not None:
            c[r.domain][r.labels.issue] += 1
    return {d: {k: round(v / sum(n.values()), 4) for k, v in sorted(n.items())} for d, n in sorted(c.items())}
