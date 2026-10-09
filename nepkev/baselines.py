"""Keyword router baseline (configs/rules/demo-1.yaml).

Produces the same prediction shape as Kev (issue probabilities over the domain's categories and a clarification
probability) so both go through one evaluation. Scores are cue counts; probabilities are smoothed cue scores, so NLL
and calibration numbers for this baseline describe a heuristic, not a probabilistic model.
"""

from __future__ import annotations

from functools import lru_cache

import yaml

from . import CONFIGS
from .records import Record
from .taxonomy import Taxonomy


@lru_cache
def load_rules(name: str = "demo-1") -> dict:
    return yaml.safe_load((CONFIGS / "rules" / f"{name}.yaml").read_text(encoding="utf-8"))


def _score(text: str, cues: list) -> float:
    s = 0.0
    for c in cues:
        if isinstance(c, str):
            s += c.lower() in text
        else:
            terms, w = (c["all"], c.get("weight", 1)) if isinstance(c, dict) else (c, 1)
            s += w * all(t.lower() in text for t in terms)
    return s


def rule_predict(rec: Record, tax: Taxonomy, rules: dict | None = None) -> dict:
    rules = rules or load_rules()
    text = rec.message.lower()
    cats = tax.categories(rec.domain)
    scores = {c: _score(text, rules["domains"][rec.domain].get(c, [])) for c in cats}
    total = sum(scores.values())
    if total == 0:
        return {"issue": {c: 1 / len(cats) for c in cats}, "clarification": 0.9, "rule": "no cue"}
    smoothed = {c: s + 0.05 for c, s in scores.items()}
    z = sum(smoothed.values())
    top = sorted(scores.values(), reverse=True)
    tie = len(top) > 1 and top[0] == top[1]
    return {"issue": {c: v / z for c, v in smoothed.items()}, "clarification": 0.6 if tie else 0.1,
            "rule": "tie" if tie else "match"}
