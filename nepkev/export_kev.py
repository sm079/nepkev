"""Canonical records -> Kev training/eval JSONL (the System One request shape plus labels).

Only the message becomes the state. Question ids are neutral (`q1` issue, `q2` clarification) and never seen by the
model. Training exports vary the question wording and the candidate set; evaluation exports use the serving wording
and the canonical candidate order, so evaluation matches the inference contract exactly.

The gold category is never removed from the candidates. Kev's own trainer does that (p_none) to teach "none of the
above", because its datasets have no out-of-scope class. Here `other` is that option, it is always a candidate, and
balanced out-of-scope records teach it; removing the gold would make `other` the most common answer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .records import Record, write_jsonl
from .taxonomy import Taxonomy

ISSUE_QID, CLAR_QID = "q1", "q2"


@dataclass(frozen=True)
class ExportOptions:
    train: bool = False
    p_paraphrase: float = 0.5    # train: use a non-serving question wording
    p_drop_others: float = 0.15  # train: remove 1-3 candidates that are neither the gold nor the fallback
    shuffle: bool = True         # train: shuffle candidate order (Kev's trainer also permutes)
    seed: int = 0


def issue_question(rec: Record, tax: Taxonomy, opts: ExportOptions, rng: random.Random) -> dict:
    crit = tax.criteria(rec.domain)
    variant = 0
    if opts.train and rng.random() < opts.p_paraphrase:
        variant = rng.randrange(1, len(tax.questions.issue.instructions))
    q = {"type": "choice", "instructions": tax.issue_instructions(rec.domain, variant)}
    keys = list(crit)
    lab = rec.labels
    if lab.issue is None:
        # soft target (vague or multi-issue without priority): the candidate set stays whole, or the target would change
        if opts.train and opts.shuffle:
            rng.shuffle(keys)
        q["criteria"] = {k: crit[k] for k in keys}
        target = lab.issue_target or {k: 1 / len(crit) for k in crit}
        q["target"] = dict(target)
        q["label"] = max(target, key=target.get)   # required by Kev; never scored
        return q
    if opts.train:
        if rng.random() < opts.p_drop_others:
            droppable = [k for k in keys if k not in (lab.issue, tax.fallback)]
            for k in rng.sample(droppable, min(len(droppable), rng.randint(1, 3))):
                keys.remove(k)
        if opts.shuffle:
            rng.shuffle(keys)
    q["criteria"] = {k: crit[k] for k in keys}
    q["label"] = lab.issue
    return q


def clarification_question(rec: Record, tax: Taxonomy, opts: ExportOptions, rng: random.Random) -> dict:
    variant = 0
    if opts.train and rng.random() < opts.p_paraphrase:
        variant = rng.randrange(1, len(tax.questions.clarification.instructions))
    return {"type": "noul", "instructions": tax.clarification_instructions(rec.domain, variant),
            "criteria": dict(tax.questions.clarification.criteria), "label": rec.labels.clarification_needed}


def to_kev(rec: Record, tax: Taxonomy, opts: ExportOptions) -> dict:
    rng = random.Random(f"{opts.seed}:{rec.record_id}")      # per record, so one record's export never shifts another's
    q1 = issue_question(rec, tax, opts, rng)
    q2 = clarification_question(rec, tax, opts, rng)
    return {"state": rec.message, "questions": {ISSUE_QID: q1, CLAR_QID: q2}}


def export(recs: list[Record], tax: Taxonomy, out: Path, opts: ExportOptions) -> int:
    """Write `out` (Kev JSONL, one line per record) and `out`.index.jsonl (line -> record id, to join predictions back)."""
    write_jsonl(out, (to_kev(r, tax, opts) for r in recs))
    write_jsonl(out.with_suffix(".index.jsonl"),
                ({"line": i, "record_id": r.record_id, "scenario_group_id": r.scenario_group_id} for i, r in enumerate(recs)))
    return len(recs)
