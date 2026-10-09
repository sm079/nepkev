"""App-review-style messages: the accepted pilot scenarios written again the way people type app-store reviews.

The pilot's messages are clean, careful Nepali. This renders each accepted pilot scenario once more, from its
situation (not from the clean message), in the style of the example reviews in the system prompt: fast, messy, often
rude, mostly Romanized. Labels come from the scenario as before, so the category balance is unchanged; the verifier
then re-labels every message from its text, and disputed ones are adjudicated.

- Script: `deva_share` of the scenarios are written in Devanagari, the rest Romanized. Scripts are assigned within
  each (domain, label) bucket, so script says nothing about the label.
- Style examples: a fixed list (configs' `realstyle.style_examples`), the same for every request, so it stays in the
  prompt cache and the run is reproducible.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .. import ROOT
from ..records import REALSTYLE, Labels, Record, Source, read_jsonl
from ..scenarios import Scenario
from ..taxonomy import Taxonomy
from .prompts import generation_user_message, system_prompt


def accepted_scenarios(scenarios_path: Path, synthetic: list[Record]) -> list[Scenario]:
    """Scenarios whose clean message was accepted (they have an accepted synthetic record)."""
    keep = {r.scenario_id for r in synthetic if r.origin == "synthetic" and r.validation.status == "accepted"}
    return [s for s in (Scenario.model_validate(d) for d in read_jsonl(scenarios_path)) if s.scenario_id in keep]


def assign_scripts(scenarios: list[Scenario], deva_share: float, seed: int) -> dict[str, str]:
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for s in sorted(scenarios, key=lambda s: s.scenario_id):
        buckets[(s.domain, s.issue or s.family.split(":")[0])].append(s.scenario_id)
    rng, out = random.Random(seed), {}
    for _, ids in sorted(buckets.items()):
        rng.shuffle(ids)
        # stochastic rounding keeps the overall share right when buckets are small
        n_deva = int(len(ids) * deva_share + rng.random())
        out |= {sid: ("deva" if i < n_deva else "roman") for i, sid in enumerate(ids)}
    return out


def load_style_examples(path: str) -> list[str]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def realstyle_system_prompt(name: str, tax: Taxonomy, examples: list[str]) -> str:
    return system_prompt(name, tax).replace("{examples}", "\n".join(f"- {' '.join(e.split())}" for e in examples))


def user_message_with_scripts(scripts: dict[str, str]):
    def build(scenarios: list[Scenario], tax: Taxonomy) -> str:
        return generation_user_message(scenarios, tax, {s.scenario_id: {"script": scripts[s.scenario_id]} for s in scenarios})
    return build


def sample_scenarios(scenarios: list[Scenario], n: int, seed: int) -> list[Scenario]:
    """A spread-out sample for a style check before a full run: round-robin over (domain, family) buckets."""
    buckets: dict[tuple[str, str], list[Scenario]] = defaultdict(list)
    for s in sorted(scenarios, key=lambda s: s.scenario_id):
        buckets[(s.domain, s.family.split(":")[0])].append(s)
    rng = random.Random(seed)
    for b in buckets.values():
        rng.shuffle(b)
    keys, out, i = sorted(buckets), [], 0
    rng.shuffle(keys)
    while len(out) < min(n, len(scenarios)):
        b = buckets[keys[i % len(keys)]]
        if b:
            out.append(b.pop())
        i += 1
    return out


def build_realstyle_records(scenarios: list[Scenario], raw: list[dict], scripts: dict[str, str], seed: int) -> list[Record]:
    scen = {s.scenario_id: s for s in scenarios}
    recs, seen = [], set()
    for row in raw:
        s = scen.get(row["scenario_id"])
        if s is None or s.scenario_id in seen:
            continue
        seen.add(s.scenario_id)
        script = scripts[s.scenario_id]
        recs.append(Record(
            record_id=f"{s.scenario_id}-rs-{script}", scenario_id=s.scenario_id, scenario_group_id=s.scenario_group_id,
            domain=s.domain, taxonomy_version=s.taxonomy_version, scenario_facts=s.facts,
            visible_facts=s.visible_facts, withheld_facts=s.withheld_facts, message=row["message"].strip(), script=script,
            style_tags=[s.style["length"], s.style["tone"], s.style["formality"], REALSTYLE,
                        *(["negation"] if s.family == "negation" else [])],
            difficulty_family=s.family,
            labels=Labels(issue=s.issue, issue_target=s.issue_target, clarification_needed=s.clarification_needed),
            issues_all=s.issues_all, stated_priority=s.stated_priority,
            source=Source(generator="llm", provider=row["provider"], model=row["model"],
                          prompt_version=row["prompt_version"], seed=seed),
            note=row.get("evidence", "")))
    return recs
