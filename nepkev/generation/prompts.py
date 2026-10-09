"""Prompt assembly and output schemas for generation and verification.

The system prompt is byte-identical across requests (all domains, fixed order) so it stays in the
provider's prompt cache; everything request-specific goes in the user message.
"""

from __future__ import annotations

import json
import re

from .. import PROMPTS
from ..scenarios import Scenario
from ..taxonomy import Taxonomy


def render_taxonomy(tax: Taxonomy) -> str:
    lines = []
    for key, dom in tax.domains.items():
        lines.append(f"## {dom.name} (service id: {key})")
        for cid, c in dom.categories.items():
            line = f"- {cid}: {c.description}"
            if c.exclude:
                line += f". Not this: {'; '.join(c.exclude)}"
            lines.append(line)
        lines.append("")
    return "\n".join(lines).strip()


def render_taxonomy_plain(tax: Taxonomy) -> str:
    """The same list without category ids, for the renderer: ids it never sees are ids it cannot copy into text."""
    lines = []
    for dom in tax.domains.values():
        lines.append(f"## {dom.name}")
        for c in dom.categories.values():
            line = f"- {c.description}"
            excl = [re.sub(r"\s*->.*$", "", e).strip() for e in c.exclude]
            if excl:
                line += f". Not this: {'; '.join(excl)}"
            lines.append(line)
        lines.append("")
    return "\n".join(lines).strip()


def system_prompt(name: str, tax: Taxonomy) -> str:
    text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    return text.replace("{taxonomy_plain}", render_taxonomy_plain(tax)).replace("{taxonomy}", render_taxonomy(tax))


def generation_user_message(scenarios: list[Scenario], tax: Taxonomy, extra: dict[str, dict] | None = None) -> str:
    """Request-specific part. Carries the situation in plain words; labels and category ids stay out.
    `extra` adds fields to a scenario's spec, by scenario_id."""
    specs = [{
        "scenario_id": s.scenario_id,
        "service": tax.domain(s.domain).name,
        "situation": s.instructions,
        "visible_facts": {k: s.facts[k] for k in s.visible_facts},
        "withheld_facts_never_mention": s.withheld_facts,
        "length": s.style["length"], "tone": s.style["tone"], "formality": s.style["formality"],
        **(extra or {}).get(s.scenario_id, {}),
    } for s in scenarios]
    return ("Write one message for each of these scenarios. Return one item per scenario_id.\n\n"
            + json.dumps(specs, ensure_ascii=False, indent=1, sort_keys=True))


def verification_user_message(items: list[dict]) -> str:
    """items: [{"id", "service", "message"}]"""
    return ("Label each message. Return one item per id.\n\n"
            + json.dumps(items, ensure_ascii=False, indent=1, sort_keys=True))


GENERATION_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"scenario_id": {"type": "string"}, "message": {"type": "string"}, "evidence": {"type": "string"}},
        "required": ["scenario_id", "message", "evidence"],
        "additionalProperties": False}}},
    "required": ["items"],
    "additionalProperties": False,
}

VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "issue": {"type": ["string", "null"]},
                       "clarification_needed": {"type": "boolean"},
                       "issues_mentioned": {"type": "array", "items": {"type": "string"}},
                       "language_ok": {"type": "boolean"}},
        "required": ["id", "issue", "clarification_needed", "issues_mentioned", "language_ok"],
        "additionalProperties": False}}},
    "required": ["items"],
    "additionalProperties": False,
}
