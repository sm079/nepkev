"""Write the demo's data files from the taxonomy and the held-out test split.

  demo/data/taxonomy.json   per domain: the serving questions exactly as evaluation sends them (instructions, criteria in
                            canonical order) plus the Nepali category names for the UI
  demo/data/presets.json    example messages picked from the test split (never trained on), with their labels

  uv run python demo/tools/build_demo_data.py --test data/mix-v3-add/splits/test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nepkev.records import read_jsonl
from nepkev.taxonomy import load_taxonomy

OUT = Path(__file__).resolve().parents[1] / "data"

# (family, origin) slots per domain, in display order; the first test record that fits each slot is used
SLOTS = [
    ("single", "clean"), ("single", "translated"), ("single", "app-review"),
    ("contrast", "clean"), ("negation", "translated"), ("corrected_statement", "clean"),
    ("multi_issue:stated_priority", "app-review"), ("out_of_scope", "clean"), ("out_of_scope", "app-review"),
    ("vague", "translated"), ("multi_issue:no_priority", "clean"),
]
MAX_CHARS = 260


def origin(rec: dict) -> str:
    src = rec["source"]
    if src["generator"] == "romanize":
        return "translated"
    return "app-review" if src.get("prompt_version", "").startswith("generate_realstyle") else "clean"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--taxonomy", default="demo-1")
    args = ap.parse_args()

    tax = load_taxonomy(args.taxonomy)
    taxonomy = {"version": tax.version, "fallback": tax.fallback, "domains": {}}
    for name in tax.domains:
        dom = tax.domain(name)
        taxonomy["domains"][name] = {
            "name": dom.name, "ne": dom.ne,
            "issue": {"type": "choice", "instructions": tax.issue_instructions(name), "criteria": tax.criteria(name)},
            "clarification": {"type": "noul", "instructions": tax.clarification_instructions(name),
                              "criteria": dict(tax.questions.clarification.criteria)},
            "names_ne": {k: c.ne for k, c in dom.categories.items()},
        }

    recs = sorted(read_jsonl(args.test), key=lambda r: r["record_id"])
    presets = []
    for name in tax.domains:
        used: set[str] = set()
        for family, want in SLOTS:
            for r in recs:
                fam = r["difficulty_family"]
                if (r["domain"] != name or origin(r) != want or r["scenario_group_id"] in used
                        or len(r["message"]) > MAX_CHARS or not (fam == family or fam.startswith(family + ":"))):
                    continue
                used.add(r["scenario_group_id"])
                lab = r["labels"]
                presets.append({"id": r["record_id"], "domain": name, "message": r["message"], "script": r["script"],
                                "origin": want, "family": fam, "issue": lab["issue"],
                                "issues_all": r.get("issues_all") or [], "clarification": lab["clarification_needed"]})
                break

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "taxonomy.json").write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "presets.json").write_text(json.dumps(presets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(taxonomy['domains'])} domains and {len(presets)} presets to {OUT}")


if __name__ == "__main__":
    main()
