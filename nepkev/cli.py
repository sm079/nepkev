"""`nepkev` command line: one subcommand per pipeline stage (see README for the full flow).

    nepkev pipeline --config fixture-dryrun --out data/dryrun --dry-run     # network-free end-to-end check
    nepkev estimate --config pilot                                          # cost estimate before a paid run
    nepkev generate --config pilot --out data/pilot --batch --yes           # paid: refuses without --yes

Every paid step skips work already on disk, so re-running it on a finished run directory rebuilds the outputs from
the stored model responses without calling any API (and without --yes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from . import CONFIGS, ROOT
from .records import file_sha256, load_records, read_jsonl, write_jsonl
from .split import SPLITS
from .taxonomy import DEFAULT_VERSION, load_taxonomy


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """KEY=value lines from .env into the environment; variables already set win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _upstream() -> dict:
    import yaml
    return yaml.safe_load((CONFIGS / "upstream.yaml").read_text(encoding="utf-8"))


def _cfg(args):
    from .config import load_genconfig
    cfg = load_genconfig(args.config)
    if getattr(args, "dry_run", False):
        for call in (cfg.generator, cfg.verifier, cfg.adjudicator):
            if call is not None:
                call.provider, call.model = "dry_run", "dry-run"
    return cfg


def _confirm_paid(cfg, args, role: str, pending: int) -> None:
    """Refuse a paid call without --yes. Nothing pending (all outputs on disk) or the dry-run provider needs no --yes."""
    call = getattr(cfg, role)
    if not pending or call.provider == "dry_run" or args.yes:
        return
    print(f"This sends {pending} items to {call.provider}:{call.model} and costs money. Check `nepkev estimate "
          f"--config {args.config}`, then re-run with --yes.", file=sys.stderr)
    sys.exit(2)


def _pending(path: Path, key: str, ids: set[str]) -> int:
    done = {r[key] for r in read_jsonl(path)} if path.exists() else set()
    return len(ids - done)


def _accepted(path) -> list:
    return [r for r in load_records(path) if r.validation.status == "accepted"]


# ---------------------------------------------------------------- generation

def cmd_taxonomy(args) -> None:
    tax = load_taxonomy(args.version)
    _print({"version": tax.version, "fallback": tax.fallback,
            "domains": {d: {"categories": len(x.categories), "contrasts": len(x.contrasts)} for d, x in tax.domains.items()}})


def cmd_estimate(args) -> None:
    from .generation.run import estimate
    cfg = _cfg(args)
    _print(estimate(cfg, load_taxonomy(cfg.taxonomy_version)))


def cmd_scenarios(args) -> None:
    from .generation.run import load_or_sample_scenarios
    cfg = _cfg(args)
    sc = load_or_sample_scenarios(cfg, load_taxonomy(cfg.taxonomy_version), Path(args.out))
    _print({"scenarios": len(sc), "by_domain": Counter(s.domain for s in sc),
            "by_family": Counter(s.family.split(":")[0] for s in sc),
            "by_issue": Counter(f"{s.domain}/{s.issue or '(clarify)'}" for s in sc)})


def cmd_generate(args) -> None:
    from .generation.run import generate, load_or_sample_scenarios
    cfg, out = _cfg(args), Path(args.out)
    tax = load_taxonomy(cfg.taxonomy_version)
    ids = {s.scenario_id for s in load_or_sample_scenarios(cfg, tax, out)}
    _confirm_paid(cfg, args, "generator", _pending(out / "raw.jsonl", "scenario_id", ids))
    _print(generate(cfg, tax, out, args.limit, args.batch))


def cmd_records(args) -> None:
    from .generation.run import build_records
    out = Path(args.out)
    n = write_jsonl(out / "records.jsonl", build_records(_cfg(args), out))
    _print({"records": n, "path": str(out / "records.jsonl")})


def _verify(args, role: str) -> None:
    from .generation.run import disputed, merge_verdicts, verify
    cfg, out = _cfg(args), Path(args.out)
    recs = load_records(out / "records.jsonl")
    if role == "adjudicator":
        merge_verdicts(recs, out / "verify.jsonl", "verifier")
    targets = disputed(recs) if role == "adjudicator" else recs
    path = out / ("verify.jsonl" if role == "verifier" else f"{role}.jsonl")
    _confirm_paid(cfg, args, role, _pending(path, "id", {r.record_id for r in targets}))
    stats = verify(cfg, load_taxonomy(cfg.taxonomy_version), targets, out, role)
    write_jsonl(out / "records.jsonl", recs)
    _print({"checked": len(targets), **stats})


def cmd_verify(args) -> None:
    """Independent labels for every generated message from a second model (text only)."""
    _verify(args, "verifier")


def cmd_adjudicate(args) -> None:
    """A stronger model's verdict on the messages the verifier disputed; run validate afterwards."""
    _verify(args, "adjudicator")


def cmd_validate(args) -> None:
    from .validate import validate_records
    recs = load_records(args.input)
    rep = validate_records(recs, load_taxonomy(recs[0].taxonomy_version if recs else DEFAULT_VERSION), args.near_dup)
    write_jsonl(args.output, recs)
    report = rep.as_dict()
    Path(args.output).with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(report)
    if args.strict and rep.status.get("rejected"):
        sys.exit(1)


def cmd_romanize(args) -> None:
    """Romanized twins (random English mixed in) for a share of the accepted Devanagari records."""
    from .romanize import derive_roman_variants
    cfg = _cfg(args)
    recs = _accepted(args.input)
    kids, dropped = derive_roman_variants(recs, load_taxonomy(cfg.taxonomy_version), cfg.roman_share, args.seed)
    write_jsonl(args.output, recs + kids)
    n_dev, n_rom = sum(r.script == "deva" for r in recs), len(kids)
    _print({"devanagari": n_dev, "romanized": n_rom, "romanized_share_of_all": round(n_rom / max(1, n_dev + n_rom), 3),
            "english_levels": Counter(o.split(":")[1] for k in kids for o in k.derivation.ops if o.startswith("english_level")),
            "romanized_rejected_by_check": dict(dropped)})


def cmd_realstyle_generate(args) -> None:
    """Write the accepted clean scenarios again in app-review style (resumable)."""
    from .generation.realstyle import (accepted_scenarios, assign_scripts, load_style_examples, realstyle_system_prompt,
                                       sample_scenarios, user_message_with_scripts)
    from .generation.run import generate
    cfg, out = _cfg(args), Path(args.out)
    if cfg.realstyle is None:
        sys.exit(f"config {args.config} has no realstyle section")
    tax = load_taxonomy(cfg.taxonomy_version)
    scenarios = accepted_scenarios(Path(args.scenarios), load_records(args.clean))
    scripts = assign_scripts(scenarios, cfg.realstyle.deva_share, cfg.seed)
    if args.sample:
        scenarios = sample_scenarios(scenarios, args.sample, cfg.seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scripts.json").write_text(json.dumps({s.scenario_id: scripts[s.scenario_id] for s in scenarios}, indent=0),
                                      encoding="utf-8")
    print(f"realstyle: {len(scenarios)} scenarios, {sum(scripts[s.scenario_id] == 'deva' for s in scenarios)} Devanagari",
          file=sys.stderr)
    _confirm_paid(cfg, args, "generator", _pending(out / "raw.jsonl", "scenario_id", {s.scenario_id for s in scenarios}))
    system = realstyle_system_prompt(cfg.generator.prompt, tax, load_style_examples(cfg.realstyle.style_examples))
    _print(generate(cfg, tax, out, args.limit, args.batch, scenarios=scenarios, system=system,
                    user_message=user_message_with_scripts(scripts)))


def cmd_realstyle_records(args) -> None:
    from .generation.realstyle import build_realstyle_records
    from .scenarios import Scenario
    cfg, out = _cfg(args), Path(args.out)
    scenarios = [Scenario.model_validate(d) for d in read_jsonl(out / "scenarios.jsonl")]
    scripts = json.loads((out / "scripts.json").read_text(encoding="utf-8"))
    recs = build_realstyle_records(scenarios, list(read_jsonl(out / "raw.jsonl")), scripts, cfg.seed)
    n = write_jsonl(out / "records.jsonl", recs)
    _print({"records": n, "by_script": Counter(r.script for r in recs), "path": str(out / "records.jsonl")})


# ---------------------------------------------------------------- splits and export

def cmd_split(args) -> None:
    """Group-level split of the clean records, plus app-review-style and external records (see nepkev.split)."""
    from .split import build_splits
    clean = _accepted(args.clean)
    realstyle = _accepted(args.realstyle) if args.realstyle else []
    external = {}
    if args.external:
        for s in ("train", "dev", "calib"):
            if (p := Path(args.external) / f"{s}.jsonl").exists():
                external[s] = load_records(p)
    splits, stats = build_splits(clean, realstyle, external, seed=args.seed)
    out = Path(args.out)
    files = {}
    for s in SPLITS:
        p = out / f"{s}.jsonl"
        write_jsonl(p, splits[s])
        files[s] = {"path": p.name, "sha256": file_sha256(p)}
    inputs = [Path(args.clean), *([Path(args.realstyle)] if args.realstyle else []),
              *(Path(args.external) / f"{s}.jsonl" for s in external)]
    manifest = {"inputs": {p.as_posix(): file_sha256(p) for p in inputs}, "seed": args.seed,
                "taxonomy_version": clean[0].taxonomy_version if clean else None, "files": files, **stats,
                "upstream_commit": _upstream()["commit"]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shares = stats["train_label_shares"]
    _print({"splits": stats["splits"], "leakage_ok": stats["leakage"]["ok"], "realstyle_unplaced": len(stats["realstyle_unplaced"]),
            "train_label_share_range": {d: [min(v.values()), max(v.values())] for d, v in shares.items()}})
    if not stats["leakage"]["ok"] or stats["realstyle_unplaced"]:
        sys.exit(1)


def cmd_export_kev(args) -> None:
    from .export_kev import ExportOptions, export
    src, out = Path(args.splits), Path(args.out)
    manifest = {"splits_manifest_sha256": file_sha256(src / "manifest.json"), "files": {}}
    for s in SPLITS:
        recs = load_records(src / f"{s}.jsonl")
        if not recs:
            continue
        opts = ExportOptions(train=(s == "train"), seed=args.seed)
        n = export(recs, load_taxonomy(recs[0].taxonomy_version), out / f"{s}.jsonl", opts)
        manifest["files"][s] = {"records": n, "questions": 2 * n, "sha256": file_sha256(out / f"{s}.jsonl"),
                                "options": opts.__dict__}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _print(manifest)


# ---------------------------------------------------------------- training and evaluation

def cmd_baseline_rules(args) -> None:
    """Keyword-router predictions for every split, in the evaluation's prediction format."""
    from .baselines import rule_predict
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        recs = load_records(Path(args.splits) / f"{s}.jsonl")
        tax = load_taxonomy(recs[0].taxonomy_version)
        (out / f"{s}.json").write_text(json.dumps({r.record_id: rule_predict(r, tax) for r in recs}, ensure_ascii=False),
                                       encoding="utf-8")
    _print({"predictions": str(out)})


def cmd_remote_train(args) -> None:
    """Untouched baseline, fine-tuning, dev selection, calib temperature and test on one RunPod pod (always terminated)."""
    from .remote import load_remote_config, run_job
    cfg = load_remote_config(args.config)
    if not args.yes:
        print(f"This launches a {cfg.cloud_type} RunPod pod ({', '.join(cfg.gpu_types)}; refuses > ${cfg.max_cost_per_hr}/hr) "
              f"for at most {cfg.max_minutes} min. Re-run with --yes to proceed.", file=sys.stderr)
        sys.exit(2)
    sys.exit(run_job(Path(args.kev_dir), Path(args.out), cfg, name=f"nepkev-{Path(args.out).name}"))


def cmd_evaluate(args) -> None:
    """Compare the keyword rules, untouched Kev and adapted Kev on dev and test (routing thresholds from calib).

    External records are training data only and are left out of every evaluation split."""
    from .evaluate import calib_threshold, evaluate, kev_predictions, markdown_summary, paired_difference
    splits, kev, runs, out = Path(args.splits), Path(args.kev_dir), Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = {s: [r for r in load_records(splits / f"{s}.jsonl") if r.origin != "external"] for s in ("dev", "calib", "test")}

    def predictor(name: str, split: str) -> dict | None:
        if name == "rules":
            f = Path(args.rules) / f"{split}.json"
            return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        d = runs / f"{name}-{split}"
        return kev_predictions(d, kev / f"{split}.index.jsonl") if (d / "rows.json").exists() else None

    names = {"rules": "keyword rules", "base": "Kev-0.8B untouched", "selected": "Kev-0.8B adapted (calibrated)"}
    full: dict = {}
    md = ["# Evaluation report", "",
          "All evaluation records are generated: clean synthetic, its Romanized twins, and app-review-style messages.", ""]
    for split in ("dev", "test"):
        reports, preds_by = {}, {}
        for key, label in names.items():
            preds = predictor(key, split)
            if preds is None:
                continue
            cal = predictor(key, "calib")
            reports[label] = evaluate(recs[split], preds, route_threshold=calib_threshold(recs["calib"], cal) if cal else None)
            preds_by[key] = preds
        if "base" in preds_by and "selected" in preds_by:
            pa = [(r, preds_by["base"][r.record_id]) for r in recs[split] if r.record_id in preds_by["base"]]
            pb = [(r, preds_by["selected"][r.record_id]) for r in recs[split] if r.record_id in preds_by["selected"]]
            reports["_paired_adapted_minus_untouched"] = paired_difference(pa, pb)
        full[split] = reports
        md.append(markdown_summary({k: v for k, v in reports.items() if not k.startswith("_")}, split))
        if d := reports.get("_paired_adapted_minus_untouched"):
            md.append(f"Adapted minus untouched issue accuracy on {split}: {d['issue_accuracy_delta']:+.3f} "
                      f"[95% CI {d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}], paired, scenario-group bootstrap.\n")
    for extra in ("selection.json", "calibration.txt"):
        if (runs / extra).exists():
            md += [f"## {extra}", "", "```", (runs / extra).read_text(encoding="utf-8").strip()[-3000:], "```", ""]
    retention = [(name, json.loads((runs / name / "report.json").read_text(encoding="utf-8"))["clean"])
                 for name in ("base-transfer", "selected-transfer") if (runs / name / "report.json").exists()]
    if retention:
        md += ["## Retention: Kev's own English out-of-domain suite (transfer-v4 development)", "",
               "| run | n | accuracy | ECE | Brier | confident errors (p>=0.9 wrong) | coverage@5% err |", "|---|---|---|---|---|---|---|"]
        md += [f"| {n} | {c['n']} | {c['acc']:.3f} | {c['ece']:.3f} | {c['brier']:.3f} | {c['confident_error_rate']:.1%} | "
               f"{c['coverage_at_5pct_error']:.1%} |" for n, c in retention]
        md.append("")
    (out / "report.json").write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    _print({"report": str(out / "report.md")})


def cmd_push_hf(args) -> None:
    """Upload a run's selected checkpoint (with a model card) to a private Hugging Face repo."""
    from .hub import push
    from .remote import load_remote_config
    url = push(Path(args.runs), args.repo, load_remote_config(args.config), Path(args.report), args.data_note, args.code_url)
    _print({"repo": args.repo, "private": True, "commit": url})


def cmd_pipeline(args) -> None:
    """generate -> records -> verify -> validate -> romanize -> split -> export-kev, in one directory."""
    out = Path(args.out)
    steps = [
        ("generate", cmd_generate, {"limit": None, "batch": False}),
        ("records", cmd_records, {}),
        ("verify", cmd_verify, {}),
        ("validate", cmd_validate, {"input": out / "records.jsonl", "output": out / "validated.jsonl", "near_dup": 0.85,
                                    "strict": False}),
        ("romanize", cmd_romanize, {"input": out / "validated.jsonl", "output": out / "with_roman.jsonl"}),
        ("split", cmd_split, {"clean": out / "with_roman.jsonl", "realstyle": None, "external": None, "out": out / "splits"}),
        ("export-kev", cmd_export_kev, {"splits": out / "splits", "out": out / "kev"}),
    ]
    for step, fn, extra in steps:
        print(f"== {step}", flush=True)
        fn(argparse.Namespace(**{**vars(args), **extra}))


# ---------------------------------------------------------------- argument parsing

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="nepkev", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, config=False, out=False, paid=False):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        if config:
            p.add_argument("--config", required=True, help="configs/generation/<name>.yaml, or a path")
            p.add_argument("--dry-run", action="store_true", help="use the network-free dry-run provider")
        if out:
            p.add_argument("--out", required=True, help="run directory, e.g. data/pilot")
        if paid:
            p.add_argument("--yes", action="store_true", help="confirm a paid call")
        return p

    p = add("taxonomy", cmd_taxonomy, "check and summarize a taxonomy version")
    p.add_argument("--version", default=DEFAULT_VERSION)
    add("estimate", cmd_estimate, "token and cost estimate for a generation config", config=True)
    add("scenarios", cmd_scenarios, "sample scenarios only", config=True, out=True)
    p = add("generate", cmd_generate, "render scenarios into Devanagari messages (resumable)", config=True, out=True, paid=True)
    p.add_argument("--limit", type=int, help="render at most this many pending scenarios")
    p.add_argument("--batch", action="store_true", help="Anthropic Message Batches (half price); re-run to collect")
    add("records", cmd_records, "build canonical records from a run directory", config=True, out=True)
    add("verify", cmd_verify, "independent verifier labels (resumable)", config=True, out=True, paid=True)
    add("adjudicate", cmd_adjudicate, "stronger model's verdict on verifier disputes", config=True, out=True, paid=True)

    p = add("validate", cmd_validate, "deterministic checks; writes records with status + report")
    p.add_argument("--in", dest="input", required=True)
    p.add_argument("--out", dest="output", required=True)
    p.add_argument("--near-dup", type=float, default=0.85)
    p.add_argument("--strict", action="store_true", help="exit 1 if anything is rejected (fixtures/CI)")

    p = add("romanize", cmd_romanize, "derive Romanized twins of accepted Devanagari records", config=True)
    p.add_argument("--in", dest="input", required=True)
    p.add_argument("--out", dest="output", required=True)
    p.add_argument("--seed", type=int, default=0)

    p = add("realstyle-generate", cmd_realstyle_generate, "write accepted scenarios again in app-review style (resumable)",
            config=True, out=True, paid=True)
    p.add_argument("--scenarios", default="data/pilot/scenarios.jsonl")
    p.add_argument("--clean", default="data/pilot/with_roman.jsonl", help="clean records whose accepted scenarios are used")
    p.add_argument("--sample", type=int, help="render only a spread-out sample of this many scenarios")
    p.add_argument("--limit", type=int, help="render at most this many pending scenarios")
    p.add_argument("--batch", action="store_true", help="Anthropic Message Batches (half price); re-run to collect")
    add("realstyle-records", cmd_realstyle_records, "build canonical records from a realstyle run directory",
        config=True, out=True)

    p = add("split", cmd_split, "group-level train/dev/calib/test split and training mix, with leakage checks")
    p.add_argument("--clean", required=True, help="accepted clean records with their Romanized twins")
    p.add_argument("--realstyle", help="validated app-review-style records")
    p.add_argument("--external", help="directory of external labelled records: train/dev/calib.jsonl (training only)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)

    p = add("export-kev", cmd_export_kev, "write Kev training/eval JSONL from a split directory")
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)

    p = add("baseline-rules", cmd_baseline_rules, "keyword-router predictions for every split")
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)

    p = add("remote-train", cmd_remote_train, "fine-tune and evaluate on a RunPod GPU pod", paid=True)
    p.add_argument("--config", default="kev-0.8b", help="configs/remote/<name>.yaml")
    p.add_argument("--kev-dir", required=True)
    p.add_argument("--out", required=True)

    p = add("evaluate", cmd_evaluate, "compare rules / untouched Kev / adapted Kev on dev and test")
    p.add_argument("--splits", required=True)
    p.add_argument("--kev-dir", required=True, help="the Kev export (for its .index.jsonl files)")
    p.add_argument("--runs", required=True, help="downloaded remote runs directory")
    p.add_argument("--rules", required=True, help="baseline-rules output directory")
    p.add_argument("--out", required=True)

    p = add("push-hf", cmd_push_hf, "upload a run's selected checkpoint to a private Hugging Face repo")
    p.add_argument("--runs", required=True, help="downloaded remote runs directory (holds selected/)")
    p.add_argument("--repo", required=True, help="e.g. <user>/nepkev")
    p.add_argument("--config", default="kev-0.8b", help="the remote config the run used (for the pins)")
    p.add_argument("--report", required=True, help="the run's evaluation report.json (nepkev evaluate)")
    p.add_argument("--data-note", default="Labelled Nepali support messages, balanced across categories.",
                   help="one or two sentences on the training data, for the model card")
    p.add_argument("--code-url", help="link to the code, shown on the model card")

    p = add("pipeline", cmd_pipeline, "generate -> records -> verify -> validate -> romanize -> split -> export-kev",
            config=True, out=True, paid=True)
    p.add_argument("--seed", type=int, default=0)

    args = ap.parse_args(argv)
    load_dotenv()
    args.fn(args)


if __name__ == "__main__":
    main()
