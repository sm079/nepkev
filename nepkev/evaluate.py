"""One evaluation for every predictor (rule router, untouched Kev, adapted Kev) on identical records.

Predictions are per record: {"issue": {category: p}, "clarification": p}. Kev predictions come from upstream
`kev.benchmark` rows.json files (joined back to records through the export's .index.jsonl).

Issue metrics use only hard-labelled questions; soft-target questions (vague / multi-issue without priority) have no
correct category and are never scored for accuracy. ECE uses 15 equal-width confidence bins (max probability).
Confidence intervals are 95% percentile bootstraps that resample whole scenario groups (a message and its Romanized
variant move together). Routing coverage: the confidence threshold is chosen on one split (calib) and applied to another.

Bias checks ask whether some category (above all the fallback `other`) is predicted far more often than it occurs.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .records import Record, read_jsonl

BINS = 15
EPS = 1e-12


# ---------------------------------------------------------------- loading

def kev_predictions(bench_dir: Path, index_path: Path) -> dict[str, dict]:
    """rows.json from kev.benchmark --data <split>.jsonl -> {record_id: prediction}."""
    lines = {row["line"]: row["record_id"] for row in read_jsonl(index_path)}
    preds: dict[str, dict] = defaultdict(dict)
    for row in json.loads((bench_dir / "rows.json").read_text(encoding="utf-8")):
        rid = lines[int(str(row["id"]).rsplit("/", 1)[-1])]
        p = dict(zip(row["keys"], row["p"]))
        if row["question"] == "q1":
            preds[rid]["issue"] = p
        else:
            preds[rid]["clarification"] = p.get("true", p.get(True))
    return dict(preds)


# ---------------------------------------------------------------- metrics

def _ece(conf: list[float], correct: list[bool]) -> float:
    if not conf:
        return float("nan")
    bins: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for c, ok in zip(conf, correct):
        bins[min(BINS - 1, int(c * BINS))].append((c, ok))
    return sum(len(b) / len(conf) * abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b)) for b in bins.values())


def coverage_at_error(conf: list[float], correct: list[bool], max_error: float = 0.05) -> tuple[float, float | None]:
    """Largest share of most-confident items whose error rate stays <= max_error, and the confidence threshold."""
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    best, thr, errors = 0.0, None, 0
    for k, i in enumerate(order, 1):
        errors += not correct[i]
        if errors / k <= max_error:
            best, thr = k / len(conf), conf[i]
    return best, thr


def issue_metrics(pairs: list[tuple[Record, dict]]) -> dict:
    hard = [(r, p) for r, p in pairs if r.labels.issue is not None and "issue" in p]
    if not hard:
        return {"n": 0}
    pred = [max(p["issue"], key=p["issue"].get) for _, p in hard]
    gold = [r.labels.issue for r, _ in hard]
    correct = [a == b for a, b in zip(pred, gold)]
    conf = [max(p["issue"].values()) for _, p in hard]
    nll = sum(-math.log(max(p["issue"].get(r.labels.issue, 0.0), EPS)) for r, p in hard) / len(hard)
    brier = sum(sum((v - (k == r.labels.issue)) ** 2 for k, v in p["issue"].items()) for r, p in hard) / len(hard)
    per_cat = {}
    for key in sorted({(r.domain, g) for (r, _), g in zip(hard, gold)} | {(r.domain, q) for (r, _), q in zip(hard, pred)}):
        d, c = key
        tp = sum(1 for (r, _), g, q in zip(hard, gold, pred) if r.domain == d and g == c and q == c)
        fp = sum(1 for (r, _), g, q in zip(hard, gold, pred) if r.domain == d and g != c and q == c)
        fn = sum(1 for (r, _), g, q in zip(hard, gold, pred) if r.domain == d and g == c and q != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_cat[f"{d}/{c}"] = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                               "support": tp + fn}
    supported = [v["f1"] for v in per_cat.values() if v["support"]]
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (r, _), g, q in zip(hard, gold, pred):
        confusion[f"{r.domain}/{g}"][q] += 1
    cov, _ = coverage_at_error(conf, correct)
    return {"n": len(hard), "accuracy": sum(correct) / len(hard), "macro_f1": sum(supported) / len(supported),
            "nll": nll, "brier": brier, "ece": _ece(conf, correct), "coverage_at_5pct_error": cov,
            "per_category": per_cat, "confusion": {k: dict(v) for k, v in confusion.items()}}


def clarification_metrics(pairs: list[tuple[Record, dict]], threshold: float = 0.5) -> dict:
    xs = [(r.labels.clarification_needed, p["clarification"]) for r, p in pairs if p.get("clarification") is not None]
    if not xs:
        return {"n": 0}
    tp = sum(1 for y, p in xs if y and p >= threshold)
    fp = sum(1 for y, p in xs if not y and p >= threshold)
    fn = sum(1 for y, p in xs if y and p < threshold)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    conf = [max(p, 1 - p) for _, p in xs]
    correct = [(p >= threshold) == y for y, p in xs]
    return {"n": len(xs), "positives": sum(y for y, _ in xs), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "accuracy": sum(correct) / len(xs),
            "nll": sum(-math.log(max(p if y else 1 - p, EPS)) for y, p in xs) / len(xs),
            "brier": sum((p - y) ** 2 for y, p in xs) / len(xs), "ece": _ece(conf, correct)}


def _slices(pairs: list[tuple[Record, dict]]) -> dict:
    def length(r: Record) -> str:
        n = len(r.message.split())
        return "short(<=12w)" if n <= 12 else "medium(13-30w)" if n <= 30 else "long(>30w)"
    def family(r: Record) -> str:
        f = r.difficulty_family
        return f if f.startswith("multi") else f.split(":")[0]
    keys = {"domain": lambda r: r.domain, "origin": lambda r: r.origin, "family": family, "length": length}
    out = {}
    for name, f in keys.items():
        groups: dict[str, list] = defaultdict(list)
        for r, p in pairs:
            groups[f(r)].append((r, p))
        out[name] = {k: {"issue_accuracy": issue_metrics(v).get("accuracy"), "issue_n": issue_metrics(v)["n"],
                         "clarification_f1": clarification_metrics(v).get("f1"), "n": len(v)} for k, v in sorted(groups.items())}
    return out


def bootstrap(pairs: list[tuple[Record, dict]], samples: int = 1000, seed: int = 0) -> dict:
    groups: dict[str, list] = defaultdict(list)
    for r, p in pairs:
        groups[r.scenario_group_id].append((r, p))
    keys = list(groups)
    rng = random.Random(seed)
    stats = defaultdict(list)
    for _ in range(samples):
        sample = [x for k in rng.choices(keys, k=len(keys)) for x in groups[k]]
        im, cm = issue_metrics(sample), clarification_metrics(sample)
        stats["issue_accuracy"].append(im.get("accuracy", float("nan")))
        stats["issue_macro_f1"].append(im.get("macro_f1", float("nan")))
        stats["clarification_f1"].append(cm.get("f1", float("nan")))
    def ci(v: list[float]) -> list[float]:
        v = sorted(x for x in v if not math.isnan(x))
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]] if v else [float("nan")] * 2
    return {k: ci(v) for k, v in stats.items()} | {"unit": "scenario group", "samples": samples}


def paired_difference(pairs_a: list[tuple[Record, dict]], pairs_b: list[tuple[Record, dict]], samples: int = 1000,
                      seed: int = 0) -> dict:
    """Issue accuracy of b minus a on the same records, group bootstrap."""
    a, b = {r.record_id: p for r, p in pairs_a}, {r.record_id: p for r, p in pairs_b}
    recs = {r.record_id: r for r, _ in pairs_a if r.record_id in b}
    groups: dict[str, list[str]] = defaultdict(list)
    for rid, r in recs.items():
        groups[r.scenario_group_id].append(rid)

    def acc(ids: list[str], preds: dict) -> float:
        hard = [i for i in ids if recs[i].labels.issue is not None]
        return sum(max(preds[i]["issue"], key=preds[i]["issue"].get) == recs[i].labels.issue for i in hard) / max(1, len(hard))
    ids = list(recs)
    point = acc(ids, b) - acc(ids, a)
    rng, keys, diffs = random.Random(seed), list(groups), []
    for _ in range(samples):
        s = [i for k in rng.choices(keys, k=len(keys)) for i in groups[k]]
        diffs.append(acc(s, b) - acc(s, a))
    diffs.sort()
    return {"issue_accuracy_delta": point, "ci95": [diffs[int(0.025 * samples)], diffs[int(0.975 * samples) - 1]]}


ORIGINS = ("synthetic", "translated", "realstyle")


BIAS_LIMITS = {"max_share_ratio": 1.5, "min_recall": 0.5, "min_other_precision": 0.7, "max_clarify_gap": 0.05,
               "min_support": 10}


def bias_checks(pairs: list[tuple[Record, dict]], limits: dict = BIAS_LIMITS) -> dict:
    """Does the router over-predict some categories (above all `other`) at the expense of others?

    Shares are per domain (categories are domain-specific). The share and recall limits apply only to categories with
    at least `min_support` gold items; smaller ones are listed without a verdict, since one prediction moves them a lot.
    """
    hard = [(r, p) for r, p in pairs if r.labels.issue is not None and "issue" in p]
    out: dict = {"limits": limits, "checks": {}}
    if not hard:
        return out
    gold: dict[str, int] = defaultdict(int)
    pred: dict[str, int] = defaultdict(int)
    hits: dict[str, int] = defaultdict(int)
    into_other: dict[str, int] = defaultdict(int)
    per_domain: dict[str, int] = defaultdict(int)
    for r, p in hard:
        q = max(p["issue"], key=p["issue"].get)
        g, k = f"{r.domain}/{r.labels.issue}", f"{r.domain}/{q}"
        gold[g] += 1
        pred[k] += 1
        per_domain[r.domain] += 1
        hits[g] += q == r.labels.issue
        into_other[g] += q == "other" and r.labels.issue != "other"
    cats = {}
    for c in sorted(set(gold) | set(pred)):
        n = per_domain[c.split("/")[0]]
        ts, ps = gold[c] / n, pred[c] / n
        cats[c] = {"support": gold[c], "predicted": pred[c], "true_share": ts, "pred_share": ps,
                   "share_ratio": ps / ts if ts else (math.inf if ps else None),
                   "recall": hits[c] / gold[c] if gold[c] else None,
                   "recall_into_other": into_other[c] / gold[c] if gold[c] else None}
    out["per_category"] = cats
    big = {c: v for c, v in cats.items() if v["support"] >= limits["min_support"]}
    over = sorted((c for c, v in big.items() if v["share_ratio"] > limits["max_share_ratio"]), key=lambda c: -big[c]["share_ratio"])
    low = sorted((c for c, v in big.items() if v["recall"] < limits["min_recall"]), key=lambda c: big[c]["recall"])
    o_gold = sum(v for c, v in gold.items() if c.endswith("/other"))
    o_pred = sum(v for c, v in pred.items() if c.endswith("/other"))
    o_hit = sum(v for c, v in hits.items() if c.endswith("/other"))
    o_ratio = (o_pred / o_gold) if o_gold else (math.inf if o_pred else None)
    o_prec = o_hit / o_pred if o_pred else None
    checks = out["checks"]
    checks["share_ratio"] = {"pass": not over if big else None, "over_limit": over, "checked": len(big)}
    checks["other_share_ratio"] = {"pass": None if o_ratio is None else o_ratio <= limits["max_share_ratio"], "value": o_ratio,
                                   "gold": o_gold, "predicted": o_pred}
    checks["recall"] = {"pass": not low if big else None, "under_limit": low, "checked": len(big),
                        "worst": sorted(((c, v["recall"]) for c, v in big.items()), key=lambda x: x[1])[:5]}
    checks["other_precision"] = {"pass": None if not (o_pred or o_gold) else o_prec is not None and o_prec >= limits["min_other_precision"], "value": o_prec}
    clar = [(r.labels.clarification_needed, p["clarification"]) for r, p in pairs if p.get("clarification") is not None]
    if clar:
        t, q = sum(y for y, _ in clar) / len(clar), sum(c >= 0.5 for _, c in clar) / len(clar)
        checks["clarify_rate"] = {"pass": abs(q - t) <= limits["max_clarify_gap"], "true": t, "predicted": q}
    return out


def evaluate(records: list[Record], preds: dict[str, dict], route_threshold: float | None = None) -> dict:
    pairs = [(r, preds[r.record_id]) for r in records if r.record_id in preds]
    report = {"records": len(records), "predicted": len(pairs), "issue": issue_metrics(pairs),
              "clarification": clarification_metrics(pairs), "slices": _slices(pairs), "ci95": bootstrap(pairs),
              "bias": bias_checks(pairs)}
    if route_threshold is not None:
        hard = [(r, p) for r, p in pairs if r.labels.issue is not None]
        routed = [(r, p) for r, p in hard if max(p["issue"].values()) >= route_threshold]
        err = sum(max(p["issue"], key=p["issue"].get) != r.labels.issue for r, p in routed)
        report["routing_at_calib_threshold"] = {"threshold": route_threshold, "coverage": len(routed) / max(1, len(hard)),
                                                "error_rate": err / max(1, len(routed))}
    return report


def calib_threshold(records: list[Record], preds: dict[str, dict], max_error: float = 0.05) -> float | None:
    hard = [(r, preds[r.record_id]) for r in records if r.record_id in preds and r.labels.issue is not None]
    if not hard:
        return None
    conf = [max(p["issue"].values()) for _, p in hard]
    correct = [max(p["issue"], key=p["issue"].get) == r.labels.issue for r, p in hard]
    return coverage_at_error(conf, correct, max_error)[1]


def markdown_summary(reports: dict[str, dict], split: str) -> str:
    """Comparison table of several predictors' reports on one split."""
    lines = [f"### {split}", "",
             "| predictor | issue acc [95% CI] | macro-F1 | issue NLL | Brier | ECE | coverage@5% err | clarify F1 [95% CI] | routed at calib threshold |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, rep in reports.items():
        im, cm, ci = rep["issue"], rep["clarification"], rep["ci95"]
        rt = rep.get("routing_at_calib_threshold")
        rtxt = f"{rt['coverage']:.1%} at {rt['error_rate']:.1%} err" if rt else "n/a"
        lines.append(f"| {name} | {im['accuracy']:.3f} [{ci['issue_accuracy'][0]:.3f}, {ci['issue_accuracy'][1]:.3f}] | "
                     f"{im['macro_f1']:.3f} | {im['nll']:.3f} | {im['brier']:.3f} | {im['ece']:.3f} | {im['coverage_at_5pct_error']:.1%} | "
                     f"{cm['f1']:.3f} [{ci['clarification_f1'][0]:.3f}, {ci['clarification_f1'][1]:.3f}] | {rtxt} |")
    lines.append("")
    lines.append("| predictor | " + " | ".join(f"{k} acc" for k in ORIGINS) + " | " +
                 " | ".join(f"{k} acc" for k in ("shop", "wallet", "bank")) + " |")
    lines.append("|---" * (len(ORIGINS) + 4) + "|")
    for name, rep in reports.items():
        s = rep["slices"]
        cells = [s["origin"].get(k, {}).get("issue_accuracy") for k in ORIGINS] + \
                [s["domain"].get(k, {}).get("issue_accuracy") for k in ("shop", "wallet", "bank")]
        lines.append(f"| {name} | " + " | ".join("n/a" if c is None else f"{c:.3f}" for c in cells) + " |")
    return "\n".join(lines) + "\n" + bias_summary(reports)


def bias_summary(reports: dict[str, dict]) -> str:
    """Pass/fail table for bias_checks, plus where `other` swallows each category, per predictor."""
    reps = {n: r["bias"] for n, r in reports.items() if r.get("bias", {}).get("checks")}
    if not reps:
        return ""
    lim = next(iter(reps.values()))["limits"]
    def mark(c: dict | None, txt: str) -> str:
        return "n/a" if c is None or c["pass"] is None else f"{'PASS' if c['pass'] else 'FAIL'} {txt}"
    def num(x: float | None, fmt: str = ".2f") -> str:
        return "n/a" if x is None else "inf" if math.isinf(x) else format(x, fmt)
    lines = ["", f"Bias checks (limits fixed before results: per-category predicted share <= {lim['max_share_ratio']}x true share, "
             f"recall >= {lim['min_recall']}, both only for categories with >= {lim['min_support']} gold items; "
             f"`other` share <= {lim['max_share_ratio']}x, `other` precision >= {lim['min_other_precision']}; "
             f"clarification rate within {lim['max_clarify_gap']:.0%} of true):", "",
             "| predictor | share ratio | `other` share ratio | recall | `other` precision | clarify rate (pred / true) |",
             "|---|---|---|---|---|---|"]
    for name, b in reps.items():
        ch = b["checks"]
        sr, orat, rc, op, cr = (ch.get(k) for k in ("share_ratio", "other_share_ratio", "recall", "other_precision", "clarify_rate"))
        over = sr["over_limit"]
        sr_txt = f"({len(over)}/{sr['checked']} over: {', '.join(over[:3])})" if over else f"({sr['checked']} checked)"
        worst = "worst " + ", ".join(f"{c} {v:.2f}" for c, v in rc["worst"][:3])
        cr_txt = f"{cr['predicted']:.1%} / {cr['true']:.1%}" if cr else ""
        cells = [mark(sr, sr_txt), mark(orat, num(orat["value"]) + "x"), mark(rc, worst), mark(op, num(op["value"])), mark(cr, cr_txt)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for name, b in reps.items():
        leaks = sorted(((c, v) for c, v in b["per_category"].items() if v["recall_into_other"]), key=lambda x: -x[1]["recall_into_other"])
        if leaks:
            lines.append(f"\n{name}, share of each category predicted as `other` (gold count): " +
                         ", ".join(f"{c} {v['recall_into_other']:.0%} ({v['support']})" for c, v in leaks[:10]))
    return "\n".join(lines) + "\n"
