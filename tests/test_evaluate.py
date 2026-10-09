import json
import math

import pytest

from nepkev.baselines import rule_predict
from nepkev.evaluate import (calib_threshold, clarification_metrics, coverage_at_error, evaluate, issue_metrics,
                             kev_predictions, paired_difference)


def _pred(issue: dict, clar: float) -> dict:
    return {"issue": issue, "clarification": clar}


def test_issue_metrics_known_values(fixtures):
    hard = [r for r in fixtures if r.labels.issue is not None][:4]
    preds = []
    for i, r in enumerate(hard):
        right = r.labels.issue
        wrong = "other" if right != "other" else "order_status"
        top = right if i < 3 else wrong                          # 3 of 4 correct
        preds.append((r, _pred({top: 0.8, (wrong if top == right else right): 0.2}, 0.1)))
    m = issue_metrics(preds)
    assert m["n"] == 4 and m["accuracy"] == pytest.approx(0.75)
    assert m["nll"] == pytest.approx((3 * -math.log(0.8) - math.log(0.2)) / 4)
    assert m["brier"] == pytest.approx((3 * (0.2 ** 2 + 0.2 ** 2) + (0.8 ** 2 + 0.8 ** 2)) / 4)
    assert m["ece"] == pytest.approx(abs(0.8 - 0.75))            # one bin holds every prediction


def test_soft_target_records_are_not_scored_for_issue(fixtures):
    soft = [r for r in fixtures if r.labels.issue is None]
    assert soft
    assert issue_metrics([(r, _pred({"other": 1.0}, 0.9)) for r in soft])["n"] == 0
    assert clarification_metrics([(r, _pred({"other": 1.0}, 0.9)) for r in soft])["recall"] == 1.0


def test_clarification_metrics():
    class R:   # minimal stand-in: only labels are read
        def __init__(self, y): self.labels = type("L", (), {"clarification_needed": y})()
    pairs = [(R(True), {"clarification": 0.9}), (R(True), {"clarification": 0.2}), (R(False), {"clarification": 0.7}),
             (R(False), {"clarification": 0.1})]
    m = clarification_metrics(pairs)
    assert (m["precision"], m["recall"], m["f1"]) == (0.5, 0.5, 0.5)


def test_coverage_at_error():
    cov, thr = coverage_at_error([0.9, 0.8, 0.7, 0.6], [True, True, False, True], max_error=0.25)
    assert cov == 1.0 and thr == 0.6
    cov, thr = coverage_at_error([0.9, 0.8, 0.7, 0.6], [True, True, False, True], max_error=0.0)
    assert cov == 0.5 and thr == 0.8


def test_bootstrap_ci_contains_point_and_paired_difference_is_zero_for_identical(fixtures, tax):
    preds = {r.record_id: rule_predict(r, tax) for r in fixtures}
    rep = evaluate(fixtures, preds)
    lo, hi = rep["ci95"]["issue_accuracy"]
    assert lo <= rep["issue"]["accuracy"] <= hi
    pairs = [(r, preds[r.record_id]) for r in fixtures]
    d = paired_difference(pairs, pairs)
    assert d["issue_accuracy_delta"] == 0 and d["ci95"] == [0, 0]
    assert calib_threshold(fixtures, preds) is None or 0 < calib_threshold(fixtures, preds) <= 1


def test_rule_router_falls_back_to_clarification(fixtures, tax):
    vague = next(r for r in fixtures if r.difficulty_family == "vague" and r.script == "deva")
    p = rule_predict(vague, tax)
    assert p["clarification"] >= 0.5 and abs(sum(p["issue"].values()) - 1) < 1e-9


def test_kev_rows_join_back_to_records(tmp_path):
    rows = [{"id": "custom/1", "question": "q1", "keys": ["a", "b"], "p": [0.3, 0.7], "label": 1},
            {"id": "custom/1", "question": "q2", "keys": ["false", "true"], "p": [0.9, 0.1], "label": 0}]
    (tmp_path / "rows.json").write_text(json.dumps(rows))
    (tmp_path / "x.index.jsonl").write_text('{"line": 0, "record_id": "r0"}\n{"line": 1, "record_id": "r1"}\n')
    assert kev_predictions(tmp_path, tmp_path / "x.index.jsonl") == {"r1": {"issue": {"a": 0.3, "b": 0.7}, "clarification": 0.1}}


def test_bias_checks_flag_other_swallowing_a_category(fixtures):
    from nepkev.evaluate import bias_checks, markdown_summary
    from nepkev.records import Labels, Record, Source
    base = next(r for r in fixtures if r.labels.issue is not None)

    def rec(i: int, issue: str, clar: bool = False) -> Record:
        return Record(record_id=f"r{i}", scenario_id=f"r{i}", scenario_group_id=f"r{i}", domain="shop",
                      taxonomy_version=base.taxonomy_version, message="x", script="deva", difficulty_family="t",
                      labels=Labels(issue=issue, clarification_needed=clar), source=Source(generator="t", provider="t"))
    # 10 order_status (4 predicted other), 10 other (all right), 10 cancellation (all right)
    recs = [rec(i, "order_status") for i in range(10)] + [rec(10 + i, "other") for i in range(10)] + \
           [rec(20 + i, "cancellation", clar=i < 3) for i in range(10)]
    preds = {}
    for i, r in enumerate(recs):
        top = "other" if i < 4 else r.labels.issue
        preds[r.record_id] = _pred({top: 0.9, "zzz": 0.1}, 0.9 if i >= 27 else 0.1)
    b = bias_checks([(r, preds[r.record_id]) for r in recs])
    ch, cats = b["checks"], b["per_category"]
    assert cats["shop/order_status"]["recall_into_other"] == pytest.approx(0.4)
    assert ch["other_share_ratio"]["value"] == pytest.approx(1.4) and ch["other_share_ratio"]["pass"]
    assert ch["other_precision"]["value"] == pytest.approx(10 / 14) and ch["other_precision"]["pass"]
    assert ch["recall"]["pass"] and ch["recall"]["worst"][0] == ("shop/order_status", 0.6)
    assert ch["share_ratio"]["pass"] and ch["share_ratio"]["checked"] == 3
    assert ch["clarify_rate"]["true"] == pytest.approx(0.1) and ch["clarify_rate"]["pass"]
    # one more leak tips `other` over both limits
    preds["r4"] = _pred({"other": 0.9}, 0.1)
    preds["r5"] = _pred({"other": 0.9}, 0.1)
    ch = bias_checks([(r, preds[r.record_id]) for r in recs])["checks"]
    assert not ch["other_share_ratio"]["pass"] and not ch["other_precision"]["pass"] and ch["share_ratio"]["over_limit"] == ["shop/other"]
    assert "FAIL" in markdown_summary({"m": evaluate(recs, preds)}, "test")
