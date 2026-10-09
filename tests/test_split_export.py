import json

import pytest

from nepkev.export_kev import CLAR_QID, ISSUE_QID, ExportOptions, export, to_kev
from nepkev.romanize import derive_roman_variants
from nepkev.records import Record
from nepkev.split import SPLITS, assign, build_splits, label_shares, leakage_report


# ---------------------------------------------------------------- split

def test_split_keeps_groups_and_texts_together(fixtures, tax):
    recs = fixtures + derive_roman_variants(fixtures, tax, share=1.0, seed=1)[0]
    # a counterfactual sibling in another group with the same text must be pulled into the same split
    clone = recs[0].model_copy(update={"record_id": "clone", "scenario_id": "clone", "scenario_group_id": "clone-group"})
    recs.append(clone)
    res = assign(recs, seed=3)
    assert set(res.assignment.values()) <= set(SPLITS)
    rep = leakage_report(recs, res.assignment)
    assert rep["ok"], rep
    assert res.assignment["clone"] == res.assignment[recs[0].record_id]


def test_split_is_deterministic(fixtures):
    assert assign(fixtures, seed=9).assignment == assign(fixtures, seed=9).assignment


def test_split_fractions_must_sum_to_one(fixtures):
    with pytest.raises(ValueError):
        assign(fixtures, fractions={"train": 0.9, "dev": 0.1, "calib": 0.1, "test": 0.1})


def _twin(r: Record, suffix: str, **update) -> Record:
    d = r.model_dump()
    d.update({"record_id": r.record_id + suffix, "message": r.message + " " + suffix, "content_hash": "", **update})
    return Record.model_validate(d)


def test_mix_puts_realstyle_in_its_groups_split_and_external_where_given(fixtures, tax):
    clean = fixtures + derive_roman_variants(fixtures, tax, share=1.0, seed=1)[0]
    rs = [_twin(r, "yaar", style_tags=[*r.style_tags, "realstyle"]) for r in fixtures]
    ext = [_twin(r, "ext", script=None, source={"generator": "external"}, scenario_group_id="ext-" + r.record_id)
           for r in fixtures[:3]]
    base = assign(clean, seed=0).assignment
    out, stats = build_splits(clean, rs, {"train": ext}, seed=0)
    assert stats["leakage"]["ok"] and not stats["realstyle_unplaced"]
    assert all(out[s].count(r) for r in clean for s in SPLITS if base[r.record_id] == s)
    split_of = {r.scenario_group_id: s for s in SPLITS for r in out[s] if r.origin == "synthetic"}
    assert all(split_of[r.scenario_group_id] == s for s in SPLITS for r in out[s] if r.origin == "realstyle")
    assert ext == [r for r in out["train"] if r.origin == "external"]
    # within a split: clean records first, then external, then realstyle (training files depend on this order)
    rank = {"synthetic": 0, "translated": 0, "external": 1, "realstyle": 2}
    for s in SPLITS:
        ranks = [rank[r.origin] for r in out[s]]
        assert ranks == sorted(ranks)
    with pytest.raises(ValueError):
        build_splits(clean, rs, {"test": ext})


def test_label_shares_sum_to_one_per_domain(fixtures):
    for shares in label_shares(fixtures).values():
        assert abs(sum(shares.values()) - 1) < 1e-3


# ---------------------------------------------------------------- Kev export

def test_export_labels_match_criteria(fixtures, tax):
    for train in (False, True):
        for r in fixtures:
            row = to_kev(r, tax, ExportOptions(train=train, seed=2))
            q1, q2 = row["questions"][ISSUE_QID], row["questions"][CLAR_QID]
            assert q1["type"] == "choice" and q1["label"] in q1["criteria"]
            if "target" in q1:
                assert set(q1["target"]) <= set(q1["criteria"]) and abs(sum(q1["target"].values()) - 1) < 1e-3
            assert q2["type"] == "noul" and isinstance(q2["label"], bool) and set(q2["criteria"]) == {"true", "false"}


def test_metadata_never_enters_model_input(fixtures, tax):
    for r in fixtures:
        row = to_kev(r, tax, ExportOptions(train=True))
        assert row["state"] == r.message
        blob = json.dumps(row, ensure_ascii=False)
        for leak in (r.record_id, r.scenario_group_id, r.scenario_id + "-", r.note):
            if leak:
                assert leak not in blob
        assert set(row) == {"state", "questions"} and set(row["questions"]) == {ISSUE_QID, CLAR_QID}


def test_eval_export_uses_serving_contract(fixtures, tax):
    for r in fixtures:
        q1 = to_kev(r, tax, ExportOptions(train=False))["questions"][ISSUE_QID]
        assert q1["instructions"] == tax.issue_instructions(r.domain)
        assert list(q1["criteria"]) == tax.categories(r.domain)


def test_gold_and_fallback_are_never_dropped(fixtures, tax):
    hard = [r for r in fixtures if r.labels.issue is not None]
    opts = ExportOptions(train=True, p_drop_others=1.0)
    for r in hard:
        q1 = to_kev(r, tax, opts)["questions"][ISSUE_QID]
        assert {r.labels.issue, tax.fallback} <= set(q1["criteria"]) and q1["label"] == r.labels.issue
        assert len(q1["criteria"]) < len(tax.categories(r.domain))


def test_soft_targets_keep_full_candidate_set(fixtures, tax):
    soft = [r for r in fixtures if r.labels.issue_target]
    assert soft
    for r in soft:
        q1 = to_kev(r, tax, ExportOptions(train=True, p_drop_others=1.0))["questions"][ISSUE_QID]
        assert set(q1["criteria"]) == set(tax.categories(r.domain)) and q1["target"] == r.labels.issue_target


def test_a_records_export_does_not_depend_on_the_others(fixtures, tax, tmp_path):
    opts = ExportOptions(train=True, seed=4)
    export(fixtures, tax, tmp_path / "all.jsonl", opts)
    export(fixtures[1::2], tax, tmp_path / "half.jsonl", opts)
    every = (tmp_path / "all.jsonl").read_text(encoding="utf-8").splitlines()
    assert (tmp_path / "half.jsonl").read_text(encoding="utf-8").splitlines() == every[1::2]


@pytest.mark.kev
def test_export_loads_in_upstream_kev(fixtures, tax, tmp_path):
    """The exported file goes through Kev's own loader and fits its training context (pinned upstream)."""
    kev_data = pytest.importorskip("kev.data")
    kev_model = pytest.importorskip("kev.model")
    for train in (False, True):
        out = tmp_path / f"{train}.jsonl"
        n = export(fixtures, tax, out, ExportOptions(train=train))
        reqs = kev_data.load_records(str(out))
        recs = [kev_data.materialize(r) for r in reqs]
        assert len(recs) == n == len(fixtures)
        tok = kev_model.load_tokenizer("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
        assert all(kev_model.fits(r, tok, **kev_model.training_context()) for r in recs)
        soft = [q for r in recs for q in r["questions"] if q.get("target")]
        assert soft and all(abs(sum(q["target"]) - 1) < 1e-6 for q in soft)
