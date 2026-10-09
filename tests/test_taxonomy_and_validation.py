import pytest

from nepkev.records import Labels, Record
from nepkev.validate import amounts_in, check_record, has_negation, nepali_hindi_scores, validate_records


def test_taxonomy_integrity(tax):
    assert tax.version == "demo-1"
    for name, dom in tax.domains.items():
        assert tax.fallback in dom.categories
        assert len(dom.categories) >= 10
        for cid in dom.categories:
            assert cid.isidentifier() and cid == cid.lower()
    # digital access and account access are separate, explicitly contrasted categories
    bank = tax.domain("bank")
    assert {"account_access", "digital_banking_login_otp"} <= set(bank.categories)
    assert any({a, b} == {"account_access", "digital_banking_login_otp"} for a, b, _ in bank.contrasts)


def test_question_wording_names_domain(tax):
    assert "online shop" in tax.issue_instructions("shop")
    assert "bank" in tax.clarification_instructions("bank", 2)


def test_all_fixtures_validate(fixtures, tax):
    rep = validate_records(fixtures, tax)
    bad = {r.record_id: r.validation.checks for r in fixtures if r.validation.status != "accepted"}
    assert not bad, bad
    assert rep.status["accepted"] == len(fixtures)


@pytest.mark.parametrize("text,amount", [
    ("रु 1299 दुई पटक गा'को छ", 1299), ("१०,००० निकालेको", 10000), ("1.5k send vaxa", 1500),
    ("साथीलाई ५ हजार पठाएँ", 5000), ("5 hajar pathaye", 5000), ("Rs.840 katyo", 840), ("2 lakh transfer", 200000),
])
def test_amounts_in_every_format(text, amount):
    assert amount in amounts_in(text)


def test_negation_markers():
    assert has_negation("पैसा चाहिँ काटेको छैन")
    assert has_negation("card ma kei problem xaina")
    assert has_negation("maile garkai navako transaction")
    assert not has_negation("payment fail vayo, paisa katyo")


def test_hindi_and_english_rejected():
    ne, hi = nepali_hindi_scores("मेरे खाते से पैसे कट गए लेकिन प्राप्तकर्ता को अभी तक नहीं मिले।", "deva")
    assert hi > ne
    ne, hi = nepali_hindi_scores("mera paisa kat gaya lekin receiver ko nahi mila", "roman")
    assert hi > ne
    assert nepali_hindi_scores("my money was deducted but not received", "roman")[0] == 0


def _rec(fixtures, **kw):
    base = next(r for r in fixtures if r.record_id == "wallet-05-roman")   # payment_failed_debited, Rs 840
    return Record.model_validate({**base.model_dump(), **kw})


def test_missing_amount_is_rejected(fixtures, tax):
    r = _rec(fixtures, message="bijuli ko bill tirda fail dekhayo tara wallet bata paisa katyo, bill chai tireko chaina")
    assert "missing_evidence" in {f.reason for f in check_record(r, tax)}


def test_label_leakage_is_rejected(fixtures, tax):
    r = _rec(fixtures, message="payment_failed_debited vayo, Rs 840 katyo tara bill tireko chaina")
    assert "label_leakage" in {f.reason for f in check_record(r, tax)}


def test_messages_are_single_script(fixtures, tax):
    assert {r.script for r in fixtures} == {"deva", "roman"}
    mixed = next(r for r in fixtures if r.record_id == "wallet-05-deva").model_copy(
        update={"message": "बिजुलीको बिल तिर्दा transaction fail देखायो तर वालेटबाट रु. 840 काट्यो, बिल चाहिँ तिरेको छैन"})
    assert "script_mismatch" in {f.reason for f in check_record(mixed, tax)}


def test_review_style_devanagari_may_keep_english_in_latin(fixtures, tax):
    base = next(r for r in fixtures if r.record_id == "wallet-05-deva")
    msg = "बिजुलीको बिल तिर्दा transaction fail देखायो तर वालेटबाट रु. 840 काट्यो, बिल चाहिँ तिरेको छैन"
    styled = base.model_copy(update={"message": msg, "style_tags": [*base.style_tags, "realstyle"]})
    assert "script_mismatch" not in {f.reason for f in check_record(styled, tax)}
    roman = styled.model_copy(update={"script": "roman"})
    assert "script_mismatch" in {f.reason for f in check_record(roman, tax)}


def test_script_mismatch_is_rejected(fixtures, tax):
    r = _rec(fixtures, message="बिजुलीको बिल तिर्दा fail देखायो तर wallet बाट रु. 840 काट्यो, बिल चाहिँ तिरेको छैन")
    assert "script_mismatch" in {f.reason for f in check_record(r, tax)}


def test_vague_family_cannot_have_hard_label(fixtures, tax):
    r = _rec(fixtures, difficulty_family="vague")
    reasons = {f.reason for f in check_record(r, tax)}
    assert "label_family_mismatch" in reasons


def test_labels_hard_and_soft_are_exclusive():
    with pytest.raises(ValueError):
        Labels(issue="login_otp", issue_target={"login_otp": 1.0}, clarification_needed=False)
    with pytest.raises(ValueError):
        Labels(issue=None, issue_target={"a": 0.3, "b": 0.3}, clarification_needed=True)


def test_cross_group_duplicate_rejected_but_siblings_kept(fixtures, tax):
    dup = fixtures[0].model_copy(deep=True, update={"record_id": "dup-1", "scenario_id": "dup", "scenario_group_id": "dup"})
    sibling = fixtures[0].model_copy(deep=True, update={"record_id": "sib-1"})       # same group: allowed
    recs = [r.model_copy(deep=True) for r in fixtures] + [dup, sibling]
    validate_records(recs, tax)
    by_id = {r.record_id: r for r in recs}
    assert by_id["dup-1"].validation.rejection_reason == "exact_duplicate"
    assert by_id["sib-1"].validation.status == "accepted"


def test_validation_keeps_the_verifier_verdict(fixtures, tax):
    recs = [r.model_copy(deep=True) for r in fixtures]
    recs[0].validation.checks["verifier"] = {"model": "m", "issue": "other", "clarification_needed": False,
                                             "issues_mentioned": [], "language_ok": True, "agrees": False}
    validate_records(recs, tax)
    assert recs[0].validation.checks["verifier"]["agrees"] is False
    assert recs[0].validation.status == "queued"


def test_fact_values_contain_no_latin_acronyms():
    """Visible facts are shown to a Devanagari-only generator; Latin acronyms in them invite Latin text."""
    import re
    from nepkev.scenarios import BANK_PRODUCTS, BILLERS, PRODUCTS
    assert not [p for p in PRODUCTS + BANK_PRODUCTS + BILLERS if re.search(r"\b[A-Z]{2,}\b", p)]


def test_adjudicator_settles_verifier_disputes(fixtures, tax):
    recs = [r.model_copy(deep=True) for r in fixtures[:3]]
    dispute = {"model": "m", "issue": None, "clarification_needed": True, "issues_mentioned": [], "language_ok": True, "agrees": False}
    for r in recs:
        r.validation.checks["verifier"] = dict(dispute)
    recs[0].validation.checks["adjudicator"] = {**dispute, "agrees": True}
    recs[1].validation.checks["adjudicator"] = {**dispute, "agrees": False}
    validate_records(recs, tax)
    assert [r.validation.status for r in recs] == ["accepted", "rejected", "queued"]
    assert recs[1].validation.rejection_reason == "label_disputed"
