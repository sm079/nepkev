import random
from collections import Counter

import pytest

from nepkev.romanize import derive_roman_variants, load_romanize, roman_word, romanize, transliterate
from nepkev.validate import amounts_in, check_record, detect_script, has_negation


@pytest.mark.parametrize("deva,roman", [
    ("घर", "ghar"), ("पसल", "pasal"), ("दिन", "din"), ("छ", "chha"), ("छैन", "chhaina"), ("पुगेन", "pugena"),
    ("भएन", "bhaena"), ("होइन", "hoina"), ("पाइन", "paina"), ("सकिन", "sakina"), ("गर्छ", "garchha"),
    ("देखाउँछ", "dekhauchha"), ("हुँदैन", "hudaina"), ("एक", "ek"), ("समय", "samaya"), ("तर", "tara"),
    ("किन", "kina"), ("गरें", "gare"), ("मेरो", "mero"), ("पैसा", "paisa"), ("काटियो", "katiyo"),
    ("सेट", "set"), ("गरेर", "garera"), ("भएर", "bhaera"), ("व्यवहार", "byabahar"), ("कोटेश्वर", "koteshwar"),
])
def test_core_romanization(deva, roman):
    assert roman_word(deva) == roman


@pytest.mark.parametrize("deva,roman", [("रङको", "rang ko"), ("उसको", "us ko"), ("गरेको", "gareko"), ("मगाको", "magako"),
                                        ("सब्सक्रिप्सनको", "subscription ko")])
def test_postposition_after_bare_consonant(deva, roman):
    assert transliterate(deva) == roman


def test_loanwords_postpositions_digits():
    assert transliterate("अर्डरको पैसा") == "order ko paisa"
    assert transliterate("ओटीपी आएन") == "OTP aaena"
    assert transliterate("वालेटबाट रु ५,००० काटियो।") == "wallet bata Rs 5,000 katiyo."
    assert transliterate("अर्डर नं ४८२१७") == "order no 48217"
    assert transliterate("मेरो खाताबाट") == "mero khata bata"


def test_typist_profile_is_consistent_within_a_message():
    cfg = load_romanize()
    profile = {**{k: next(iter(v)) for k, v in cfg.profile_options.items()}, "छ": "x", "भ": "v"}
    assert transliterate("पैसा भयो तर छैन, अझै छ", cfg, profile) == "paisa vayo tara xaina, ajhai xa"


def test_output_is_always_pure_latin_and_keeps_meaning():
    text = "कार्डमा केही समस्या छैन, एटीएमबाट 5000 निकाल्दा पैसा निस्केन तर खाताबाट चाहिँ काट्यो"
    outs = set()
    for i in range(60):
        out, ops = romanize(text, random.Random(i))
        assert detect_script(out) == "roman", out
        assert 5000 in amounts_in(out) and has_negation(out), out
        outs.add(out)
    assert len(outs) > 30                                               # many different typists


def test_english_is_mixed_in_at_varying_levels():
    text = "तर समस्या अझै छ, खाताबाट पैसा काटियो तर सामान आएन"
    levels, english = Counter(), Counter()
    for i in range(200):
        out, ops = romanize(text, random.Random(i))
        levels.update(o.split(":")[1] for o in ops if o.startswith("english_level"))
        english["but" in out.split() or "problem" in out.split()] += 1
    assert set(levels) == {"none", "light", "heavy"}
    assert english[True] and english[False]                             # sometimes English, sometimes not
    none_runs = [romanize(text, random.Random(i)) for i in range(200)]
    for out, ops in none_runs:
        if "english_level:none" in ops:
            assert not any(o.startswith("english:") for o in ops)


def test_contractions_happen_before_romanizing():
    outs = {romanize("पैसा पुगेको भएको छैन, के भयो?", random.Random(i))[0] for i in range(60)}
    assert any("bho" in o or "vo" in o for o in outs) and any("bhayo" in o or "vayo" in o for o in outs)


def test_every_devanagari_fixture_romanizes_cleanly(fixtures, tax):
    deva = [r for r in fixtures if r.script == "deva"]
    kids, dropped = derive_roman_variants(deva, tax, share=1.0, seed=3)
    assert not dropped and len(kids) == len(deva), dropped
    parents = {r.record_id: r for r in deva}
    for k in kids:
        p = parents[k.derivation.parent_record_id]
        assert k.scenario_group_id == p.scenario_group_id and k.labels == p.labels
        assert k.script == "roman" and k.record_id == f"{p.scenario_id}-roman" and k.source.generator == "romanize"
        assert not check_record(k, tax)


def test_roman_share_and_determinism(fixtures, tax):
    deva = [r for r in fixtures if r.script == "deva"]
    a, _ = derive_roman_variants(deva, tax, share=0.5, seed=0)
    b, _ = derive_roman_variants(deva, tax, share=0.5, seed=0)
    c, _ = derive_roman_variants(deva, tax, share=0.5, seed=1)
    assert [k.message for k in a] == [k.message for k in b] != [k.message for k in c]
    assert 0.2 * len(deva) < len(a) < 0.8 * len(deva)
    assert derive_roman_variants(deva, tax, share=0.0)[0] == []


def test_yaml_words_stay_strings():
    cfg = load_romanize()
    assert all(isinstance(v, str) for v in {**cfg.loanwords, **cfg.english}.values())
