from collections import Counter

from nepkev.cli import main
from nepkev.config import load_genconfig
from nepkev.generation.prompts import GENERATION_SCHEMA, generation_user_message, system_prompt
from nepkev.records import load_records
from nepkev.scenarios import generate_scenarios


def test_pilot_scenarios_are_balanced_and_deterministic(tax):
    cfg = load_genconfig("pilot")
    a, b = generate_scenarios(cfg, tax), generate_scenarios(cfg, tax)
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]
    assert len(a) == 3 * cfg.scenarios_per_domain
    for dom in tax.domains:
        hard = Counter(s.issue for s in a if s.domain == dom and s.issue and s.issue != tax.fallback)
        assert set(hard) == set(tax.categories(dom)) - {tax.fallback}
        assert max(hard.values()) - min(hard.values()) <= 0.35 * max(hard.values()), hard


def test_scenario_labels_follow_family(tax):
    for s in generate_scenarios(load_genconfig("pilot"), tax):
        if s.family in ("vague", "multi_issue:no_priority"):
            assert s.clarification_needed and s.issue is None and abs(sum(s.issue_target.values()) - 1) < 1e-3
        else:
            assert not s.clarification_needed and s.issue is not None
        if s.family == "vague":
            assert "underlying_issue" in s.withheld_facts and "underlying_issue" not in s.visible_facts
        if s.family == "multi_issue:stated_priority":
            assert s.stated_priority == s.issue and len(s.issues_all) == 2


def test_contrast_siblings_share_a_group(tax):
    sc = generate_scenarios(load_genconfig("pilot"), tax)
    groups = Counter(s.scenario_group_id for s in sc if s.family.startswith("contrast"))
    assert groups and set(groups.values()) == {2}


def test_generator_never_sees_category_ids_or_labels(tax):
    """Ids the renderer never sees are ids it cannot copy into a message."""
    system = system_prompt("generate_v2", tax)
    assert system == system_prompt("generate_v2", tax)                  # byte-stable for prompt caching
    sc = generate_scenarios(load_genconfig("pilot"), tax)
    families = {}
    for s_ in sc:
        families.setdefault(s_.family.split(":")[0], s_)
    user = generation_user_message(list(families.values()), tax)
    for dom in tax.domains.values():
        for cid in dom.categories:
            if "_" in cid:
                assert cid not in system and cid not in user, cid
    for s_ in families.values():
        if s_.family == "vague":
            assert s_.facts["underlying_issue"] not in user              # withheld facts never reach the renderer
    assert '"intended_category"' not in user and '"label' not in user
    assert GENERATION_SCHEMA["properties"]["items"]["items"]["required"] == ["scenario_id", "message", "evidence"]


def test_dry_run_pipeline_end_to_end(tmp_path):
    out = tmp_path / "run"
    main(["pipeline", "--config", "fixture-dryrun", "--out", str(out)])
    recs = load_records(out / "validated.jsonl")
    assert sum(r.validation.status == "accepted" for r in recs) > 0.7 * len(recs)
    assert all(r.source.generator == "dry-run" and "mock" in r.style_tags and r.script == "deva" for r in recs)
    both = load_records(out / "with_roman.jsonl")
    scripts = Counter(r.script for r in both)
    assert set(scripts) == {"deva", "roman"} and scripts["deva"] > scripts["roman"] > 0   # most data stays Devanagari
    import json
    manifest = json.loads((out / "splits" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["leakage"]["ok"]
    assert (out / "kev" / "train.jsonl").exists() and (out / "kev" / "train.index.jsonl").exists()


def test_realstyle_config_extends_pilot_and_overrides_only_its_keys():
    pilot, rs = load_genconfig("pilot"), load_genconfig("realstyle")
    assert rs.realstyle is not None and pilot.realstyle is None
    assert (rs.name, rs.seed, rs.family_mix, rs.style) == (pilot.name, pilot.seed, pilot.family_mix, pilot.style)
    assert rs.generator.prompt == "generate_realstyle_v1" and rs.generator.model == pilot.generator.model
    assert rs.verifier.prompt == rs.adjudicator.prompt == "verify_v2" and rs.verifier.model == pilot.verifier.model


def test_realstyle_scripts_follow_the_share_within_every_label(tax):
    from nepkev.generation.realstyle import assign_scripts
    sc = generate_scenarios(load_genconfig("pilot"), tax)
    scripts = assign_scripts(sc, 0.15, seed=7)
    assert scripts == assign_scripts(list(reversed(sc)), 0.15, seed=7)       # input order does not matter
    assert abs(sum(v == "deva" for v in scripts.values()) / len(sc) - 0.15) < 0.01
    by_label: dict = {}
    for s in sc:
        by_label.setdefault((s.domain, s.issue or s.family), []).append(scripts[s.scenario_id] == "deva")
    assert all(abs(sum(v) / len(v) - 0.15) < 0.12 for v in by_label.values() if len(v) >= 30)


def test_realstyle_prompt_carries_the_style_examples_and_scripts(tax):
    from nepkev.generation.realstyle import load_style_examples, realstyle_system_prompt, user_message_with_scripts
    cfg = load_genconfig("realstyle")
    examples = load_style_examples(cfg.realstyle.style_examples)
    system = realstyle_system_prompt(cfg.generator.prompt, tax, examples)
    assert len(examples) == 30 and "{examples}" not in system and "{taxonomy_plain}" not in system
    assert all(" ".join(e.split()) in system for e in examples)
    sc = generate_scenarios(load_genconfig("pilot"), tax)[:3]
    msg = user_message_with_scripts({s.scenario_id: "roman" for s in sc})(sc, tax)
    assert msg.count('"script": "roman"') == 3


def test_dotenv_fills_only_unset_variables(tmp_path, monkeypatch):
    from nepkev.cli import load_dotenv
    env = tmp_path / ".env"
    env.write_text('# comment\nNEPKEV_TEST_A="from file"\nNEPKEV_TEST_B=from file\n\n', encoding="utf-8")
    monkeypatch.delenv("NEPKEV_TEST_A", raising=False)
    monkeypatch.setenv("NEPKEV_TEST_B", "already set")
    load_dotenv(env)
    import os
    assert os.environ["NEPKEV_TEST_A"] == "from file" and os.environ["NEPKEV_TEST_B"] == "already set"
    monkeypatch.delenv("NEPKEV_TEST_A")
