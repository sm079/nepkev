import pytest

from nepkev.export_kev import ExportOptions, to_kev
from nepkev.router import KevRouter, Routing, serving_request


def test_serving_request_is_the_evaluation_export_without_labels(fixtures, tax):
    for rec in fixtures:
        exported = to_kev(rec, tax, ExportOptions(train=False))
        for q in exported["questions"].values():
            q.pop("label")
            q.pop("target", None)
        assert serving_request(rec.message, rec.domain, tax) == exported


def test_routing_picks_the_likeliest_category():
    r = Routing("shop", {"order_status": 0.2, "cancellation": 0.7, "other": 0.1}, clarification=0.3)
    assert (r.intent, r.confidence) == ("cancellation", 0.7)
    assert [c for c, _ in r.ranked()] == ["cancellation", "order_status", "other"]


def test_kev_router_validates_before_loading():
    with pytest.raises(ValueError, match="dtype"):
        KevRouter("sm079/nepkev", dtype="int4")
    r = KevRouter("sm079/nepkev", revision="8daf1b3bc570ea94e13ead9f8f0a2f2092bb3a3c")
    assert r.version == "sm079/nepkev@8daf1b3" and not r.loaded
    with pytest.raises(RuntimeError, match="before load"):
        r.route("hello", "shop")


@pytest.mark.slow
def test_real_checkpoint_routes_clear_messages():
    """Downloads and runs the published checkpoint on CPU (minutes on a cold cache): uv run pytest -m slow"""
    pytest.importorskip("kev.checkpoint", reason="needs uv sync --extra kev")
    pytest.importorskip("pydantic_settings", reason="needs uv sync --extra service")
    from nepkev.service.settings import Settings

    s = Settings(_env_file=None)
    router = KevRouter(s.model, revision=s.model_revision)
    router.load()
    cases = [("OTP aaudaina login garna sakdina", "wallet", "login_otp"),
             ("मेरो डेबिट कार्ड हरायो, ब्लक गरिदिनुस्", "bank", "card_issue"),
             ("sathi lai 2000 pathako, mero khata bata katyo tara usle paayena", "wallet", "transfer_not_credited")]
    for text, domain, intent in cases:
        r = router.route(text, domain)
        assert r.intent == intent and r.confidence > 0.5 and r.clarification < 0.5
        assert sum(r.issue.values()) == pytest.approx(1, abs=1e-4)
    assert router.route("sabai kura bigryo", "wallet").clarification > 0.5
