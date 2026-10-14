import hashlib

import pytest

from nepkev.service.clarify import CHOOSE, UNCLEAR

from .conftest import rows

DEVA = "मेरो डेबिट कार्ड हरायो, ब्लक गरिदिनुस्"
ROMAN = "OTP aaudaina login garna sakdina"


def test_route_answers_and_logs_one_row(client, stub, engine):
    stub.canned[ROMAN] = ({"login_otp": 0.97}, 0.01)
    r = client.post("/route", json={"text": ROMAN, "channel": "wallet"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "login_otp" and body["confidence"] == pytest.approx(0.97)
    assert body["channel"] == "wallet" and body["needs_clarification"] is False
    assert body["clarifying_question"] is None and body["clarification_reason"] is None
    assert body["model_version"] == stub.version and body["latency_ms"] >= 0
    assert r.headers["X-Request-ID"] == body["request_id"] and len(body["request_id"]) == 32

    [row] = rows(engine)
    assert row["request_id"] == body["request_id"]
    assert row["text_hash"] == hashlib.sha256(ROMAN.encode()).hexdigest()
    assert row["text"] is None   # raw text is not stored by default
    assert (row["script"], row["channel"], row["intent"]) == ("romanized", "wallet", "login_otp")
    assert row["confidence"] == pytest.approx(0.97) and row["needs_clarification"] is False
    assert row["model_version"] == stub.version and row["idempotency_key"] is None
    assert row["response"] == body


def test_channel_defaults_to_the_configured_one(client, stub, settings):
    assert client.post("/route", json={"text": DEVA}).json()["channel"] == settings.default_channel
    client.post("/route", json={"text": DEVA, "channel": "bank"})
    assert stub.calls == [(DEVA, settings.default_channel), (DEVA, "bank")]


def test_script_is_logged_and_text_is_hashed_after_trimming(client, engine):
    client.post("/route", json={"text": f"  {DEVA}\n"})
    client.post("/route", json={"text": "mero card हरायो"})
    a, b = rows(engine)
    assert a["script"] == "devanagari" and a["text_hash"] == hashlib.sha256(DEVA.encode()).hexdigest()
    assert b["script"] == "mixed"


def test_raw_text_is_stored_only_when_enabled(settings, stub, engine):
    from fastapi.testclient import TestClient

    from nepkev.service.app import create_app

    with TestClient(create_app(settings.model_copy(update={"log_text": True}), router=stub)) as c:
        c.post("/route", json={"text": DEVA})
    assert rows(engine)[0]["text"] == DEVA


@pytest.mark.parametrize("payload", [
    {"text": ""},
    {"text": "   \n "},
    {"text": "क" * 2001},
    {"text": 42},
    {},
    {"text": ROMAN, "channel": "insurance"},
    {"text": ROMAN, "extra": 1},
])
def test_invalid_requests_are_422_and_never_reach_the_model(client, stub, engine, payload):
    assert client.post("/route", json=payload).status_code == 422
    assert stub.calls == [] and rows(engine) == []


@pytest.mark.parametrize("content_type", ["application/x-www-form-urlencoded", "text/plain"])
def test_body_that_is_not_declared_json_is_415(client, stub, content_type):
    r = client.post("/route", content=b'{"text": "hello"}', headers={"Content-Type": content_type})
    assert r.status_code == 415 and "application/json" in r.json()["detail"] and stub.calls == []


def test_json_content_type_with_charset_is_accepted(client):
    r = client.post("/route", content=b'{"text": "hello"}', headers={"Content-Type": "application/json; charset=utf-8"})
    assert r.status_code == 200


def test_text_of_exactly_2000_characters_is_accepted(client):
    assert client.post("/route", json={"text": "क" * 2000}).status_code == 200


def test_unclear_message_asks_in_the_customers_script(client, stub):
    stub.canned["sabai bigryo"] = ({"login_otp": 0.2}, 0.9)
    stub.canned["सबै बिग्रियो"] = ({"login_otp": 0.2}, 0.9)
    roman = client.post("/route", json={"text": "sabai bigryo", "channel": "shop"}).json()
    deva = client.post("/route", json={"text": "सबै बिग्रियो", "channel": "shop"}).json()
    for body in (roman, deva):
        assert body["needs_clarification"] is True and body["clarification_reason"] == "unclear_message"
    assert deva["clarifying_question"] == UNCLEAR.format(subject="अर्डर")
    q = roman["clarifying_question"]
    assert q.isascii() and "order" in q and q.startswith("tapainko samasya")


def test_low_confidence_asks_between_the_two_likeliest_named_categories(client, stub, settings):
    # `other` is likeliest but is not offered as a choice; the next two are
    stub.canned["paisa katyo"] = ({"other": 0.3, "payment_failed_debited": 0.25, "transfer_not_credited": 0.2}, 0.05)
    body = client.post("/route", json={"text": "paisa katyo", "channel": "wallet"}).json()
    assert body["intent"] == "other" and body["confidence"] < settings.min_confidence
    assert body["needs_clarification"] is True and body["clarification_reason"] == "low_confidence"
    assert "bhuktani fail bhayo" in body["clarifying_question"] and "pugena" in body["clarifying_question"]


def test_low_confidence_question_in_devanagari():
    from nepkev.router import Routing
    from nepkev.service.clarify import decide
    from nepkev.taxonomy import load_taxonomy

    tax = load_taxonomy()
    cats = tax.categories("bank")
    issue = {c: 0.05 for c in cats} | {"card_issue": 0.3, "account_access": 0.2}
    d = decide(Routing("bank", issue, 0.1), "कार्ड चलेन", tax, clarify_threshold=0.5, min_confidence=0.5)
    dom = tax.domain("bank")
    assert d.question == CHOOSE.format(first=dom.categories["card_issue"].ne, second=dom.categories["account_access"].ne)


def test_route_is_503_while_the_database_is_down(settings, stub):
    from fastapi.testclient import TestClient

    from nepkev.service.app import create_app

    from .conftest import UNREACHABLE_DB

    with TestClient(create_app(settings.model_copy(update={"database_url": UNREACHABLE_DB}), router=stub)) as c:
        r = c.post("/route", json={"text": ROMAN})
    assert r.status_code == 503 and r.json() == {"detail": "database unavailable"}


def test_health_is_always_ok(client):
    assert client.get("/health").json() == {"status": "ok"}
