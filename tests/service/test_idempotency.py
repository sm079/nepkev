from sqlalchemy import text

from nepkev.service.app import REPLAY_HEADER

from .conftest import rows

BODY = {"text": "sathi lai 2000 pathako tara pugena", "channel": "wallet"}


def post(client, body=BODY, key="key-1"):
    return client.post("/route", json=body, headers={"Idempotency-Key": key})


def test_same_key_and_body_replays_the_stored_response(client, stub, engine):
    first = post(client)
    second = post(client)
    assert first.status_code == second.status_code == 200
    assert second.json() == first.json()
    assert REPLAY_HEADER not in first.headers and second.headers[REPLAY_HEADER] == "true"
    assert second.headers["X-Request-ID"] == first.json()["request_id"]
    assert len(stub.calls) == 1   # the replay never runs the model
    [row] = rows(engine)
    assert row["idempotency_key"] == "key-1" and row["request_hash"]


def test_same_key_with_a_different_body_is_rejected(client, stub, engine):
    post(client)
    r = post(client, {**BODY, "channel": "bank"})
    assert r.status_code == 422 and "different request body" in r.json()["detail"]
    assert len(stub.calls) == 1 and len(rows(engine)) == 1


def test_whitespace_around_the_text_does_not_change_the_body(client, stub):
    post(client)
    assert post(client, {**BODY, "text": f" {BODY['text']} "}).headers[REPLAY_HEADER] == "true"
    assert len(stub.calls) == 1


def test_different_keys_are_separate_requests(client, stub, engine):
    a, b = post(client, key="a").json(), post(client, key="b").json()
    assert a["request_id"] != b["request_id"]
    assert len(stub.calls) == 2 and [r["idempotency_key"] for r in rows(engine)] == ["a", "b"]


def test_requests_without_a_key_are_never_replayed(client, stub, engine):
    client.post("/route", json=BODY)
    client.post("/route", json=BODY)
    assert len(stub.calls) == 2 and len(rows(engine)) == 2


def test_a_concurrent_request_that_commits_first_wins(client, stub, engine):
    """Both requests miss the stored key; the one committing second answers with the first one's response."""
    def commit_first():
        stub.before_return = None
        assert post(client).status_code == 200

    stub.before_return = commit_first
    r = post(client)
    assert r.status_code == 200 and r.headers[REPLAY_HEADER] == "true"
    [row] = rows(engine)
    assert r.json() == row["response"]


def test_overlong_key_is_rejected(client, stub):
    assert post(client, key="k" * 256).status_code == 422 and stub.calls == []


def test_replay_works_while_the_model_is_unavailable(client, stub):
    stored = post(client).json()
    client.app.state.model.value = None
    assert post(client).json() == stored
    assert client.post("/route", json=BODY).status_code == 503


def test_unique_key_is_enforced_by_the_schema(engine):
    with engine.begin() as conn:
        defs = conn.execute(text("SELECT indexdef FROM pg_indexes WHERE tablename = 'route_requests'")).scalars().all()
    assert any("UNIQUE" in d and "idempotency_key" in d for d in defs)
