"""Shared sibling thread: history, sender attribution, and urgent alerts.

Everything runs in MOCK mode or through scripted doubles on a temporary
SQLite file; no provider request is made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import create_app
from core.agent import CaregiverAgent
from core.chat import ChatService
from core.metrics_store import MetricsStore
from core.notify import BotmailNotifier, NullNotifier, notifier_from_env
from core.store import CareStore
from router.config import RouterConfig
from tests.test_api import (
    MOCK_ENV,
    PRICED_ENV,
    URGENT_MESSAGE,
    StubClient,
    build_api,
    mock_agent_client,
    reply,
    usage,
)

ROUTINE = "How did Mom's visit with her neighbor go yesterday?"


def mock_api(tmp_path: Path) -> TestClient:
    return build_api(
        config=RouterConfig.from_env(MOCK_ENV),
        db_path=tmp_path / "thread.sqlite3",
        client=mock_agent_client(),
    )


# -- store --------------------------------------------------------------------


def test_store_keeps_thread_in_order_and_limits_recent(tmp_path: Path) -> None:
    store = CareStore(tmp_path / "s.sqlite3")
    for index in range(5):
        store.add_message("Anna", "user", f"m{index}", "routine")
        store.add_message("assistant", "assistant", f"r{index}", "routine")

    recent = store.recent_messages(4)

    assert [m["content"] for m in recent] == ["m3", "r3", "m4", "r4"]
    assert recent[0]["sender"] == "Anna"
    assert recent[0]["created_at"]


def test_store_alert_lifecycle(tmp_path: Path) -> None:
    store = CareStore(tmp_path / "s.sqlite3")
    created = store.add_alert("Pavel", "Mom fell", "call 911")

    assert created["acknowledged"] is False
    assert store.alerts()[0]["id"] == created["id"]
    assert store.acknowledge_alert(created["id"])["acknowledged"] is True
    assert store.alerts()[0]["acknowledged"] is True
    assert store.acknowledge_alert(9999) is None


# -- agent --------------------------------------------------------------------


def test_agent_sends_history_with_sender_prefixes(tmp_path: Path) -> None:
    seen: list[list[dict[str, object]]] = []

    def complete(*, messages: list[dict[str, object]], **_: object) -> object:
        seen.append(list(messages))
        return reply("ok", "m")

    CaregiverAgent(CareStore(tmp_path / "a.sqlite3"), complete).run(
        "she took it at 9",
        sender="Anna",
        history=[
            {"role": "user", "sender": "Pavel", "content": "Did Mom take metformin?"},
            {"role": "assistant", "content": "I don't have a record yet."},
        ],
    )

    messages = seen[0]
    assert "The person writing now is Anna." in str(messages[0]["content"])
    assert messages[1] == {"role": "user", "content": "Pavel: Did Mom take metformin?"}
    assert messages[2] == {"role": "assistant", "content": "I don't have a record yet."}
    assert messages[3] == {"role": "user", "content": "Anna: she took it at 9"}


def test_agent_without_sender_or_history_is_unchanged(tmp_path: Path) -> None:
    seen: list[list[dict[str, object]]] = []

    def complete(*, messages: list[dict[str, object]], **_: object) -> object:
        seen.append(list(messages))
        return reply("ok", "m")

    CaregiverAgent(CareStore(tmp_path / "a.sqlite3"), complete).run("hello")

    assert [m["role"] for m in seen[0]] == ["system", "user"]
    assert seen[0][1]["content"] == "hello"


# -- API ----------------------------------------------------------------------


def test_chat_persists_the_thread_and_reloads_it(tmp_path: Path) -> None:
    api = mock_api(tmp_path)

    first = api.post("/chat", json={"message": ROUTINE, "sender": "Anna"}).json()
    assert first["sender"] == "Anna"
    assert first["alert"] is None

    thread = api.get("/thread").json()
    assert thread["siblings"] == ["Anna", "Pavel"]
    assert [(m["sender"], m["role"]) for m in thread["messages"]] == [
        ("Anna", "user"), ("assistant", "assistant"),
    ]
    assert thread["messages"][0]["content"] == ROUTINE
    assert thread["messages"][0]["message_class"] == "routine"


def test_chat_defaults_the_sender_and_rejects_long_names(tmp_path: Path) -> None:
    api = mock_api(tmp_path)

    assert api.post("/chat", json={"message": ROUTINE}).json()["sender"] == "You"
    assert api.post("/chat", json={"message": ROUTINE, "sender": "  "}).json()["sender"] == "You"
    assert api.post("/chat", json={"message": ROUTINE, "sender": "x" * 41}).status_code == 422


def test_prior_turns_reach_the_model_on_the_next_message(tmp_path: Path) -> None:
    client = StubClient(
        reply("routine", "glm-5.3-flash", usage_obj=usage(5, 1)),
        reply("Noted.", "glm-5.3-flash", usage_obj=usage(20, 3)),
        reply("routine", "glm-5.3-flash", usage_obj=usage(5, 1)),
        reply("Got it.", "glm-5.3-flash", usage_obj=usage(30, 3)),
    )
    api = build_api(
        config=RouterConfig.from_env(PRICED_ENV),
        db_path=tmp_path / "ctx.sqlite3",
        client=client,
    )

    api.post("/chat", json={"message": "Mom seemed cheerful today", "sender": "Pavel"})
    api.post("/chat", json={"message": "Same again this evening", "sender": "Anna"})

    # Request order: classify, agent, classify, agent. The second agent call
    # carries the first exchange ahead of Anna's new message.
    agent_messages = client.requests[3]["messages"]
    contents = [m["content"] for m in agent_messages if m["role"] != "system"]
    assert contents == [
        "Pavel: Mom seemed cheerful today",
        "Noted.",
        "Anna: Same again this evening",
    ]
    # The classifier still sees the bare message, never the sender prefix.
    assert "Anna:" not in str(client.requests[2]["messages"])


def test_urgent_chat_records_an_alert_that_siblings_can_acknowledge(
    tmp_path: Path,
) -> None:
    api = mock_api(tmp_path)

    turn = api.post("/chat", json={"message": URGENT_MESSAGE, "sender": "Pavel"}).json()
    assert turn["urgent"] is True
    alert = turn["alert"]
    assert alert["sender"] == "Pavel"
    assert alert["message"] == URGENT_MESSAGE
    assert "911" in alert["note"]
    assert alert["acknowledged"] is False

    listed = api.get("/alerts").json()["alerts"]
    assert [a["id"] for a in listed] == [alert["id"]]

    acked = api.post(f"/alerts/{alert['id']}/ack")
    assert acked.status_code == 200
    assert acked.json()["acknowledged"] is True
    assert api.get("/alerts").json()["alerts"][0]["acknowledged"] is True
    assert api.post("/alerts/424242/ack").status_code == 404


def test_routine_chat_records_no_alert(tmp_path: Path) -> None:
    api = mock_api(tmp_path)
    api.post("/chat", json={"message": ROUTINE, "sender": "Anna"})
    assert api.get("/alerts").json()["alerts"] == []


def test_ui_has_sender_picker_alerts_panel_and_thread_reload(tmp_path: Path) -> None:
    html = mock_api(tmp_path).get("/").text
    for required in (
        'id="sender"', 'id="alerts-panel"', 'id="alerts"',
        "fetch('/thread'", "fetch('/alerts'", "/ack'",
    ):
        assert required in html, f"the served page is missing {required}"


# -- outbound alert email (Botmail) ------------------------------------------


def _fake_transport(calls: list[dict[str, object]], *, fail: Exception | None = None):
    def transport(url: str, headers: dict[str, str], body: bytes) -> dict[str, object]:
        calls.append({"url": url, "headers": headers, "body": json.loads(body)})
        if fail is not None:
            raise fail
        return {"id": "msg_123", "status": "queued"}
    return transport


def notifying_api(tmp_path: Path, notifier: object) -> TestClient:
    service = ChatService(
        client=mock_agent_client(),
        config=RouterConfig.from_env(MOCK_ENV),
        store=CareStore(tmp_path / "n.sqlite3"),
        metrics=MetricsStore(tmp_path / "n.sqlite3"),
        notifier=notifier,  # type: ignore[arg-type]
    )
    return TestClient(create_app(service=service))


def test_urgent_alert_is_emailed_through_botmail_without_leaking_the_key(
    tmp_path: Path, outbound_connections: list[object],
) -> None:
    calls: list[dict[str, object]] = []
    notifier = BotmailNotifier(
        api_key="bm_secret_test", recipients=["anna@example.com", "pavel@example.com"],
        transport=_fake_transport(calls),
    )
    api = notifying_api(tmp_path, notifier)

    alert = api.post("/chat", json={"message": URGENT_MESSAGE, "sender": "Pavel"}).json()["alert"]

    assert alert["email_status"] == "sent"
    assert alert["email_recipients"] == ["anna@example.com", "pavel@example.com"]
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://botmail.pro/v1/send"
    headers = call["headers"]
    assert headers["Authorization"] == "Bearer bm_secret_test"
    assert headers["Idempotency-Key"] == f"nodfirst-alert-{alert['id']}"
    body = call["body"]
    assert body["to"] == "anna@example.com, pavel@example.com"
    assert body["subject"] == "Urgent about Mom from Pavel"
    assert URGENT_MESSAGE in body["text"]
    assert "911" in body["text"]
    assert "bm_secret_test" not in json.dumps(alert)
    assert outbound_connections == []

    listed = api.get("/alerts").json()
    assert listed["delivery"] == {
        "enabled": True, "provider": "botmail",
        "recipients": ["anna@example.com", "pavel@example.com"],
    }
    assert listed["alerts"][0]["email_status"] == "sent"


def test_botmail_failure_is_recorded_and_never_breaks_the_chat(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    notifier = BotmailNotifier(
        api_key="bm_secret_test", recipients=["anna@example.com"],
        transport=_fake_transport(calls, fail=TimeoutError("slow")),
    )
    api = notifying_api(tmp_path, notifier)

    response = api.post("/chat", json={"message": URGENT_MESSAGE, "sender": "Anna"})

    assert response.status_code == 200
    alert = response.json()["alert"]
    assert alert["email_status"] == "failed"
    assert alert["email_detail"] == "Botmail request failed: TimeoutError"
    assert "bm_secret_test" not in response.text


def test_botmail_http_error_reports_the_status_code_only(tmp_path: Path) -> None:
    import urllib.error
    error = urllib.error.HTTPError(
        "https://botmail.pro/v1/send", 429, "Too Many", {}, None,  # type: ignore[arg-type]
    )
    notifier = BotmailNotifier(
        api_key="bm_secret_test", recipients=["anna@example.com"],
        transport=_fake_transport([], fail=error),
    )
    sent = notifier.send_alert({"id": 1, "sender": "Anna", "message": "m", "note": "n"})
    assert sent.status == "failed"
    assert sent.detail == "Botmail returned HTTP 429"


def test_alerts_stay_local_when_email_is_not_configured(tmp_path: Path) -> None:
    api = mock_api(tmp_path)

    alert = api.post("/chat", json={"message": URGENT_MESSAGE}).json()["alert"]

    assert alert["email_status"] == "disabled"
    assert alert["email_recipients"] == []
    delivery = api.get("/alerts").json()["delivery"]
    assert delivery["enabled"] is False
    assert "not configured" in delivery["reason"]


def test_notifier_from_env_requires_recipients_and_a_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    for name in ("ALERT_EMAIL_TO", "BOTMAIL_API_KEY", "BOTMAIL_KEY_FILE"):
        monkeypatch.delenv(name, raising=False)
    assert isinstance(notifier_from_env(), NullNotifier)

    monkeypatch.setenv("ALERT_EMAIL_TO", "anna@example.com, pavel@example.com")
    missing = notifier_from_env()
    assert isinstance(missing, NullNotifier)
    assert "no Botmail key" in missing.reason

    key_file = tmp_path / "key"
    key_file.write_text("bm_from_file\n")
    monkeypatch.setenv("BOTMAIL_KEY_FILE", str(key_file))
    monkeypatch.setenv("PORT", "8765")
    built = notifier_from_env()
    assert isinstance(built, BotmailNotifier)
    assert built.recipients == ["anna@example.com", "pavel@example.com"]
    assert built.app_url == "http://127.0.0.1:8765"

    monkeypatch.setenv("BOTMAIL_API_KEY", "bm_from_env")
    assert isinstance(notifier_from_env(), BotmailNotifier)


def test_ui_shows_email_delivery_state(tmp_path: Path) -> None:
    html = mock_api(tmp_path).get("/").text
    assert 'id="alert-delivery"' in html
    assert "via Botmail" in html
