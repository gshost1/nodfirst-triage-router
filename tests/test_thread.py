"""Shared sibling thread: history, sender attribution, and urgent alerts.

Everything runs in MOCK mode or through scripted doubles on a temporary
SQLite file; no provider request is made.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from core.agent import CaregiverAgent
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
