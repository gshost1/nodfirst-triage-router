"""The daily digest: store-grounded tool, endpoints, and optional email."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app
from core.agent import CaregiverAgent
from core.chat import ChatService
from core.metrics_store import MetricsStore
from core.mock_agent import mock_agent_completion
from core.notify import BotmailNotifier
from core.store import CareStore
from router.config import RouterConfig
from tests.test_api import MOCK_ENV, URGENT_MESSAGE, mock_agent_client
from tests.test_thread import _fake_transport


def service(tmp_path: Path, notifier: object = None) -> ChatService:
    db = tmp_path / "d.sqlite3"
    kwargs = {"notifier": notifier} if notifier is not None else {}
    return ChatService(
        client=mock_agent_client(), config=RouterConfig.from_env(MOCK_ENV),
        store=CareStore(db), metrics=MetricsStore(db), **kwargs,  # type: ignore[arg-type]
    )


def test_digest_is_grounded_in_the_store(tmp_path: Path) -> None:
    store = CareStore(tmp_path / "s.sqlite3")
    assert store.daily_digest()["summary"].startswith("Daily digest for Mom (Ludmila, 81)")
    assert "Messages: none." in store.daily_digest()["summary"]

    store.add_message("Anna", "user", "hi", "routine")
    store.add_message("Pavel", "user", "fall", "urgent")
    store.add_message("assistant", "assistant", "reply", "urgent")
    alert = store.add_alert("Pavel", "Mom fell.", "call 911")
    store.log_med("metformin", "8am", False)
    store.log_med("vitamin D", "9am", True)
    store.add_appointment("Cardiology", "Thursday")
    store.assign_sibling_task("Pavel", "Drive Mom")

    digest = store.daily_digest()
    text = digest["summary"]
    assert digest["message_counts"] == {"routine": 1, "urgent": 1}
    assert digest["senders"] == ["Anna", "Pavel"]
    assert "Messages: 2 (1 routine, 1 urgent) from Anna, Pavel." in text
    assert "Urgent alerts: 1, 1 still unacknowledged: Pavel: Mom fell." in text
    assert "Medications recorded: 1 taken, 1 missed (metformin at 8am)." in text
    assert "Appointments: Cardiology: Thursday." in text
    assert "Sibling tasks: Pavel: Drive Mom." in text

    store.acknowledge_alert(alert["id"])
    assert "Urgent alerts: 1, 0 still unacknowledged." in store.daily_digest()["summary"]


def test_agent_reply_for_the_digest_is_the_store_text_not_the_model(tmp_path: Path) -> None:
    store = CareStore(tmp_path / "a.sqlite3")
    store.log_med("metformin", "8am", False)

    result = CaregiverAgent(store, mock_agent_completion).run("Send today's digest")

    assert [a.name for a in result.tool_actions] == ["summarize_day"]
    assert result.reply == store.daily_digest()["summary"]
    assert "metformin at 8am" in result.reply


def test_digest_chat_turn_is_routine_and_cheap(tmp_path: Path) -> None:
    api = TestClient(create_app(service=service(tmp_path)))
    api.post("/chat", json={"message": URGENT_MESSAGE, "sender": "Anna"})

    turn = api.post("/chat", json={"message": "Send today's digest", "sender": "Pavel"}).json()

    assert turn["message_class"] == "routine"
    assert turn["model"] == "glm-5.3-flash"
    assert turn["reply"].startswith("Daily digest for Mom")
    assert "1 still unacknowledged" in turn["reply"]
    assert [a["name"] for a in turn["tool_actions"]] == ["summarize_day"]


def test_get_digest_makes_no_model_call(tmp_path: Path) -> None:
    api = TestClient(create_app(service=service(tmp_path)))
    body = api.get("/digest").json()
    assert body["summary"].startswith("Daily digest for Mom")
    assert api.get("/stats").json()["total_calls"] == 0


def test_digest_email_goes_through_botmail_or_reports_why_not(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    notifier = BotmailNotifier(
        api_key="bm_secret", recipients=["anna@example.com"],
        transport=_fake_transport(calls),
    )
    api = TestClient(create_app(service=service(tmp_path, notifier)))

    body = api.post("/digest/send").json()

    assert body["email"]["status"] == "sent"
    assert calls[0]["body"]["subject"] == "Daily digest about Mom"  # type: ignore[index]
    assert "Daily digest for Mom" in calls[0]["body"]["text"]  # type: ignore[index]
    assert "Idempotency-Key" not in calls[0]["headers"]  # type: ignore[operator]
    assert "bm_secret" not in json.dumps(body)

    off_dir = tmp_path / "off"
    off_dir.mkdir()
    off = TestClient(create_app(service=service(off_dir))).post("/digest/send").json()
    assert off["email"]["status"] == "disabled"


def test_ui_has_the_digest_controls(tmp_path: Path) -> None:
    html = TestClient(create_app(service=service(tmp_path))).get("/").text
    for required in ('id="digest-control"', 'id="digest-email"', "fetch('/digest/send'"):
        assert required in html
