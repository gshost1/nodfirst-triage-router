"""`POST /inbound`: gateway messages triaged like typed turns, MOCK mode only."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app
from core.chat import ChatService
from core.inbound import parse_contacts, resolve_sender, twiml_reply
from core.metrics_store import MetricsStore
from core.store import CareStore
from router.config import RouterConfig
from tests.test_api import MOCK_ENV, URGENT_MESSAGE, mock_agent_client

CONTACTS = {"+15551234567": "Anna", "pavel@example.com": "Pavel"}
LOGISTICS = "Book Mom's cardiology for Thursday, Pavel drives"


def api(tmp_path: Path, *, secret: str = "") -> TestClient:
    service = ChatService(
        client=mock_agent_client(),
        config=RouterConfig.from_env(MOCK_ENV),
        store=CareStore(tmp_path / "in.sqlite3"),
        metrics=MetricsStore(tmp_path / "in.sqlite3"),
    )
    return TestClient(create_app(
        service=service, inbound_contacts=CONTACTS, inbound_secret=secret,
    ))


def test_contacts_parse_and_normalize() -> None:
    contacts = parse_contacts(" +1 (555) 123-4567 = Anna , Pavel <Pavel@Example.com>=Pavel,bad")
    assert contacts == CONTACTS
    assert resolve_sender("(555) 123 4567", {"5551234567": "Anna"}) == "Anna"
    assert resolve_sender("PAVEL@example.com", CONTACTS) == "Pavel"
    assert resolve_sender("+15550000000", CONTACTS) == "+15550000000"
    assert resolve_sender("", CONTACTS) == "Unknown sender"
    assert resolve_sender("x" * 80 + "@example.com", {}) == ("x" * 80 + "@example.com")[:40]
    assert resolve_sender("letters only", {}) == "Unknown sender"


def test_json_inbound_is_triaged_and_attributed(tmp_path: Path) -> None:
    client = api(tmp_path)

    response = client.post("/inbound", json={"from": "+1 555 123 4567", "text": LOGISTICS})

    assert response.status_code == 200
    body = response.json()
    assert body["message_class"] == "logistics"
    assert body["sender"] == "Anna"
    assert body["inbound"] == {"channel": "json", "origin": "+1 555 123 4567", "sender": "Anna"}
    assert [a["name"] for a in body["tool_actions"]] == ["add_appointment", "assign_sibling_task"]
    thread = client.get("/thread").json()["messages"]
    assert thread[0]["sender"] == "Anna" and thread[0]["content"] == LOGISTICS


def test_twilio_form_inbound_returns_twiml_and_records_an_alert(tmp_path: Path) -> None:
    client = api(tmp_path)

    response = client.post(
        "/inbound", data={"From": "pavel@example.com", "Body": URGENT_MESSAGE},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert response.text.startswith('<?xml version="1.0" encoding="UTF-8"?><Response><Message>')
    assert "call 911" in response.text
    alerts = client.get("/alerts").json()["alerts"]
    assert len(alerts) == 1 and alerts[0]["sender"] == "Pavel"


def test_unknown_sender_is_shown_by_address_not_as_a_sibling(tmp_path: Path) -> None:
    body = api(tmp_path).post(
        "/inbound", json={"from": "Mallory <mallory@evil.example>", "text": "hi"},
    ).json()
    assert body["sender"] == "mallory@evil.example"


def test_inbound_rejects_blank_long_and_malformed_bodies(tmp_path: Path) -> None:
    client = api(tmp_path)
    assert client.post("/inbound", json={"from": "+15551234567", "text": "  "}).status_code == 422
    assert client.post("/inbound", json={"from": "x", "text": "y" * 2001}).status_code == 422
    assert client.post("/inbound", json=["not", "an", "object"]).status_code == 422


def test_inbound_secret_gates_the_webhook(tmp_path: Path) -> None:
    client = api(tmp_path, secret="s3cret")
    payload = {"from": "+15551234567", "text": "Mom seems cheerful today"}

    assert client.post("/inbound", json=payload).status_code == 401
    assert client.post(
        "/inbound", json=payload, headers={"X-Inbound-Token": "wrong"},
    ).status_code == 401
    assert client.post(
        "/inbound", json=payload, headers={"X-Inbound-Token": "s3cret"},
    ).status_code == 200


def test_twiml_escapes_markup() -> None:
    assert "&lt;b&gt;" in twiml_reply("<b>hi</b> & bye")
