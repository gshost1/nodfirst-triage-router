"""Static UI page and the local API values it renders.

These tests read the single-file UI from disk and exercise the same MOCK-mode
surfaces the browser uses (`POST /chat`, `GET /stats`) with an isolated SQLite
file. No provider request, no `.env` value, and no browser is involved.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app import create_app
from core.chat import ChatService
from core.llm import LLMClient
from core.metrics_store import MetricsStore
from core.store import CareStore
from router.classify import MessageClass, keyword_classify, urgent_keyword_category
from router.config import RouterConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO_ROOT / "static" / "index.html"

MOCK_ENV = {
    "ROUTER_MODE": "mock",
    "MODEL_SMALL": "glm-5.3-flash",
    "MODEL_MID": "glm-5.3",
    "MODEL_LARGE": "qwen3.8-max",
}
LOGISTICS_MESSAGE = "Book Mom's cardiology for Thursday, Pavel drives"
ROUTINE_MESSAGE = "How did Mom's visit with her neighbor go yesterday?"

DEMO_SCRIPT_RE = re.compile(
    r'<script id="demo-script" type="application/json">(.*?)</script>', re.DOTALL
)


def build_api(db_path: Path) -> TestClient:
    """A MOCK-mode app over an isolated SQLite file, as the browser uses it."""
    service = ChatService(
        client=LLMClient(base_url="mock", api_key="", model="mock"),
        config=RouterConfig.from_env(MOCK_ENV),
        store=CareStore(db_path),
        metrics=MetricsStore(db_path),
    )
    return TestClient(create_app(service=service))


def read_index() -> str:
    assert INDEX_PATH.is_file(), f"the single-file UI is missing at {INDEX_PATH}"
    return INDEX_PATH.read_text(encoding="utf-8")


def demo_script() -> list[dict[str, Any]]:
    match = DEMO_SCRIPT_RE.search(read_index())
    assert match is not None, "the UI must embed the demo script as application/json"
    script = json.loads(match.group(1))
    assert isinstance(script, list)
    return script


def expected_class(message: str) -> MessageClass:
    """Mirror the MOCK-mode classifier (keyword base plus urgent safety net)."""
    if urgent_keyword_category(message) is not None:
        return MessageClass.URGENT
    return keyword_classify(message)


def test_root_serves_the_single_file_ui_with_every_required_region(
    tmp_path: Path,
) -> None:
    api = build_api(tmp_path / "ui.sqlite3")

    response = api.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    for required in (
        'id="composer-form"',
        'id="composer"',
        'id="conversation"',
        'id="tool-actions"',
        'id="stats-panel"',
        'id="disclaimer"',
        'id="replay-control"',
    ):
        assert required in html, f"the served page is missing {required}"
    assert "not medical advice" in html.lower()
    # One static file: no build pipeline, no external asset fetch at load time.
    assert "<script src=" not in html
    assert "<link" not in html


def test_ui_uses_only_the_local_api_and_carries_no_credentials() -> None:
    html = read_index()

    assert "fetch('/chat'" in html
    assert "fetch('/stats'" in html
    for forbidden in (
        "do-ai.run",
        "inference.do",
        "digitalocean",
        "api_key",
        "apiKey",
        "Authorization",
        "Bearer",
        "http://",
        "https://",
    ):
        assert forbidden.lower() not in html.lower(), f"the UI must not contain {forbidden}"


def test_embedded_demo_replay_has_the_approved_five_message_mix() -> None:
    script = demo_script()
    messages = [item["text"] for item in script]

    assert len(messages) == 5
    classes = Counter(expected_class(message) for message in messages)
    assert classes[MessageClass.ROUTINE] == 1
    assert classes[MessageClass.LOGISTICS] == 2
    assert classes[MessageClass.URGENT] == 2
    # The embedded expectation must match what the MOCK classifier returns.
    for item in script:
        assert expected_class(item["text"]) is MessageClass(item["expect"])


def test_logistics_chat_returns_the_badge_values_and_both_tool_actions(
    tmp_path: Path,
) -> None:
    api = build_api(tmp_path / "logistics.sqlite3")

    body = api.post("/chat", json={"message": LOGISTICS_MESSAGE}).json()

    assert body["message_class"] == "logistics"
    assert body["urgent"] is False
    assert body["route"] == "mock"
    assert body["model"] == "glm-5.3"
    assert body["latency_ms"] >= 0.0
    assert body["estimated_cost_usd"] is not None
    assert body["cost_label"] == "estimated"
    action_names = [action["name"] for action in body["tool_actions"]]
    assert action_names == ["add_appointment", "assign_sibling_task"]
    arguments = {action["name"]: action["arguments"] for action in body["tool_actions"]}
    assert arguments["add_appointment"] == {
        "title": "Mom's cardiology",
        "datetime": "Thursday",
    }
    assert arguments["assign_sibling_task"]["sibling"] == "Pavel"


def test_no_tool_reply_reports_no_fabricated_tool_actions(tmp_path: Path) -> None:
    api = build_api(tmp_path / "routine.sqlite3")

    body = api.post("/chat", json={"message": ROUTINE_MESSAGE}).json()

    assert body["message_class"] == "routine"
    assert body["tool_actions"] == []


def test_urgent_reply_carries_escalation_and_a_distinct_route(tmp_path: Path) -> None:
    api = build_api(tmp_path / "urgent.sqlite3")

    body = api.post(
        "/chat", json={"message": "Mom had a fall in the kitchen this morning."}
    ).json()

    assert body["message_class"] == "urgent"
    assert body["route"] == "mock:direct-large"
    assert body["model"] == "qwen3.8-max"
    assert body["escalation_note"] is not None
    assert "911" in body["reply"]


def test_stats_panel_values_come_from_the_stats_api(tmp_path: Path) -> None:
    api = build_api(tmp_path / "stats.sqlite3")

    empty = api.get("/stats").json()
    assert empty["total_calls"] == 0
    assert empty["calls_by_tier"] == {}
    assert empty["average_latency_ms"] is None
    assert empty["estimated_spend_usd"] == 0.0
    assert empty["estimated_savings_usd"] == 0.0
    assert empty["cost_label"] == "estimated"

    api.post("/chat", json={"message": LOGISTICS_MESSAGE})
    after_chat = api.get("/stats").json()

    # The displayed panel is re-read from this endpoint, so it must move.
    assert after_chat["total_calls"] > empty["total_calls"]
    assert after_chat["calls_by_tier"]["mid"] >= 1
    assert after_chat["average_latency_ms"] is not None
    assert after_chat["estimated_spend_usd"] > empty["estimated_spend_usd"]
    assert after_chat["estimated_savings_usd"] > 0.0
