"""FastAPI chat/stats surfaces over classification, routing, and the agent.

Every test uses MOCK or provider-neutral doubles plus a temporary SQLite file;
no DigitalOcean request is made anywhere in this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from app import create_app
from core.chat import ChatService
from core.llm import LLMClient, RawCompletion
from core.metrics_store import MetricsStore
from core.store import CareStore
from router.config import RouterConfig

PRICED_ENV = {
    "ROUTER_MODE": "client",
    "MODEL_SMALL": "glm-5.3-flash",
    "MODEL_MID": "glm-5.3",
    "MODEL_LARGE": "qwen3.8-max",
}
MOCK_ENV = {
    "ROUTER_MODE": "mock",
    "MODEL_SMALL": "glm-5.3-flash",
    "MODEL_MID": "glm-5.3",
    "MODEL_LARGE": "qwen3.8-max",
}
LOGISTICS_MESSAGE = "Book Mom's cardiology for Thursday, Pavel drives"
URGENT_MESSAGE = "Mom had a fall in the kitchen this morning."


class FakeClock:
    """A deterministic clock: every reading advances by one fixed step."""

    def __init__(self, step_ms: float = 5.0) -> None:
        self.now = 0.0
        self.step = step_ms / 1000.0

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def usage(prompt: int, completion: int) -> Any:
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def reply(
    content: str,
    model: str,
    *,
    usage_obj: Any = None,
    calls: list[Any] | None = None,
) -> Any:
    return SimpleNamespace(
        model=model,
        usage=usage_obj,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=calls)
            )
        ],
    )


def tool(name: str, **arguments: Any) -> Any:
    return SimpleNamespace(
        id=f"call-{name}",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class StubClient:
    """Provider-neutral client double returning scripted completions.

    A scripted item may be a `(completion, headers)` pair to emulate a router
    response that carries the selected-route header.
    """

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.is_mock = False

    def _next(self, request: dict[str, Any]) -> Any:
        self.requests.append(request)
        return self.responses.pop(0)

    def complete(self, messages: Any, *, model: str | None = None, **kwargs: Any) -> Any:
        response = self._next({"messages": list(messages), "model": model, **kwargs})
        return response[0] if isinstance(response, tuple) else response

    def complete_raw(
        self, messages: Any, *, model: str | None = None, **kwargs: Any
    ) -> RawCompletion:
        response = self._next({"messages": list(messages), "model": model, **kwargs})
        if isinstance(response, tuple):
            return RawCompletion(completion=response[0], headers=response[1])
        return RawCompletion(completion=response, headers={})


def build_api(
    *,
    config: RouterConfig,
    db_path: Path,
    client: Any,
    clock: FakeClock | None = None,
) -> TestClient:
    service = ChatService(
        client=client,
        config=config,
        store=CareStore(db_path),
        metrics=MetricsStore(db_path),
        clock=clock or FakeClock(),
    )
    return TestClient(create_app(service=service))


def mock_agent_client() -> LLMClient:
    """A canned local client: mock mode never reaches it, and it cannot connect."""
    return LLMClient(base_url="mock", api_key="", model="mock")


def test_root_page_is_html(tmp_path: Path) -> None:
    api = build_api(
        config=RouterConfig.from_env(MOCK_ENV),
        db_path=tmp_path / "root.sqlite3",
        client=mock_agent_client(),
    )

    response = api.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_chat_returns_reply_route_metadata_and_tool_actions(tmp_path: Path) -> None:
    api = build_api(
        config=RouterConfig.from_env(MOCK_ENV),
        db_path=tmp_path / "chat.sqlite3",
        client=mock_agent_client(),
    )

    response = api.post("/chat", json={"message": LOGISTICS_MESSAGE})
    body = response.json()

    assert response.status_code == 200
    assert body["message_class"] == "logistics"
    assert body["route"] == "mock"
    assert body["model"] == "glm-5.3"
    assert body["reply"].strip()
    assert {action["name"] for action in body["tool_actions"]} == {
        "add_appointment",
        "assign_sibling_task",
    }
    assert [call["tier"] for call in body["calls"]] == ["mid", "mid"]
    assert all(call["usage_source"] == "estimated" for call in body["calls"])
    assert body["latency_ms"] == 10.0
    assert body["usage"]["source"] == "estimated"
    assert body["usage"]["prompt_tokens"] > 0
    assert body["disclaimer"].startswith("Not medical advice")
    assert body["cost_label"] == "estimated"


def test_urgent_chat_escalates_without_a_diagnosis(tmp_path: Path) -> None:
    api = build_api(
        config=RouterConfig.from_env(MOCK_ENV),
        db_path=tmp_path / "urgent.sqlite3",
        client=mock_agent_client(),
    )

    body = api.post("/chat", json={"message": URGENT_MESSAGE}).json()

    assert body["message_class"] == "urgent"
    assert body["urgent"] is True
    assert body["route"] == "mock:direct-large"
    assert body["model"] == "qwen3.8-max"
    assert body["override_reason"] == "urgent keyword safety net: falls"
    assert "911" in body["reply"]
    assert body["escalation_note"] and "911" in body["escalation_note"]
    assert [call["tier"] for call in body["calls"]] == ["large"]


def test_chat_rejects_a_blank_message(tmp_path: Path) -> None:
    api = build_api(
        config=RouterConfig.from_env(MOCK_ENV),
        db_path=tmp_path / "blank.sqlite3",
        client=mock_agent_client(),
    )

    assert api.post("/chat", json={"message": ""}).status_code == 422
    assert api.post("/chat", json={"message": "   "}).status_code == 422


def test_chat_reports_configured_tier_metadata_and_counts_the_classifier(
    tmp_path: Path,
) -> None:
    stub = StubClient(
        reply("logistics", "glm-5.3-flash", usage_obj=usage(10, 4)),
        reply("", "glm-5.3", usage_obj=usage(120, 30), calls=[
            tool("add_appointment", title="Mom's cardiology", datetime="Thursday"),
        ]),
        reply("Booked it.", "glm-5.3", usage_obj=usage(160, 12)),
    )
    api = build_api(
        config=RouterConfig.from_env(PRICED_ENV),
        db_path=tmp_path / "client.sqlite3",
        client=stub,
    )

    body = api.post("/chat", json={"message": LOGISTICS_MESSAGE}).json()

    assert body["route"] == "client:mid"
    assert body["mode"] == "client"
    assert body["model"] == "glm-5.3"
    assert body["response_model"] == "glm-5.3"
    assert body["selected_route"] is None
    assert [call["tier"] for call in body["calls"]] == ["small", "mid", "mid"]
    assert [call["model"] for call in body["calls"]] == [
        "glm-5.3-flash",
        "glm-5.3",
        "glm-5.3",
    ]
    assert all(call["usage_source"] == "reported" for call in body["calls"])
    assert body["usage"] == {
        "prompt_tokens": 290,
        "completion_tokens": 46,
        "total_tokens": 336,
        "source": "reported",
    }
    assert abs(body["estimated_cost_usd"] - 0.0005803) < 1e-12
    assert body["latency_ms"] == 15.0
    assert len(stub.requests) == 3
    assert stub.requests[0]["model"] == "glm-5.3-flash"  # classifier on the small tier


def test_do_mode_chat_uses_the_router_boundary_and_keeps_metadata(
    tmp_path: Path,
) -> None:
    stub = StubClient(
        reply("logistics", "glm-5.3-flash", usage_obj=usage(12, 3)),
        (
            reply("", "glm-5.3", usage_obj=usage(100, 20), calls=[
                tool("add_appointment", title="Mom's cardiology", datetime="Thursday"),
            ]),
            {"X-Model-Router-Selected-Route": "logistics-task"},
        ),
        reply("Booked it.", "glm-5.3", usage_obj=usage(140, 9)),
    )
    api = build_api(
        config=RouterConfig.from_env({**PRICED_ENV, "ROUTER_MODE": "do"}),
        db_path=tmp_path / "do.sqlite3",
        client=stub,
    )

    body = api.post("/chat", json={"message": LOGISTICS_MESSAGE}).json()

    assert body["route"] == "do:inference-router"
    assert body["response_model"] == "glm-5.3"  # provider-reported serving model
    assert body["selected_route"] == "logistics-task"
    assert body["model"] == "glm-5.3"
    assert [call["tier"] for call in body["calls"]] == ["small", "router", "router"]
    assert [call["model"] for call in body["calls"]] == [
        "glm-5.3-flash",
        "glm-5.3",
        "glm-5.3",
    ]
    assert abs(body["estimated_cost_usd"] - 0.0004669) < 1e-12
    assert len(stub.requests) == 3
    assert stub.requests[1]["model"] == "router:nodfirst"
    assert stub.requests[2]["model"] == "router:nodfirst"
    assert stub.requests[1]["tools"]  # the agent's tools reach the router boundary


def test_stats_aggregate_per_call_counts_latency_spend_and_savings(
    tmp_path: Path,
) -> None:
    stub = StubClient(
        reply("logistics", "glm-5.3-flash", usage_obj=usage(10, 4)),
        reply("", "glm-5.3", usage_obj=usage(120, 30), calls=[
            tool("add_appointment", title="Mom's cardiology", datetime="Thursday"),
        ]),
        reply("Booked it.", "glm-5.3", usage_obj=usage(160, 12)),
    )
    api = build_api(
        config=RouterConfig.from_env(PRICED_ENV),
        db_path=tmp_path / "stats.sqlite3",
        client=stub,
    )
    api.post("/chat", json={"message": LOGISTICS_MESSAGE})

    stats = api.get("/stats").json()

    assert stats["total_calls"] == 3
    assert stats["calls_by_tier"] == {"small": 1, "mid": 2}
    assert stats["average_latency_ms"] == 5.0
    assert stats["average_latency_ms_by_tier"] == {"small": 5.0, "mid": 5.0}
    assert abs(stats["estimated_spend_usd"] - 0.0005803) < 1e-12
    assert abs(stats["large_model_baseline_usd"] - 0.000856) < 1e-12
    assert abs(stats["estimated_savings_usd"] - 0.0002757) < 1e-12
    assert stats["large_model"] == "qwen3.8-max"
    assert stats["calls_without_usage"] == 0
    assert stats["priced_calls"] == 3
    assert stats["cost_label"] == "estimated"
    assert stats["price_source"] and stats["price_source_date"]


def test_empty_stats_state_is_zeroed_and_honest(tmp_path: Path) -> None:
    api = build_api(
        config=RouterConfig.from_env(PRICED_ENV),
        db_path=tmp_path / "empty.sqlite3",
        client=mock_agent_client(),
    )

    stats = api.get("/stats").json()

    assert stats["total_calls"] == 0
    assert stats["calls_by_tier"] == {}
    assert stats["average_latency_ms"] is None
    assert stats["estimated_spend_usd"] == 0.0
    assert stats["estimated_savings_usd"] == 0.0
    assert stats["cost_label"] == "estimated"


def test_stats_do_not_present_unavailable_usage_as_exact(tmp_path: Path) -> None:
    stub = StubClient(
        reply("logistics", "glm-5.3-flash"),
        reply("", "glm-5.3", calls=[
            tool("add_appointment", title="Mom's cardiology", datetime="Thursday"),
        ]),
        reply("Booked it.", "glm-5.3"),
    )
    api = build_api(
        config=RouterConfig.from_env(PRICED_ENV),
        db_path=tmp_path / "nousage.sqlite3",
        client=stub,
    )
    body = api.post("/chat", json={"message": LOGISTICS_MESSAGE}).json()

    assert body["usage"] == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "source": "unavailable",
    }
    assert body["estimated_cost_usd"] is None
    assert all(call["estimated_cost_usd"] is None for call in body["calls"])

    stats = api.get("/stats").json()
    assert stats["total_calls"] == 3
    assert stats["calls_without_usage"] == 3
    assert stats["priced_calls"] == 0
    assert stats["estimated_spend_usd"] == 0.0
    assert stats["estimated_savings_usd"] == 0.0
    assert "no provider-reported usage" in " ".join(stats["notes"])


def test_mock_chat_makes_no_outbound_connection_and_persists_stats(
    tmp_path: Path, outbound_connections: list[Any]
) -> None:
    db_path = tmp_path / "offline.sqlite3"
    api = build_api(
        config=RouterConfig.from_env(MOCK_ENV),
        db_path=db_path,
        client=mock_agent_client(),
    )

    api.post("/chat", json={"message": LOGISTICS_MESSAGE})
    stats = api.get("/stats").json()

    assert outbound_connections == []
    assert stats["total_calls"] == 2
    # A second MetricsStore over the same file sees the same persisted calls.
    assert len(MetricsStore(db_path).calls()) == 2
