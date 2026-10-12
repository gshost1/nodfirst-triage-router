"""DO-mode dispatch through the router boundary test double."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from core.llm import LLMClient, RawCompletion
from router import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    DORouterAdapter,
    MessageClass,
    RouterConfig,
    RouterMode,
    TriageClassifier,
    dispatch_route,
)
from router.execute import RouteExecutor

CANNED_ROUTER_REPLY = "Canned caregiver reply from the router double."


class FakeRouterProvider:
    """Test double at the DO router boundary: records requests, canned responses."""

    def __init__(self, response_model: str | None = "glm-5.3-flash") -> None:
        self.calls: list[dict[str, Any]] = []
        self.response_model = response_model
        self.headers: dict[str, str] = {
            "x-model-router-selected-route": "logistics-task"
        }

    def complete_raw(
        self,
        *,
        messages: Sequence[Any],
        model: str,
        max_completion_tokens: int,
        **kwargs: Any,
    ) -> RawCompletion:
        del kwargs
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "max_completion_tokens": max_completion_tokens,
            }
        )
        message = SimpleNamespace(
            role="assistant", content=CANNED_ROUTER_REPLY, tool_calls=None
        )
        choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
        completion = SimpleNamespace(
            model=self.response_model, choices=[choice], usage=None
        )
        return RawCompletion(completion=completion, headers=dict(self.headers))


def mock_llm_client(model: str = "mock-model") -> LLMClient:
    return LLMClient(base_url="mock", api_key="", model=model)


def test_do_mode_dispatches_routine_and_logistics_to_nodfirst_router(
    tier_env: dict[str, str],
    outbound_connections: list[Any],
    routine_message: str,
    logistics_message: str,
) -> None:
    config = RouterConfig.from_env({**tier_env, "ROUTER_MODE": RouterMode.DO.value})
    provider = FakeRouterProvider()
    adapter = DORouterAdapter(provider.complete_raw, config.do_router_model)
    client = mock_llm_client()
    classifier = TriageClassifier(client, config)
    executor = RouteExecutor(client, adapter)

    cases = (
        (routine_message, MessageClass.ROUTINE, "routine-task"),
        (logistics_message, MessageClass.LOGISTICS, "logistics-task"),
    )
    for message, message_class, selected_task in cases:
        provider.headers = {"x-model-router-selected-route": selected_task}
        decision = dispatch_route(config, classifier.classify(message))
        assert decision.message_class is message_class
        assert decision.model == "router:nodfirst"
        assert decision.route == "do:inference-router"

        outcome = executor.execute(decision, [{"role": "user", "content": message}])
        assert outcome.completion.choices[0].message.content == CANNED_ROUTER_REPLY
        assert outcome.response_model == "glm-5.3-flash"
        assert outcome.selected_route == selected_task

    assert len(provider.calls) == 2
    assert all(call["model"] == "router:nodfirst" for call in provider.calls)
    assert all(
        call["max_completion_tokens"] == DEFAULT_MAX_COMPLETION_TOKENS
        for call in provider.calls
    )
    assert outbound_connections == []  # no real DO request occurred


def test_adapter_passes_router_model_and_token_cap_to_provider() -> None:
    provider = FakeRouterProvider()
    adapter = DORouterAdapter(provider.complete_raw, "router:nodfirst")

    dispatched = adapter.dispatch(
        [{"role": "user", "content": "hello"}], max_completion_tokens=64
    )

    assert provider.calls == [
        {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "router:nodfirst",
            "max_completion_tokens": 64,
        }
    ]
    assert dispatched.requested_model == "router:nodfirst"
    assert dispatched.response_model == "glm-5.3-flash"
    assert dispatched.selected_route == "logistics-task"


def test_adapter_defaults_the_token_cap() -> None:
    provider = FakeRouterProvider()
    adapter = DORouterAdapter(provider.complete_raw, "router:nodfirst")

    adapter.dispatch([{"role": "user", "content": "hello"}])

    assert provider.calls[0]["max_completion_tokens"] == DEFAULT_MAX_COMPLETION_TOKENS


def test_adapter_does_not_invent_unavailable_metadata() -> None:
    provider = FakeRouterProvider(response_model=None)
    provider.headers = {}
    adapter = DORouterAdapter(provider.complete_raw, "router:nodfirst")

    dispatched = adapter.dispatch([{"role": "user", "content": "hello"}])

    assert dispatched.response_model is None
    assert dispatched.selected_route is None


def test_selected_route_header_is_read_case_insensitively() -> None:
    provider = FakeRouterProvider()
    provider.headers = {"X-Model-Router-Selected-Route": "urgent-health-task"}
    adapter = DORouterAdapter(provider.complete_raw, "router:nodfirst")

    dispatched = adapter.dispatch([{"role": "user", "content": "hello"}])

    assert dispatched.selected_route == "urgent-health-task"
