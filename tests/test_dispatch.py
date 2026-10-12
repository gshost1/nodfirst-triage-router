"""Route dispatch for mock, client, and DO routing modes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from core.llm import LLMClient
from router import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    URGENT_ESCALATION_NOTE,
    ClassificationResult,
    DORouterAdapter,
    MessageClass,
    RouterConfig,
    RouterConfigError,
    RouterMode,
    TriageClassifier,
    dispatch_route,
)
from router.execute import RouteExecutor

URGENT_MESSAGE = "Mom fell down in the kitchen and can't get up."
DIRECT_LARGE_ROUTES = {
    RouterMode.MOCK: "mock:direct-large",
    RouterMode.CLIENT: "client:direct-large",
    RouterMode.DO: "do:direct-large",
}


def mock_llm_client(model: str = "mock-model") -> LLMClient:
    return LLMClient(base_url="mock", api_key="", model=model)


def env_for(tier_env: dict[str, str], mode: RouterMode) -> dict[str, str]:
    return {**tier_env, "ROUTER_MODE": mode.value}


class SpyRouterAdapter(DORouterAdapter):
    """Router boundary double that records dispatch attempts."""

    def __init__(self, complete_raw: Any, router_model: str) -> None:
        super().__init__(complete_raw, router_model)
        self.calls: list[dict[str, Any]] = []

    def dispatch(
        self,
        messages: Sequence[Any],
        *,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    ) -> Any:
        self.calls.append(
            {"messages": list(messages), "max_completion_tokens": max_completion_tokens}
        )
        return super().dispatch(messages, max_completion_tokens=max_completion_tokens)


def test_client_mode_routes_routine_to_small_and_logistics_to_mid(
    tier_env: dict[str, str],
    routine_message: str,
    logistics_message: str,
) -> None:
    config = RouterConfig.from_env(env_for(tier_env, RouterMode.CLIENT))
    classifier = TriageClassifier(mock_llm_client(), config)

    routine = dispatch_route(config, classifier.classify(routine_message))
    logistics = dispatch_route(config, classifier.classify(logistics_message))

    assert routine.message_class is MessageClass.ROUTINE
    assert routine.route == "client:small"
    assert routine.model == "distinct-small-model"
    assert logistics.message_class is MessageClass.LOGISTICS
    assert logistics.route == "client:mid"
    assert logistics.model == "distinct-mid-model"


def test_client_mode_completes_on_the_configured_tier_with_reply_metadata(
    tier_env: dict[str, str],
    routine_message: str,
    logistics_message: str,
) -> None:
    config = RouterConfig.from_env(env_for(tier_env, RouterMode.CLIENT))
    client = mock_llm_client()
    adapter = DORouterAdapter.from_client(client, config.do_router_model)
    classifier = TriageClassifier(client, config)
    executor = RouteExecutor(client, adapter)

    cases = (
        (routine_message, "client:small", "distinct-small-model"),
        (logistics_message, "client:mid", "distinct-mid-model"),
    )
    for message, expected_route, expected_model in cases:
        decision = dispatch_route(config, classifier.classify(message))
        outcome = executor.execute(decision, [{"role": "user", "content": message}])

        assert outcome.decision is decision
        assert decision.route == expected_route
        assert decision.model == expected_model
        assert outcome.response_model == expected_model  # mock echoes the request
        assert outcome.selected_route is None
        assert outcome.completion.choices[0].message.content  # warm canned reply


@pytest.mark.parametrize("mode", [RouterMode.MOCK, RouterMode.CLIENT, RouterMode.DO])
def test_urgent_dispatches_directly_to_large_model_bypassing_every_router(
    tier_env: dict[str, str], mode: RouterMode
) -> None:
    config = RouterConfig.from_env(env_for(tier_env, mode))
    client = mock_llm_client()
    spy = SpyRouterAdapter(client.complete_raw, config.do_router_model)
    classifier = TriageClassifier(client, config)
    executor = RouteExecutor(client, spy)

    decision = dispatch_route(config, classifier.classify(URGENT_MESSAGE))
    outcome = executor.execute(decision, [{"role": "user", "content": URGENT_MESSAGE}])

    assert decision.message_class is MessageClass.URGENT
    assert decision.model == "distinct-large-model"
    assert decision.route == DIRECT_LARGE_ROUTES[mode]
    assert decision.override_reason == "urgent keyword safety net: falls"
    assert decision.escalation_note == URGENT_ESCALATION_NOTE
    assert "911" in URGENT_ESCALATION_NOTE
    assert "doctor" in URGENT_ESCALATION_NOTE
    assert outcome.response_model == "distinct-large-model"
    assert spy.calls == []  # the router boundary was never called
    reply = outcome.completion.choices[0].message.content
    assert "911" in f"{reply} {URGENT_ESCALATION_NOTE}"
    assert "doctor" in f"{reply} {URGENT_ESCALATION_NOTE}"


@pytest.mark.parametrize("mode", [RouterMode.MOCK, RouterMode.CLIENT, RouterMode.DO])
def test_model_classified_urgent_also_dispatches_directly_to_large(
    tier_env: dict[str, str], mode: RouterMode
) -> None:
    config = RouterConfig.from_env(env_for(tier_env, mode))
    client = mock_llm_client()
    spy = SpyRouterAdapter(client.complete_raw, config.do_router_model)
    classifier = TriageClassifier(
        client, config, classify_with=lambda message: MessageClass.URGENT
    )
    executor = RouteExecutor(client, spy)

    decision = dispatch_route(
        config, classifier.classify("Mom is very short of breath tonight.")
    )
    outcome = executor.execute(
        decision, [{"role": "user", "content": "Mom is very short of breath tonight."}]
    )

    assert decision.message_class is MessageClass.URGENT
    assert decision.override_reason is None
    assert decision.model == "distinct-large-model"
    assert decision.escalation_note == URGENT_ESCALATION_NOTE
    assert outcome.response_model == "distinct-large-model"
    assert spy.calls == []


@pytest.mark.parametrize(
    "category", ["cannot reach", "chest pain", "confusion", "falls", "missed dose"]
)
def test_safety_net_category_overrides_lower_risk_result_and_bypasses_router(
    tier_env: dict[str, str],
    urgent_messages: dict[str, str],
    category: str,
) -> None:
    config = RouterConfig.from_env(env_for(tier_env, RouterMode.DO))
    client = mock_llm_client()
    spy = SpyRouterAdapter(client.complete_raw, config.do_router_model)
    classifier = TriageClassifier(
        client, config, classify_with=lambda message: MessageClass.ROUTINE
    )
    executor = RouteExecutor(client, spy)
    message = urgent_messages[category]

    decision = dispatch_route(config, classifier.classify(message))
    outcome = executor.execute(decision, [{"role": "user", "content": message}])

    assert decision.message_class is MessageClass.URGENT
    assert decision.override_reason == f"urgent keyword safety net: {category}"
    assert decision.model == "distinct-large-model"
    assert outcome.response_model == "distinct-large-model"
    assert spy.calls == []  # direct-large path; the router was never consulted


def test_mock_mode_returns_canned_route_metadata_without_external_requests(
    tier_env: dict[str, str],
    outbound_connections: list[Any],
    logistics_message: str,
) -> None:
    config = RouterConfig.from_env(env_for(tier_env, RouterMode.MOCK))
    client = mock_llm_client()
    adapter = DORouterAdapter.from_client(client, config.do_router_model)
    classifier = TriageClassifier(client, config)
    executor = RouteExecutor(client, adapter)

    decision = dispatch_route(config, classifier.classify(logistics_message))
    outcome = executor.execute(decision, [{"role": "user", "content": logistics_message}])

    assert decision.route == "mock"
    assert decision.model == "distinct-mid-model"  # canned tier metadata
    assert decision.escalation_note is None
    assert outcome.response_model == "distinct-mid-model"
    assert outcome.selected_route is None
    assert outcome.completion.choices[0].message.content
    assert outbound_connections == []  # zero external provider requests


def test_unknown_router_mode_is_rejected(tier_env: dict[str, str]) -> None:
    with pytest.raises(RouterConfigError, match="ROUTER_MODE"):
        RouterConfig.from_env({**tier_env, "ROUTER_MODE": "batch"})


def test_client_mode_requires_every_tier_model() -> None:
    with pytest.raises(RouterConfigError, match="MODEL_LARGE"):
        RouterConfig.from_env(
            {"ROUTER_MODE": "client", "MODEL_SMALL": "s", "MODEL_MID": "m"}
        )


def test_mock_mode_allows_missing_tiers_and_falls_back() -> None:
    config = RouterConfig.from_env({"ROUTER_MODE": "mock"})

    routine = dispatch_route(
        config, ClassificationResult(MessageClass.LOGISTICS, "keyword-classifier")
    )
    urgent = dispatch_route(
        config,
        ClassificationResult(
            MessageClass.URGENT, "keyword-safety-net", safety_override="falls"
        ),
    )

    assert routine.model == "mock"
    assert routine.route == "mock"
    assert urgent.model == "mock"
    assert urgent.route == "mock:direct-large"


def test_do_mode_defaults_router_to_nodfirst_and_allows_override(
    tier_env: dict[str, str],
) -> None:
    config = RouterConfig.from_env(env_for(tier_env, RouterMode.DO))
    assert config.do_router_model == "router:nodfirst"
    assert config.mode is RouterMode.DO

    overridden = RouterConfig.from_env(
        {**env_for(tier_env, RouterMode.DO), "DO_ROUTER_MODEL": "router:beta"}
    )
    assert overridden.do_router_model == "router:beta"
