"""Caregiving message classification and the urgent keyword safety net."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from core.llm import LLMClient
from router import (
    URGENT_KEYWORD_CATEGORIES,
    MessageClass,
    RouterConfig,
    RouterMode,
    TriageClassifier,
    model_classify,
)


def mock_llm_client(model: str = "mock-model") -> LLMClient:
    return LLMClient(base_url="mock", api_key="", model=model)


def client_mode_config() -> RouterConfig:
    return RouterConfig(
        mode=RouterMode.CLIENT,
        small_model="distinct-small-model",
        mid_model="distinct-mid-model",
        large_model="distinct-large-model",
    )


class StubReplyClient(LLMClient):
    """Model test double that replies with canned classifier text."""

    def __init__(self, reply: str) -> None:
        super().__init__(base_url="mock", api_key="", model="stub-model")
        # Stand in for a remote client so the model path runs, without
        # constructing the real SDK client.
        self.is_mock = False
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: Sequence[Any],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append({"messages": list(messages), "model": model, "kwargs": kwargs})
        message = SimpleNamespace(role="assistant", content=self.reply, tool_calls=None)
        choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
        return SimpleNamespace(model=model or self.model, choices=[choice], usage=None)


def test_safety_net_covers_exactly_the_approved_categories() -> None:
    assert set(URGENT_KEYWORD_CATEGORIES) == {
        "falls",
        "chest pain",
        "confusion",
        "missed dose",
        "cannot reach",
    }


@pytest.mark.parametrize("category", sorted(URGENT_KEYWORD_CATEGORIES))
def test_urgent_keyword_overrides_lower_risk_classifier_result(
    category: str, urgent_messages: dict[str, str]
) -> None:
    """A listed urgent signal overrides a classifier that answered routine."""
    consulted: list[str] = []

    def lower_risk_classifier(message: str) -> MessageClass:
        consulted.append(message)
        return MessageClass.ROUTINE

    classifier = TriageClassifier(
        mock_llm_client(),
        client_mode_config(),
        classify_with=lower_risk_classifier,
    )
    message = urgent_messages[category]

    result = classifier.classify(message)

    assert consulted == [message]  # the classifier was consulted first...
    assert result.message_class is MessageClass.URGENT  # ...then overridden
    assert result.is_urgent
    assert result.safety_override == category
    assert result.source == "keyword-safety-net"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("ROUTINE.", MessageClass.ROUTINE),
        ("Logistics", MessageClass.LOGISTICS),
        ("urgent", MessageClass.URGENT),
        ("I cannot decide between classes.", MessageClass.LOGISTICS),
    ],
)
def test_model_classification_parses_and_defaults_cautiously(
    reply: str, expected: MessageClass
) -> None:
    config = client_mode_config()
    client = StubReplyClient(reply)

    result = model_classify(client, config, "Mom slept well and waved to a neighbor.")

    assert result is expected
    assert client.calls[0]["model"] == "distinct-small-model"
    assert client.calls[0]["kwargs"] == {"max_completion_tokens": 16}


def test_triage_classifier_uses_the_model_path_for_remote_clients() -> None:
    config = client_mode_config()
    client = StubReplyClient("logistics")
    classifier = TriageClassifier(client, config)

    result = classifier.classify("Mom slept well and waved to a neighbor.")

    assert result.message_class is MessageClass.LOGISTICS
    assert result.source == "model-classifier"
    assert result.safety_override is None
    assert len(client.calls) == 1


def test_model_urgent_judgment_stands_without_keyword_signals() -> None:
    classifier = TriageClassifier(StubReplyClient("urgent"), client_mode_config())

    result = classifier.classify("Mom is very short of breath tonight.")

    assert result.message_class is MessageClass.URGENT
    assert result.source == "model-classifier"
    assert result.safety_override is None


def test_mock_mode_uses_deterministic_keyword_classifier(
    routine_message: str, logistics_message: str
) -> None:
    classifier = TriageClassifier(mock_llm_client(), RouterConfig(mode=RouterMode.MOCK))

    routine = classifier.classify(routine_message)
    logistics = classifier.classify(logistics_message)

    assert routine.message_class is MessageClass.ROUTINE
    assert routine.source == "keyword-classifier"
    assert logistics.message_class is MessageClass.LOGISTICS
    assert logistics.source == "keyword-classifier"


def test_mock_client_in_client_mode_also_uses_keyword_classifier(
    routine_message: str,
) -> None:
    classifier = TriageClassifier(mock_llm_client(), client_mode_config())

    result = classifier.classify(routine_message)

    assert result.message_class is MessageClass.ROUTINE
    assert result.source == "keyword-classifier"
