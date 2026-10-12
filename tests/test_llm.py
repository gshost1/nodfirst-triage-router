import socket
from types import SimpleNamespace
from typing import Any

import pytest

from core import llm


@pytest.fixture(autouse=True)
def isolate_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)


def test_client_uses_environment_configuration_and_sdk_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "configured-model")
    captured: dict[str, Any] = {}
    completion_calls: list[dict[str, Any]] = []

    def create_completion(**kwargs: Any) -> None:
        completion_calls.append(kwargs)

    class StubOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=create_completion)
            )

    monkeypatch.setattr(llm, "OpenAI", StubOpenAI)

    client = llm.create_llm_client()

    assert client.model == "configured-model"
    assert captured == {
        "api_key": "test-key",
        "base_url": "https://llm.example/v1",
        "timeout": 30.0,
        "max_retries": 2,
    }
    client.complete([{"role": "user", "content": "hello"}])
    client.complete([], model="caller-selected-model")
    assert [call["model"] for call in completion_calls] == [
        "configured-model",
        "caller-selected-model",
    ]


def test_real_sdk_has_configured_timeout_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "configured-model")

    client = llm.create_llm_client()
    try:
        assert client._sdk_client is not None
        assert str(client._sdk_client.base_url) == "https://llm.example/v1/"
        assert client._sdk_client.timeout == 30.0
        assert client._sdk_client.max_retries == 2
    finally:
        client.close()


def test_mock_base_url_returns_canned_completion_without_sdk_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "mock")
    monkeypatch.setenv("LLM_MODEL", "mock-configured-model")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    outbound_attempts: list[object] = []

    def unexpected_connect(*args: Any, **kwargs: Any) -> None:
        outbound_attempts.append(args)
        raise AssertionError("mock mode must not attempt an outbound connection")

    monkeypatch.setattr(socket.socket, "connect", unexpected_connect)

    class UnexpectedOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            raise AssertionError("mock mode must not construct the network client")

    monkeypatch.setattr(llm, "OpenAI", UnexpectedOpenAI)
    client = llm.create_llm_client()

    first = client.chat.completions.create(
        model="requested-model",
        messages=[{"role": "user", "content": "hello"}],
    )
    second = client.chat.completions.create(
        model="requested-model",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert client.is_mock
    assert first.model == "requested-model"
    assert first.choices[0].message.content == second.choices[0].message.content
    assert first.choices[0].message.content
    assert first.usage is None
    assert outbound_attempts == []
    assert client.complete([]).model == "mock-configured-model"


def test_remote_client_requires_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "configured-model")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="LLM_API_KEY must be set"):
        llm.create_llm_client()
