"""Provider-neutral OpenAI SDK client with a local canned-response mode."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

REQUEST_TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 2
MOCK_BASE_URL = "mock"
MOCK_MODEL = "mock"
MOCK_REPLY = "I’m here to help. What would be most useful right now?"


@dataclass
class MockMessage:
    role: str
    content: str
    tool_calls: list[object] | None = None


@dataclass
class MockChoice:
    index: int
    message: MockMessage
    finish_reason: str


@dataclass
class MockCompletion:
    id: str
    model: str
    choices: list[MockChoice]
    usage: None = None


class _MockCompletions:
    def __init__(self, default_model: str) -> None:
        self._default_model = default_model

    def create(
        self,
        *,
        model: str | None = None,
        messages: Sequence[object] | None = None,
        **_kwargs: Any,
    ) -> MockCompletion:
        del messages
        return MockCompletion(
            id="mock-completion",
            model=model or self._default_model,
            choices=[
                MockChoice(
                    index=0,
                    message=MockMessage(role="assistant", content=MOCK_REPLY),
                    finish_reason="stop",
                )
            ],
        )


class LLMClient:
    """Expose the OpenAI chat-completions surface in real and mock modes."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.model = model
        self.is_mock = base_url.lower() == MOCK_BASE_URL
        self._sdk_client: OpenAI | None = None

        if self.is_mock:
            self.chat: Any = SimpleNamespace(
                completions=_MockCompletions(default_model=model)
            )
            return

        if not api_key:
            raise ValueError("LLM_API_KEY must be set when using a remote endpoint.")
        if not model:
            raise ValueError("LLM_MODEL must be set when using a remote endpoint.")

        self._sdk_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
        )
        self.chat = self._sdk_client.chat

    def complete(
        self,
        messages: Sequence[Any],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Use the configured model unless the caller selects another model."""
        return self.chat.completions.create(
            model=model or self.model, messages=messages, **kwargs
        )

    def close(self) -> None:
        if self._sdk_client is not None:
            self._sdk_client.close()


def create_llm_client() -> LLMClient:
    """Create a client from environment settings, loading a local `.env` if present."""
    load_dotenv()
    base_url = os.getenv("LLM_BASE_URL", MOCK_BASE_URL).strip()
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "").strip()

    if not base_url:
        raise ValueError("LLM_BASE_URL must not be empty.")
    if base_url.lower() == MOCK_BASE_URL:
        model = model or MOCK_MODEL

    return LLMClient(base_url=base_url, api_key=api_key, model=model)
