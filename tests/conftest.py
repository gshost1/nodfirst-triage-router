"""Shared fixtures for the routing test suite."""

from __future__ import annotations

import socket
from typing import Any

import pytest

from core import llm


@pytest.fixture(autouse=True)
def isolate_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must never load the user's `.env` file."""
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)


@pytest.fixture
def outbound_connections(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Fail on any outbound socket connection and record the attempts."""
    attempts: list[Any] = []

    def unexpected_connect(*args: Any, **kwargs: Any) -> None:
        attempts.append(args)
        raise AssertionError("no outbound provider connection is allowed")

    monkeypatch.setattr(socket.socket, "connect", unexpected_connect)
    return attempts


@pytest.fixture
def routine_message() -> str:
    return "How did Mom's visit with her neighbor go yesterday?"


@pytest.fixture
def logistics_message() -> str:
    return "Book Mom's cardiology for Thursday, Pavel drives"


@pytest.fixture
def urgent_messages() -> dict[str, str]:
    """One message per approved urgent keyword category."""
    return {
        "falls": "Mom had a fall in the kitchen this morning.",
        "chest pain": "Mom called and says she has chest pain.",
        "confusion": "Mom sounds confused and isn't making sense on the phone.",
        "missed dose": "Mom missed her dose of blood pressure medication.",
        "cannot reach": "I can't reach Mom and she isn't answering the phone.",
    }


@pytest.fixture
def tier_env() -> dict[str, str]:
    """Distinct configured tier IDs so hardcoded tier selection cannot pass."""
    return {
        "ROUTER_MODE": "client",
        "MODEL_SMALL": "distinct-small-model",
        "MODEL_MID": "distinct-mid-model",
        "MODEL_LARGE": "distinct-large-model",
    }
