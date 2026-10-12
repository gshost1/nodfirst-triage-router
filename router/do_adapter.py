"""DigitalOcean Inference Router dispatch boundary.

Every provider request is delegated to an injected completion callable, so
tests (and the app) can substitute a test double at this boundary; this class
itself never makes a live request. DigitalOcean-specific metadata handling
lives here, not in the provider-neutral `core/llm.py` client.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from core.llm import LLMClient, RawCompletion

ROUTER_SELECTED_ROUTE_HEADER = "x-model-router-selected-route"
DEFAULT_MAX_COMPLETION_TOKENS = 512


@dataclass(frozen=True)
class RouterDispatch:
    """A router-boundary completion plus the metadata retained from it."""

    completion: Any
    requested_model: str
    response_model: str | None
    selected_route: str | None


def selected_route_header(headers: dict[str, str]) -> str | None:
    """Read the DO router's selected-route header, case-insensitively."""
    for name, value in headers.items():
        if name.lower() == ROUTER_SELECTED_ROUTE_HEADER:
            return value
    return None


class DORouterAdapter:
    """Dispatch a message through the DigitalOcean Inference Router boundary."""

    def __init__(
        self,
        complete_raw: Callable[..., RawCompletion],
        router_model: str,
    ) -> None:
        self._complete_raw = complete_raw
        self._router_model = router_model

    @classmethod
    def from_client(cls, client: LLMClient, router_model: str) -> DORouterAdapter:
        """Adapt the provider-neutral client to the router boundary."""
        return cls(client.complete_raw, router_model)

    @property
    def router_model(self) -> str:
        return self._router_model

    def dispatch(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] | None = None,
        tool_choice: Any = None,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    ) -> RouterDispatch:
        """Request a router completion and retain the available metadata.

        The actual serving model comes from the response body when present and
        the router's selected route from its response header when present;
        unavailable values stay None and are never invented. Optional agent
        arguments are forwarded only when the caller supplies them.
        """
        raw = self._complete_raw(
            messages=messages,
            model=self._router_model,
            max_completion_tokens=max_completion_tokens,
            **_forwarded(tools=tools, tool_choice=tool_choice),
        )
        model = getattr(raw.completion, "model", None)
        return RouterDispatch(
            completion=raw.completion,
            requested_model=self._router_model,
            response_model=model if isinstance(model, str) else None,
            selected_route=selected_route_header(raw.headers),
        )


def _forwarded(**kwargs: Any) -> dict[str, Any]:
    """Drop unset agent arguments so existing router callers keep working."""
    return {name: value for name, value in kwargs.items() if value is not None}
