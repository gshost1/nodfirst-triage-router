"""Execute route decisions against the model boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from core.llm import LLMClient
from router.config import RouterMode
from router.dispatch import RouteDecision
from router.do_adapter import DEFAULT_MAX_COMPLETION_TOKENS, DORouterAdapter


@dataclass(frozen=True)
class CompletionOutcome:
    """The completion produced for a route decision plus response metadata.

    `response_model` is the actual serving model reported by the provider when
    available, and `selected_route` is the DigitalOcean router's selected
    route header when present; both stay None when the provider omits them.
    """

    decision: RouteDecision
    completion: Any
    response_model: str | None
    selected_route: str | None


class RouteExecutor:
    """Execute a route decision.

    DO-mode routine/logistics decisions go through the DO router adapter;
    every other decision — and every urgent decision in every mode — goes
    directly to the selected model and never touches a router boundary. The
    agent's optional `tools`/`tool_choice` arguments are forwarded only when
    the caller supplies them.
    """

    def __init__(self, client: LLMClient, adapter: DORouterAdapter) -> None:
        self._client = client
        self._adapter = adapter

    def execute(
        self,
        decision: RouteDecision,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] | None = None,
        tool_choice: Any = None,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    ) -> CompletionOutcome:
        forwarded = _forwarded(tools=tools, tool_choice=tool_choice)
        if decision.mode is RouterMode.DO and not decision.is_urgent:
            dispatched = self._adapter.dispatch(
                messages,
                max_completion_tokens=max_completion_tokens,
                **forwarded,
            )
            return CompletionOutcome(
                decision=decision,
                completion=dispatched.completion,
                response_model=dispatched.response_model,
                selected_route=dispatched.selected_route,
            )

        raw = self._client.complete_raw(
            messages=messages,
            model=decision.model,
            max_completion_tokens=max_completion_tokens,
            **forwarded,
        )
        model = getattr(raw.completion, "model", None)
        return CompletionOutcome(
            decision=decision,
            completion=raw.completion,
            response_model=model if isinstance(model, str) else None,
            selected_route=None,
        )


def _forwarded(**kwargs: Any) -> dict[str, Any]:
    """Drop unset agent arguments so existing callers keep their signature."""
    return {name: value for name, value in kwargs.items() if value is not None}
