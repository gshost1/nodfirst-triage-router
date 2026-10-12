"""Warm caregiver agent with native, JSON-block, and plain-reply fallbacks."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from core.store import CareStore
from core.tools import (
    TOOL_SCHEMAS,
    ToolAction,
    execute_tool,
    json_requests,
    native_requests,
)
from router.dispatch import URGENT_ESCALATION_NOTE

SYSTEM_PROMPT = (
    "You are a warm, concise caregiver assistant for Mom (Ludmila, 81), "
    "with siblings Anna and Pavel. Never diagnose or suggest medication changes. "
    "This is not medical advice. For possible emergencies direct the user to "
    "call 911 now; otherwise contact a doctor promptly. "
    "Only record values the user requested, and never claim an action succeeded "
    "until its tool result confirms it. Preserve relative dates such as Thursday "
    "rather than inventing a calendar date. Use only the five supplied tools. "
    "Call get_parent_summary or summarize_day only when the user asks for a "
    "summary, update, or digest; never for an urgent message, a check-in, or a "
    "request to record something. "
    "If native tool calls are unavailable, return a fenced JSON block containing "
    '{"name":"approved_tool_name","arguments":{...}}, or a list of such objects. '
    "If no tool is needed, respond in plain text. Summaries must reflect only "
    "the data returned by get_parent_summary or summarize_day."
)


# Replies for these tools are the store's own text, never a model paraphrase.
GROUNDED_SUMMARY_TOOLS = frozenset({"get_parent_summary", "summarize_day"})


@dataclass(frozen=True)
class AgentResult:
    reply: str
    tool_actions: list[ToolAction]
    completions: list[Any]


class CaregiverAgent:
    """Inject a completion callable to preserve routing and per-call metrics.

    The callable accepts messages, tools, tool_choice, max_completion_tokens.
    It may return a completion directly or a router CompletionOutcome.
    Every invocation is retained in AgentResult.completions for the API.
    """

    def __init__(self, store: CareStore, complete: Callable[..., Any]) -> None:
        self.store = store
        self.complete = complete

    def run(
        self,
        message: str,
        *,
        urgent: bool = False,
        sender: str | None = None,
        history: Sequence[dict[str, Any]] = (),
    ) -> AgentResult:
        """Run one turn.

        `history` is prior thread messages (dicts with `role`, `content`, and
        optionally `sender`) in chronological order. User turns are prefixed
        with the sender's name so the model can tell siblings apart.
        """
        system = SYSTEM_PROMPT
        if sender:
            system += f" The person writing now is {sender}."
        if urgent:
            system += "\n" + URGENT_ESCALATION_NOTE
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for prior in history:
            role = prior["role"]
            content = str(prior.get("content", ""))
            if role == "user" and prior.get("sender"):
                content = f"{prior['sender']}: {content}"
            messages.append({"role": role, "content": content})
        messages.append({
            "role": "user",
            "content": f"{sender}: {message}" if sender else message,
        })
        initial = self.complete(
            messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto",
            max_completion_tokens=512,
        )
        completions = [initial]
        response = getattr(initial, "completion", initial)
        assistant = response.choices[0].message
        text = assistant.content or ""
        calls = getattr(assistant, "tool_calls", None)
        requests = native_requests(calls) if calls else json_requests(text)
        actions = [execute_tool(self.store, request) for request in requests]
        if actions:
            if calls:
                # Keep the original native calls so tool_call_id references match.
                messages.append({
                    "role": "assistant", "content": text or None,
                    "tool_calls": [
                        call.model_dump(exclude_none=True)
                        if hasattr(call, "model_dump") else {
                            "id": call.id, "type": "function",
                            "function": {"name": call.function.name,
                                         "arguments": call.function.arguments},
                        }
                        for call in calls
                    ],
                })
                results = {request.call_id: action.result
                           for request, action in zip(requests, actions)}
                for call in calls:
                    messages.append({
                        "role": "tool", "tool_call_id": call.id,
                        "content": json.dumps(results.get(call.id, {
                            "error": "Tool name or arguments were not approved; "
                                     "nothing was recorded for this call."
                        })),
                    })
            else:
                messages.extend([
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "Tool results: " + json.dumps([
                        {"name": action.name, "result": action.result}
                        for action in actions
                    ]) + "\nBriefly confirm these recorded results."},
                ])
            followup = self.complete(
                messages=messages, tools=TOOL_SCHEMAS, tool_choice="none",
                max_completion_tokens=512,
            )
            completions.append(followup)
            final = getattr(followup, "completion", followup)
            text = final.choices[0].message.content or "I've recorded that for you."
            # Store-grounded summaries cannot gain invented entries in a follow-up.
            # An urgent message keeps the model's reply: a fall must never be
            # answered with a digest, even if the model called a summary tool.
            summaries = [a.result["summary"] for a in actions
                         if a.name in GROUNDED_SUMMARY_TOOLS]
            if summaries and not urgent:
                text = "\n".join(summaries)
        if urgent:
            text = f"{text}\n\n{URGENT_ESCALATION_NOTE}".strip()
        return AgentResult(text, actions, completions)
