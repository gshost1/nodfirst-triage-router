"""Warm caregiver agent with native, JSON-block, and plain-reply fallbacks."""

from __future__ import annotations

import json
import re
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
    "rather than inventing a calendar date: if only a day is given, record the "
    "appointment with that day as the datetime and do not ask for the time. "
    "Sibling tasks recorded under 'unassigned' are open; when the user assigns "
    "one (for example 'Pavel takes the cardiology drive'), call "
    "assign_sibling_task with the sibling's name and that task's wording, and "
    "it claims the open task. "
    "Use only the seven supplied tools. Email tools (email_digest, "
    "email_siblings) only when the user asks to send or email something; "
    "report the email status from the tool result, and if it says disabled "
    "or failed, say the email did not go out. "
    "Call get_parent_summary or summarize_day only when the user asks for a "
    "summary, update, or digest, asks how Mom's day is going, or asks a question about recorded medications, "
    "appointments, or sibling tasks (for example who is driving Mom); never for an "
    "urgent message, a check-in, or a request to record something. Answer such "
    "questions directly from the tool result in one or two sentences. "
    "If native tool calls are unavailable, return a fenced JSON block containing "
    '{"name":"approved_tool_name","arguments":{...}}, or a list of such objects. '
    "If no tool is needed, respond in plain text. Summaries must reflect only "
    "the data returned by get_parent_summary or summarize_day."
)


# Shown when a model returns no text at all (for example a reasoning budget
# exhausted before any answer). Honest, and never pretends an action happened.
EMPTY_REPLY_FALLBACK = (
    "I'm here with you, but I couldn't put a reply together just now. "
    "Please send that again. If this could be an emergency, call 911 now."
)

# Replies for these tools are the store's own text, never a model paraphrase,
# but only when the user asked for a summary or digest. A question such as
# "who's driving Mom today?" keeps the model's short answer, grounded by the
# tool result in context.
GROUNDED_SUMMARY_TOOLS = frozenset({"get_parent_summary", "summarize_day"})
SUMMARY_REQUEST = re.compile(r"\b(summar\w*|digest|recap|overview|full update)\b", re.IGNORECASE)

# Urgent turns: the model gets a brief, not the note itself, so it does not
# echo the 911 line that the app appends (live 2026-10-12 the reply carried
# the guidance twice). If the model does say "call 911" anyway, the app's
# note is skipped.
URGENT_MODEL_BRIEF = (
    "This message may describe an emergency. The app appends fixed guidance "
    "to call 911 or Mom's doctor after your reply, so for this turn ignore the "
    "rule above about telling the user to call 911: do not give that "
    "guidance yourself and do not judge how serious it is. Reply in one or two "
    "short sentences: acknowledge what happened and name one practical next "
    "step for the family (stay with Mom, keep the phone nearby, tell the "
    "siblings)."
)
URGENT_ESCALATION_NOTE_VOICE = (
    "If this could be an emergency, call 911 now. Otherwise call Mom's doctor "
    "right away; do not wait for me."
)
ALREADY_ESCALATED = re.compile(r"\b911\b", re.IGNORECASE)
VOICE_MAX_COMPLETION_TOKENS = 80  # keeps spoken replies to a sentence or two
VOICE_STYLE = (
    " The caregiver is speaking and will hear your reply read aloud: answer in "
    "one short plain sentence, two at most, under 30 words in all, no lists, "
    "no markdown, no emoji. Lead with the answer, skip preamble."
)


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

    def __init__(
        self,
        store: CareStore,
        complete: Callable[..., Any],
        notifier: Any = None,
    ) -> None:
        self.store = store
        self.complete = complete
        self.notifier = notifier

    def run(
        self,
        message: str,
        *,
        urgent: bool = False,
        sender: str | None = None,
        history: Sequence[dict[str, Any]] = (),
        voice: bool = False,
    ) -> AgentResult:
        """Run one turn.

        `history` is prior thread messages (dicts with `role`, `content`, and
        optionally `sender`) in chronological order. User turns are prefixed
        with the sender's name so the model can tell siblings apart.
        """
        system = SYSTEM_PROMPT
        if sender:
            system += f" The person writing now is {sender}."
        if voice:
            system += VOICE_STYLE
        if urgent:
            system += "\n" + URGENT_MODEL_BRIEF
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
        # An urgent turn gets no tools: the reply must be the escalation, not a
        # record lookup (live qwen3.8-max otherwise calls get_parent_summary
        # first and can come back with no text at all).
        budget = VOICE_MAX_COMPLETION_TOKENS if voice else 512
        if urgent:
            initial = self.complete(messages=messages, max_completion_tokens=budget)
        else:
            initial = self.complete(
                messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto",
                max_completion_tokens=budget,
            )
        completions = [initial]
        response = getattr(initial, "completion", initial)
        assistant = response.choices[0].message
        text = assistant.content or ""
        calls = getattr(assistant, "tool_calls", None)
        if urgent:
            calls, requests = None, []
        else:
            requests = native_requests(calls) if calls else json_requests(text)
        actions = [execute_tool(self.store, request, self.notifier) for request in requests]
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
                max_completion_tokens=budget,
            )
            completions.append(followup)
            final = getattr(followup, "completion", followup)
            text = final.choices[0].message.content or "I've recorded that for you."
            # Store-grounded summaries cannot gain invented entries in a follow-up.
            # An urgent message keeps the model's reply: a fall must never be
            # answered with a digest, even if the model called a summary tool.
            summaries = [a.result["summary"] for a in actions
                         if a.name in GROUNDED_SUMMARY_TOOLS]
            emailed = any(action.name.startswith("email_") for action in actions)
            if summaries and not urgent and not emailed and SUMMARY_REQUEST.search(message):
                text = "\n".join(summaries)
        if not text.strip():
            text = EMPTY_REPLY_FALLBACK
        if urgent and not ALREADY_ESCALATED.search(text):
            note = URGENT_ESCALATION_NOTE_VOICE if voice else URGENT_ESCALATION_NOTE
            text = f"{text}\n\n{note}".strip()
        return AgentResult(text, actions, completions)
