"""Offline caregiver rehearsal, separate from the provider-neutral client."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from core.llm import MockChoice, MockCompletion, MockMessage


def mock_agent_completion(
    *,
    messages: Sequence[dict[str, Any]],
    model: str = "mock",
    **_kwargs: Any,
) -> MockCompletion:
    """Canned replies and the approved two-action logistics example.

    This is a demo double, not a natural-language date parser or live inference.
    Thursday remains Thursday, with no invented appointment time.
    """
    text = str(messages[-1].get("content", ""))
    if messages[-1]["role"] == "tool" or text.startswith("Tool results:"):
        reply = "I've recorded that for you. You're keeping everyone in the loop."
    elif "book mom's cardiology for thursday, pavel drives" in text.lower():
        reply = "```json\n" + json.dumps([
            {"name": "add_appointment", "arguments": {
                "title": "Mom's cardiology", "datetime": "Thursday",
            }},
            {"name": "assign_sibling_task", "arguments": {
                "sibling": "Pavel", "task": "Drive Mom to cardiology on Thursday",
            }},
        ]) + "\n```"
    elif "digest" in text.lower():
        reply = '```json\n{"name":"summarize_day","arguments":{}}\n```'
    elif "summary" in text.lower() or "summarize" in text.lower():
        reply = '```json\n{"name":"get_parent_summary","arguments":{}}\n```'
    else:
        reply = (
            "I'm here with you. A brief check-in with Mom and a shared update "
            "with Anna and Pavel can help everyone stay connected."
        )
    return MockCompletion(
        id="mock-caregiver", model=model,
        choices=[MockChoice(
            index=0, message=MockMessage(role="assistant", content=reply),
            finish_reason="stop",
        )],
    )
