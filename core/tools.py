"""The four approved tool schemas and strict argument parsing."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.store import CareStore

TOOL_PARAMETERS: dict[str, dict[str, str]] = {
    "log_med": {"med": "string", "time": "string", "taken": "boolean"},
    "add_appointment": {"title": "string", "datetime": "string"},
    "assign_sibling_task": {"sibling": "string", "task": "string"},
    "get_parent_summary": {},
}
TOOL_DESCRIPTIONS = {
    "log_med": "Record a medication as taken or missed, at the requested time.",
    "add_appointment": "Save an appointment with the requested title and datetime.",
    "assign_sibling_task": "Record a sibling's requested caregiving task.",
    "get_parent_summary": "Read Mom's identity and recorded caregiving data.",
}
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": name, "description": TOOL_DESCRIPTIONS[name],
        "parameters": {
            "type": "object",
            "properties": {key: {"type": kind} for key, kind in parameters.items()},
            "required": list(parameters), "additionalProperties": False,
        },
    }}
    for name, parameters in TOOL_PARAMETERS.items()
]


@dataclass(frozen=True)
class ToolRequest:
    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass(frozen=True)
class ToolAction:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


def validate_request(name: Any, arguments: Any, call_id: str) -> ToolRequest | None:
    if not isinstance(name, str) or name not in TOOL_PARAMETERS:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            return None
    parameters = TOOL_PARAMETERS[name]
    if not isinstance(arguments, dict) or set(arguments) != set(parameters):
        return None
    for key, kind in parameters.items():
        value = arguments[key]
        if kind == "boolean":
            if type(value) is not bool:
                return None
        elif not isinstance(value, str) or not value.strip():
            return None
    return ToolRequest(name, arguments, call_id)


def native_requests(calls: list[Any]) -> list[ToolRequest]:
    requests = []
    for index, call in enumerate(calls):
        function = getattr(call, "function", None)
        request = validate_request(
            getattr(function, "name", None), getattr(function, "arguments", None),
            getattr(call, "id", None) or f"call-{index}",
        )
        if request is not None:
            requests.append(request)
    return requests


def json_requests(content: str) -> list[ToolRequest]:
    """Accept a whole JSON object/list or fenced JSON, never arbitrary prose."""
    blocks = re.findall(
        r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE
    )
    for block in blocks or [content.strip()]:
        try:
            payload = json.loads(block)
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and "tool_calls" in payload:
            payload = payload["tool_calls"]
        items = payload if isinstance(payload, list) else [payload]
        requests = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            function = item.get("function", item)
            if not isinstance(function, dict):
                continue
            request = validate_request(
                function.get("name"), function.get("arguments"), f"json-{index}"
            )
            if request is not None:
                requests.append(request)
        if requests:
            return requests
    return []


def execute_tool(store: CareStore, request: ToolRequest) -> ToolAction:
    # Dispatch only through the validated allowlist, never model-selected attributes.
    handlers: dict[str, Callable[..., dict[str, Any]]] = {
        "log_med": store.log_med,
        "add_appointment": store.add_appointment,
        "assign_sibling_task": store.assign_sibling_task,
        "get_parent_summary": store.get_parent_summary,
    }
    result = handlers[request.name](**request.arguments)
    return ToolAction(request.name, request.arguments, result)
