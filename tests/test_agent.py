"""Caregiver tools run only against temporary SQLite and completion doubles."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.agent import SYSTEM_PROMPT, TOOL_SCHEMAS, CaregiverAgent
from core.mock_agent import mock_agent_completion
from core.store import CareStore


def completion(content: str = "", calls: list[Any] | None = None) -> Any:
    return SimpleNamespace(
        model="test-model",
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=content, tool_calls=calls
        ))],
        usage=None,
    )


def tool(name: str, **arguments: Any) -> Any:
    return SimpleNamespace(
        id=f"call-{name}", type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class Replies:
    def __init__(self, *responses: Any) -> None:
        self.responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return next(self.responses)


@pytest.fixture
def store(tmp_path: Path) -> CareStore:
    return CareStore(tmp_path / "care.sqlite3")


def test_seed_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "care.sqlite3"
    first = CareStore(path).snapshot()
    second = CareStore(path).snapshot()
    assert first == second
    assert first["parent"] == {"name": "Ludmila", "nickname": "Mom", "age": 81}
    assert first["siblings"] == ["Anna", "Pavel"]


def test_native_medications_preserve_taken_and_missed(store: CareStore) -> None:
    replies = Replies(
        completion(calls=[
            tool("log_med", med="Vitamin D", time="08:00", taken=True),
            tool("log_med", med="Vitamin D", time="20:00", taken=False),
        ]),
        completion("Both entries are recorded."),
    )
    result = CaregiverAgent(store, replies).run("Record both doses.")
    assert result.reply == "Both entries are recorded."
    assert len(result.tool_actions) == 2
    assert [(r["med"], r["time"], r["taken"]) for r in
            store.snapshot()["medications"]] == [
        ("Vitamin D", "08:00", True), ("Vitamin D", "20:00", False)
    ]
    assert replies.requests[0]["tools"] == TOOL_SCHEMAS
    assert replies.requests[1]["messages"][-1]["role"] == "tool"
    assert len(result.completions) == 2


def test_two_logistics_actions_in_one_turn(store: CareStore) -> None:
    replies = Replies(completion(calls=[
        tool("add_appointment", title="Mom's cardiology", datetime="Thursday"),
        tool("assign_sibling_task", sibling="Pavel", task="Drive Mom to cardiology"),
    ]), completion("All set, Pavel has the ride."))
    result = CaregiverAgent(store, replies).run(
        "Book Mom's cardiology for Thursday, Pavel drives"
    )
    assert [a.name for a in result.tool_actions] == [
        "add_appointment", "assign_sibling_task"
    ]
    data = store.snapshot()
    assert data["appointments"][0]["title"] == "Mom's cardiology"
    assert data["appointments"][0]["datetime"] == "Thursday"
    assert data["sibling_tasks"][0]["sibling"] == "Pavel"
    assert data["sibling_tasks"][0]["task"] == "Drive Mom to cardiology"


def test_summary_reflects_only_stored_values_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "care.sqlite3"
    store = CareStore(path)
    store.log_med("Vitamin D", "08:00", True)
    store.add_appointment("Cardiology", "Thursday")
    store.assign_sibling_task("Pavel", "Drive Mom")
    reopened = CareStore(path)
    replies = Replies(completion(calls=[tool("get_parent_summary")]),
                      completion("Here is Mom's update."))
    result = CaregiverAgent(reopened, replies).run("Summarize Mom")
    summary = result.tool_actions[0].result["summary"]
    for value in ["Mom", "Ludmila", "81", "Vitamin D", "08:00",
                  "Cardiology", "Thursday", "Pavel", "Drive Mom"]:
        assert value in summary
    assert "Aspirin" not in summary
    assert reopened.snapshot() == store.snapshot()
    assert "Ludmila" in result.reply


def test_json_block_fallback(store: CareStore) -> None:
    block = '```json\n{"name":"add_appointment","arguments":' \
            '{"title":"Dentist","datetime":"2026-10-15T10:00"}}\n```'
    result = CaregiverAgent(store, Replies(
        completion(block), completion("The dentist visit is saved.")
    )).run("Save the dentist visit")
    assert result.tool_actions[0].name == "add_appointment"
    assert store.snapshot()["appointments"][0]["datetime"] == "2026-10-15T10:00"


@pytest.mark.parametrize("text", ["Glad you checked in.", "```json\n{bad}\n```",
                                 '{"name":"delete_parent","arguments":{}}'])
def test_plain_or_invalid_output_does_not_mutate(store: CareStore, text: str) -> None:
    before = store.snapshot()
    result = CaregiverAgent(store, Replies(completion(text))).run("Hello")
    assert result.reply == text
    assert result.tool_actions == []
    assert store.snapshot() == before


def test_native_calls_take_priority_over_json_block(store: CareStore) -> None:
    result = CaregiverAgent(store, Replies(
        completion('{"name":"add_appointment","arguments":'
                   '{"title":"Do not save","datetime":"Friday"}}',
                   [tool("log_med", med="D", time="09:00", taken=True)]),
        completion("Recorded."),
    )).run("Record D")
    assert [a.name for a in result.tool_actions] == ["log_med"]
    assert store.snapshot()["appointments"] == []


@pytest.mark.parametrize("arguments", [
    {"med": "D", "time": "08:00", "taken": "false"},
    {"med": "D", "time": "08:00"},
    {"med": "D", "time": "08:00", "taken": True, "extra": "ignored?"},
])
def test_invalid_arguments_do_not_mutate(store: CareStore,
                                        arguments: dict[str, Any]) -> None:
    before = store.snapshot()
    result = CaregiverAgent(store, Replies(
        completion(calls=[tool("log_med", **arguments)])
    )).run("Record dose")
    assert result.tool_actions == []
    assert store.snapshot() == before


def test_urgent_guidance_is_non_diagnostic(store: CareStore) -> None:
    replies = Replies(completion("I can't assess the cause."))
    result = CaregiverAgent(store, replies).run("Mom has chest pain", urgent=True)
    assert "911" in result.reply and "doctor" in result.reply
    prompt = replies.requests[0]["messages"][0]["content"]
    assert "Never diagnose" in prompt
    assert "warm" in prompt and "concise" in prompt


def test_mock_logistics_is_offline_and_records_both_actions(
    store: CareStore, outbound_connections: list[Any],
) -> None:
    result = CaregiverAgent(store, mock_agent_completion).run(
        "Book Mom's cardiology for Thursday, Pavel drives"
    )
    assert len(result.tool_actions) == 2
    assert len(result.completions) == 2
    assert store.snapshot()["appointments"][0]["datetime"] == "Thursday"
    assert store.snapshot()["sibling_tasks"][0]["sibling"] == "Pavel"
    assert outbound_connections == []


def test_data_survives_two_separate_processes(tmp_path: Path) -> None:
    path = tmp_path / "restart.sqlite3"
    writer = """
import sys
from core.store import CareStore
store = CareStore(sys.argv[1])
store.log_med("Vitamin D", "08:00", False)
store.add_appointment("Dentist", "Thursday")
store.assign_sibling_task("Anna", "Pick up Mom")
"""
    reader = """
import json, sys
from core.store import CareStore
print(json.dumps(CareStore(sys.argv[1]).get_parent_summary()))
"""
    subprocess.run([sys.executable, "-c", writer, str(path)], check=True)
    second = subprocess.run(
        [sys.executable, "-c", reader, str(path)], check=True,
        capture_output=True, text=True,
    )
    summary = json.loads(second.stdout)
    assert summary["medications"][0]["taken"] is False
    assert summary["appointments"][0]["title"] == "Dentist"
    assert summary["sibling_tasks"][0]["sibling"] == "Anna"
    for value in ("Ludmila", "81", "Vitamin D", "missed", "Dentist", "Pick up Mom"):
        assert value in summary["summary"]


def test_schema_allowlist_and_required_types() -> None:
    functions = {schema["function"]["name"]: schema["function"]
                 for schema in TOOL_SCHEMAS}
    assert set(functions) == {
        "log_med", "add_appointment", "assign_sibling_task", "get_parent_summary",
        "summarize_day",
    }
    assert functions["log_med"]["parameters"]["properties"]["taken"] == {
        "type": "boolean"
    }
    for function in functions.values():
        parameters = function["parameters"]
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(parameters["properties"])


def test_invalid_native_call_gets_error_alongside_valid_result(
    store: CareStore,
) -> None:
    replies = Replies(completion(calls=[
        tool("delete_parent"),
        tool("log_med", med="D", time="08:00", taken=True),
    ]), completion("Recorded D only."))
    result = CaregiverAgent(store, replies).run("Record D")
    assert [action.name for action in result.tool_actions] == ["log_med"]
    tool_messages = [m for m in replies.requests[-1]["messages"]
                     if m["role"] == "tool"]
    assert len(tool_messages) == 2
    assert "error" in json.loads(tool_messages[0]["content"])


def test_followup_cannot_trigger_unbounded_tool_execution(store: CareStore) -> None:
    replies = Replies(
        completion(calls=[tool("add_appointment", title="Dentist", datetime="Friday")]),
        completion("Saved.", [
            tool("add_appointment", title="Unexpected", datetime="Saturday")
        ]),
    )
    result = CaregiverAgent(store, replies).run("Save Friday's dentist visit")
    assert len(result.completions) == 2
    assert len(result.tool_actions) == 1
    assert len(store.snapshot()["appointments"]) == 1
    assert replies.requests[-1]["tool_choice"] == "none"


def test_empty_summary_has_no_invented_records(store: CareStore) -> None:
    result = CaregiverAgent(store, Replies(
        completion(calls=[tool("get_parent_summary")]),
        completion("Mom takes Aspirin daily and has a dentist appointment."),
    )).run("Mom's summary")
    assert "Mom (Ludmila, 81)" in result.reply
    assert result.reply.count("none recorded") == 3
    assert "Aspirin" not in result.reply


def test_urgent_turns_offer_no_tools_and_keep_the_escalation_reply(tmp_path: Path) -> None:
    """Live qwen3.8-max reached for get_parent_summary on 'Mom had a fall'."""
    store = CareStore(tmp_path / "u.sqlite3")
    before = store.snapshot()
    replies = Replies(completion("Please call 911 now, then let Anna and Pavel know."))

    result = CaregiverAgent(store, replies).run("Mom had a fall", urgent=True)

    assert "tools" not in replies.requests[0]
    assert result.tool_actions == []
    assert result.reply.startswith("Please call 911 now")
    assert store.snapshot() == before
    assert "only when the user asks for a" in SYSTEM_PROMPT
    assert "do not ask for the time" in SYSTEM_PROMPT


def test_empty_model_reply_gets_an_honest_fallback(tmp_path: Path) -> None:
    """A thinking model that spends its budget on reasoning returns no text."""
    store = CareStore(tmp_path / "e.sqlite3")
    result = CaregiverAgent(store, Replies(completion(""))).run("How was Mom today?")
    assert result.tool_actions == []
    assert "couldn't put a reply together" in result.reply
    assert "911" in result.reply


def test_voice_turns_ask_for_short_spoken_replies(tmp_path: Path) -> None:
    store = CareStore(tmp_path / "v.sqlite3")
    replies = Replies(completion("Pavel is driving Mom on Thursday."))
    CaregiverAgent(store, replies).run("Who's driving Mom?", voice=True)
    prompt = replies.requests[0]["messages"][0]["content"]
    assert "read aloud" in prompt and "one short plain sentence" in prompt
    assert replies.requests[0]["max_completion_tokens"] == 80


def test_question_about_records_keeps_the_models_answer(tmp_path: Path) -> None:
    """'Who's driving Mom?' must not be answered with the whole summary dump."""
    store = CareStore(tmp_path / "q.sqlite3")
    store.assign_sibling_task("Pavel", "Drive Mom to cardiology on Thursday")
    replies = Replies(
        completion(calls=[tool("get_parent_summary")]),
        completion("Pavel is driving Mom to cardiology on Thursday."),
    )

    result = CaregiverAgent(store, replies).run("Who's driving Mom today?")

    assert [a.name for a in result.tool_actions] == ["get_parent_summary"]
    assert result.reply == "Pavel is driving Mom to cardiology on Thursday."

    asked = Replies(
        completion(calls=[tool("get_parent_summary")]),
        completion("Here is a made-up summary with an invented appointment."),
    )
    summary = CaregiverAgent(store, asked).run("Give me a summary of Mom")
    assert summary.reply.startswith("Here's the recorded update for Mom")
