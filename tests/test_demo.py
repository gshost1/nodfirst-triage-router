"""The five-message MOCK demo CLI (`scripts/demo.py`).

These tests run entirely offline: the demo builds the canned MOCK client
directly and never reads `.env`. The shared `outbound_connections` fixture
fails the test if any provider connection is attempted.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from scripts import demo

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO_ROOT / "static" / "index.html"
DEMO_SCRIPT_RE = re.compile(
    r'<script id="demo-script" type="application/json">(.*?)</script>', re.DOTALL
)
EXPECTED_MIX = {"routine": 1, "logistics": 2, "urgent": 2}


def _db(tmp_path: Path, name: str = "demo.sqlite3") -> str:
    return str(tmp_path / name)


def test_run_demo_reports_the_scripted_mix(tmp_path: Path) -> None:
    rows = demo.run_demo(demo.build_mock_service(_db(tmp_path)))

    assert len(rows) == 5
    assert Counter(row.message_class for row in rows) == EXPECTED_MIX
    assert [row.message_class for row in rows] == [row.expected for row in rows]
    assert [row.route for row in rows] == [
        "mock",
        "mock",
        "mock",
        "mock:direct-large",
        "mock:direct-large",
    ]
    assert rows[1].tools == ("add_appointment", "assign_sibling_task")
    assert rows[0].tools == ()


def test_demo_makes_no_outbound_connection(
    tmp_path: Path, outbound_connections: list[object]
) -> None:
    rows = demo.run_demo(demo.build_mock_service(_db(tmp_path, "offline.sqlite3")))

    assert len(rows) == 5
    assert outbound_connections == []


def test_demo_ignores_a_non_mock_router_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outbound_connections: list[object]
) -> None:
    monkeypatch.setenv("ROUTER_MODE", "do")
    monkeypatch.setenv("LLM_BASE_URL", "https://inference.do-ai.run/v1")

    rows = demo.run_demo(demo.build_mock_service(_db(tmp_path, "router-env.sqlite3")))

    assert {row.route for row in rows} == {"mock", "mock:direct-large"}
    assert outbound_connections == []
    assert any("ROUTER_MODE=do" in line for line in demo.banner("demo.sqlite3"))


def test_demo_uses_exported_tier_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_SMALL", "tier-small")
    monkeypatch.setenv("MODEL_MID", "tier-mid")
    monkeypatch.setenv("MODEL_LARGE", "tier-large")

    rows = demo.run_demo(demo.build_mock_service(_db(tmp_path, "tiers.sqlite3")))

    assert [row.model for row in rows] == [
        "tier-small",
        "tier-mid",
        "tier-mid",
        "tier-large",
        "tier-large",
    ]


def test_main_prints_table_and_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = demo.main(["--db", _db(tmp_path, "main.sqlite3")])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "MOCK (offline rehearsal; canned client, no provider calls)" in out
    assert "Counts: routine 1, logistics 2, urgent 2 (5 messages)" in out
    assert re.findall(r"(?m)^([1-5])\s", out) == ["1", "2", "3", "4", "5"]
    assert "WARNING" not in out


def test_demo_cli_runs_as_a_script(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "ROUTER_MODE": "mock",
        "LLM_BASE_URL": "mock",
        "PYTHONPATH": str(REPO_ROOT),
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "demo.py"),
            "--db",
            _db(tmp_path, "cli.sqlite3"),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Counts: routine 1, logistics 2, urgent 2 (5 messages)" in completed.stdout
    assert re.findall(r"(?m)^([1-5])\s", completed.stdout) == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]


def test_demo_script_matches_the_ui_replay_script() -> None:
    match = DEMO_SCRIPT_RE.search(INDEX_PATH.read_text(encoding="utf-8"))

    assert match is not None, "the UI demo-script block is missing"
    ui_pairs = [
        (item["expect"], item["text"]) for item in json.loads(match.group(1))
    ]
    assert list(demo.DEMO_MESSAGES) == ui_pairs
