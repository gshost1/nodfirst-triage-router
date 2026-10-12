"""Demo week seeding."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app
from core.demo_data import APPOINTMENTS, THREAD, is_seeded, seed_demo
from core.store import CareStore
from tests.test_digest import service


def test_seed_is_idempotent_and_grounds_the_digest(tmp_path: Path) -> None:
    store = CareStore(tmp_path / "s.sqlite3")
    assert not is_seeded(store)
    first = seed_demo(store)
    assert first["seeded"] is True and is_seeded(store)
    assert seed_demo(store)["seeded"] is False
    snapshot = store.snapshot()
    assert len(snapshot["appointments"]) == len(APPOINTMENTS)
    assert {t["sibling"] for t in snapshot["sibling_tasks"]} == {"unassigned", "Anna", "Pavel"}
    assert len(store.open_tasks()) == 3
    claimed = store.assign_sibling_task("Pavel", "drive Mom to cardiology Thursday")
    assert claimed["claimed_open_task"] is True and "10:30" in claimed["task"]
    assert len(store.open_tasks()) == 2
    fresh = store.assign_sibling_task("Anna", "Bake a birthday cake")
    assert "claimed_open_task" not in fresh and len(store.open_tasks()) == 13
    assert [m["content"] for m in store.recent_messages(50)] == [row[2] for row in THREAD]
    digest = store.daily_digest()
    assert "metformin" in digest["summary"] and digest["alerts_total"] == 1
    assert store.alerts()[0]["acknowledged"] is False
    assert store.alerts()[0]["email_status"] == "skipped"


def test_demo_routes(tmp_path: Path) -> None:
    api = TestClient(create_app(service=service(tmp_path)))
    before = api.get("/demo").json()
    assert before["seeded"] is False
    assert api.post("/demo/seed").json()["seeded"] is True
    assert api.get("/demo").json()["seeded"] is True
    assert api.post("/demo/seed").json()["seeded"] is False
    assert len(api.get("/thread").json()["messages"]) == len(THREAD)


def test_ui_has_demo_seed(tmp_path: Path) -> None:
    html = TestClient(create_app(service=service(tmp_path))).get("/").text
    for required in ('id="demo-seed"', "fetch('/demo'", "fetch('/demo/seed'"):
        assert required in html, f"the served page is missing {required}"
