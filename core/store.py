"""SQLite caregiver demo data, with short-lived connections per operation."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

UNASSIGNED = "unassigned"
_STOPWORDS = frozenset(
    ["the", "a", "an", "to", "for", "of", "on", "at", "in", "and", "or", "with", "her", "his", "mom", "mom's", "moms", "this", "that", "next", "week", "weekly", "every", "please", "take", "bring", "go", "get"]
)


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


class CareStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with closing(self._connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS parent (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    name TEXT NOT NULL, nickname TEXT NOT NULL, age INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS siblings (name TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS medications (
                    id INTEGER PRIMARY KEY, med TEXT NOT NULL, time TEXT NOT NULL,
                    taken INTEGER NOT NULL CHECK (taken IN (0, 1))
                );
                CREATE TABLE IF NOT EXISTS appointments (
                    id INTEGER PRIMARY KEY, title TEXT NOT NULL, datetime TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sibling_tasks (
                    id INTEGER PRIMARY KEY, sibling TEXT NOT NULL, task TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY,
                    sender TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    message_class TEXT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY,
                    sender TEXT NOT NULL,
                    message TEXT NOT NULL,
                    note TEXT NOT NULL,
                    acknowledged INTEGER NOT NULL DEFAULT 0 CHECK (acknowledged IN (0, 1)),
                    email_status TEXT,
                    email_detail TEXT,
                    email_recipients TEXT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );
                INSERT OR IGNORE INTO parent VALUES (1, 'Ludmila', 'Mom', 81);
                INSERT OR IGNORE INTO siblings VALUES ('Anna'), ('Pavel');
            """)
            for column in ("model", "route"):
                try:
                    with db:
                        db.execute(f"ALTER TABLE messages ADD COLUMN {column} TEXT")
                except sqlite3.OperationalError:
                    pass  # column already there

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def _insert(self, sql: str, values: tuple[Any, ...]) -> int:
        db = self._connect()
        try:
            with db:
                cursor = db.execute(sql, values)
                assert cursor.lastrowid is not None
                return cursor.lastrowid
        finally:
            db.close()

    def log_med(self, med: str, time: str, taken: bool) -> dict[str, Any]:
        identifier = self._insert(
            "INSERT INTO medications (med, time, taken) VALUES (?, ?, ?)",
            (med, time, int(taken)),
        )
        return {"id": identifier, "med": med, "time": time, "taken": taken}

    def add_appointment(self, title: str, datetime: str) -> dict[str, Any]:
        identifier = self._insert(
            "INSERT INTO appointments (title, datetime) VALUES (?, ?)",
            (title, datetime),
        )
        return {"id": identifier, "title": title, "datetime": datetime}

    def assign_sibling_task(self, sibling: str, task: str) -> dict[str, Any]:
        """Record a task for a sibling; an open ('unassigned') task with the
        same gist is claimed rather than duplicated."""
        match = self._open_task_matching(task)
        if match is not None and sibling.strip().lower() != UNASSIGNED:
            db = self._connect()
            try:
                with db:
                    db.execute(
                        "UPDATE sibling_tasks SET sibling = ? WHERE id = ?",
                        (sibling, match["id"]),
                    )
            finally:
                db.close()
            return {"id": match["id"], "sibling": sibling, "task": match["task"],
                    "claimed_open_task": True}
        identifier = self._insert(
            "INSERT INTO sibling_tasks (sibling, task) VALUES (?, ?)",
            (sibling, task),
        )
        return {"id": identifier, "sibling": sibling, "task": task}

    def open_tasks(self) -> list[dict[str, Any]]:
        db = self._connect()
        try:
            return [dict(r) for r in db.execute(
                "SELECT * FROM sibling_tasks WHERE lower(sibling) = ? ORDER BY id",
                (UNASSIGNED,),
            )]
        finally:
            db.close()

    def _open_task_matching(self, task: str) -> dict[str, Any] | None:
        wanted = _keywords(task)
        if not wanted:
            return None
        best: tuple[float, dict[str, Any]] | None = None
        for row in self.open_tasks():
            have = _keywords(row["task"])
            if not have:
                continue
            overlap = len(wanted & have) / min(len(wanted), len(have))
            if overlap >= 0.5 and (best is None or overlap > best[0]):
                best = (overlap, row)
        return best[1] if best else None

    # -- shared sibling thread ------------------------------------------------

    def add_message(
        self, sender: str, role: str, content: str, message_class: str | None = None,
        model: str | None = None, route: str | None = None,
    ) -> dict[str, Any]:
        identifier = self._insert(
            "INSERT INTO messages (sender, role, content, message_class, model, route) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sender, role, content, message_class, model, route),
        )
        return {"id": identifier, "sender": sender, "role": role,
                "content": content, "message_class": message_class,
                "model": model, "route": route}

    def recent_messages(self, limit: int = 20) -> list[dict[str, Any]]:
        """The last `limit` thread messages in chronological order."""
        db = self._connect()
        try:
            rows = db.execute(
                "SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in reversed(rows)]
        finally:
            db.close()

    def siblings(self) -> list[str]:
        db = self._connect()
        try:
            return [r["name"] for r in db.execute(
                "SELECT name FROM siblings ORDER BY name"
            )]
        finally:
            db.close()

    # -- urgent alerts for the sibling group ---------------------------------

    def add_alert(self, sender: str, message: str, note: str) -> dict[str, Any]:
        identifier = self._insert(
            "INSERT INTO alerts (sender, message, note) VALUES (?, ?, ?)",
            (sender, message, note),
        )
        return {"id": identifier, "sender": sender, "message": message,
                "note": note, "acknowledged": False, "email_status": None,
                "email_detail": None, "email_recipients": []}

    def record_alert_email(
        self, alert_id: int, status: str, detail: str, recipients: list[str]
    ) -> None:
        db = self._connect()
        try:
            with db:
                db.execute(
                    "UPDATE alerts SET email_status = ?, email_detail = ?, "
                    "email_recipients = ? WHERE id = ?",
                    (status, detail, ", ".join(recipients), alert_id),
                )
        finally:
            db.close()

    @staticmethod
    def _alert_row(row: Any) -> dict[str, Any]:
        result = dict(row)
        result["acknowledged"] = bool(result["acknowledged"])
        raw = result.get("email_recipients") or ""
        result["email_recipients"] = [r for r in raw.split(", ") if r]
        return result

    def alerts(self) -> list[dict[str, Any]]:
        db = self._connect()
        try:
            return [self._alert_row(r) for r in db.execute(
                "SELECT * FROM alerts ORDER BY id DESC"
            )]
        finally:
            db.close()

    def acknowledge_alert(self, alert_id: int) -> dict[str, Any] | None:
        db = self._connect()
        try:
            with db:
                cursor = db.execute(
                    "UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,)
                )
                if cursor.rowcount == 0:
                    return None
            row = db.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
            return self._alert_row(row)
        finally:
            db.close()

    def snapshot(self) -> dict[str, Any]:
        db = self._connect()
        try:
            parent = dict(db.execute(
                "SELECT name, nickname, age FROM parent WHERE id = 1"
            ).fetchone())
            result: dict[str, Any] = {
                "parent": parent,
                "siblings": [r["name"] for r in db.execute(
                    "SELECT name FROM siblings ORDER BY name"
                )],
            }
            for table in ("medications", "appointments", "sibling_tasks"):
                result[table] = [dict(row) for row in db.execute(
                    f"SELECT * FROM {table} ORDER BY id"
                )]
            for row in result["medications"]:
                row["taken"] = bool(row["taken"])
            return result
        finally:
            db.close()

    def daily_digest(self, hours: int = 24) -> dict[str, Any]:
        """A store-grounded digest: last-`hours` thread activity, open alerts,
        and everything recorded for Mom. Nothing here comes from a model."""
        data = self.snapshot()
        db = self._connect()
        try:
            since = f"-{int(hours)} hours"
            counts = {
                row["message_class"] or "unclassified": row["n"]
                for row in db.execute(
                    "SELECT message_class, COUNT(*) AS n FROM messages "
                    "WHERE role = 'user' AND created_at >= strftime("
                    "'%Y-%m-%dT%H:%M:%fZ', 'now', ?) GROUP BY message_class",
                    (since,),
                )
            }
            senders = [row["sender"] for row in db.execute(
                "SELECT DISTINCT sender FROM messages WHERE role = 'user' AND "
                "created_at >= strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?) "
                "ORDER BY sender", (since,),
            )]
            alerts = [self._alert_row(r) for r in db.execute(
                "SELECT * FROM alerts WHERE created_at >= strftime("
                "'%Y-%m-%dT%H:%M:%fZ', 'now', ?) ORDER BY id", (since,),
            )]
        finally:
            db.close()
        open_alerts = [a for a in alerts if not a["acknowledged"]]
        parent = data["parent"]
        heading = (
            f"Daily digest for {parent['nickname']} ({parent['name']}, "
            f"{parent['age']}), last {hours} hours."
        )
        lines = [heading]
        total = sum(counts.values())
        if total:
            mix = ", ".join(f"{counts[k]} {k}" for k in sorted(counts))
            lines.append(f"Messages: {total} ({mix}) from {', '.join(senders)}.")
        else:
            lines.append("Messages: none.")
        if alerts:
            lines.append(
                f"Urgent alerts: {len(alerts)}, {len(open_alerts)} still "
                "unacknowledged" + (": " + "; ".join(
                    f"{a['sender']}: {a['message'].rstrip('.')}" for a in open_alerts
                ) if open_alerts else "") + "."
            )
        else:
            lines.append("Urgent alerts: none.")
        missed = [f"{r['med']} at {r['time']}" for r in data["medications"]
                  if not r["taken"]]
        taken = sum(1 for r in data["medications"] if r["taken"])
        lines.append(
            f"Medications recorded: {taken} taken, {len(missed)} missed"
            + (" (" + "; ".join(missed) + ")" if missed else "") + "."
        )
        appointments = [f"{r['title']}: {r['datetime']}" for r in data["appointments"]]
        lines.append("Appointments: " + ("; ".join(appointments) if appointments
                                         else "none recorded") + ".")
        tasks = [f"{r['sibling']}: {r['task']}" for r in data["sibling_tasks"]]
        lines.append("Sibling tasks: " + ("; ".join(tasks) if tasks
                                          else "none recorded") + ".")
        return {
            "summary": " ".join(lines),
            "hours": hours,
            "message_counts": counts,
            "senders": senders,
            "alerts_total": len(alerts),
            "alerts_open": len(open_alerts),
            **data,
        }

    def get_parent_summary(self) -> dict[str, Any]:
        data = self.snapshot()
        parent = data["parent"]
        parts = [(f"Here's the recorded update for {parent['nickname']} "
                  f"({parent['name']}, {parent['age']}).")]
        medications = [
            f"{r['med']} at {r['time']} ({'taken' if r['taken'] else 'missed'})"
            for r in data["medications"]
        ]
        appointments = [
            f"{r['title']}: {r['datetime']}" for r in data["appointments"]
        ]
        tasks = [f"{r['sibling']}: {r['task']}" for r in data["sibling_tasks"]]
        for label, entries in (
            ("Medications", medications), ("Appointments", appointments),
            ("Sibling tasks", tasks),
        ):
            parts.append(f"{label}: {'; '.join(entries) if entries else 'none recorded'}.")
        return {"summary": " ".join(parts), **data}
