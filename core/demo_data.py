"""Demo data for Mom's care record: a realistic week, no model calls.

`seed_demo(store)` fills medications, appointments, sibling tasks, a short
sibling thread, and one acknowledged alert so the digest, the summary tools,
and the suggested questions have something to talk about. It is idempotent:
a second call is a no-op and reports `seeded: False`. `SUGGESTED_QUESTIONS`
feeds the chips above the composer.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from core.store import CareStore

MEDICATIONS: tuple[tuple[str, str, bool], ...] = (
    ("metformin", "yesterday 8:00 am", True),
    ("metformin", "yesterday 6:00 pm", True),
    ("lisinopril", "yesterday 8:00 am", True),
    ("vitamin D", "today 9:00 am", True),
    ("metformin", "today 8:00 am", False),
    ("lisinopril", "today 8:00 am", True),
)

APPOINTMENTS: tuple[tuple[str, str], ...] = (
    ("Cardiology follow-up with Dr. Reyes", "Thursday 10:30 am"),
    ("Pharmacy refill delivery (metformin, lisinopril)", "Friday afternoon"),
    ("Physiotherapy, balance exercises", "next Tuesday 2:00 pm"),
    ("Eye exam, new reading glasses", "October 24, 11:00 am"),
)

SIBLING_TASKS: tuple[tuple[str, str], ...] = (
    ("Pavel", "Drive Mom to cardiology on Thursday"),
    ("Anna", "Call the pharmacy about the Friday refill"),
    ("Anna", "Visit Saturday and check the bathroom grab bar"),
    ("Pavel", "Replace the kitchen night light"),
)

URGENT_REPLY = (
    "If this could be an emergency, call 911 now. Otherwise call Mom's doctor "
    "right away about the dizziness; please do not wait for a reply from me."
)

# (sender, role, content, class)
THREAD: tuple[tuple[str, str, str, str], ...] = (
    ("Anna", "user", "Mom had her metformin and lisinopril at 8 this morning, both taken.", "routine"),
    ("assistant", "assistant", "Recorded: metformin and lisinopril taken at 8 am. Thanks, Anna.", "routine"),
    ("Pavel", "user", "Book Mom's cardiology follow-up for Thursday, I will drive.", "logistics"),
    ("assistant", "assistant",
     "Booked the cardiology follow-up for Thursday 10:30 am and noted Pavel as the driver.",
     "logistics"),
    ("Anna", "user", "Can you schedule the pharmacy refill delivery for Friday?", "logistics"),
    ("assistant", "assistant",
     "Added the pharmacy refill delivery for Friday afternoon. I also noted you will call them.",
     "logistics"),
    ("Pavel", "user", "Mom felt dizzy standing up this morning and sat back down.", "urgent"),
    ("assistant", "assistant", URGENT_REPLY, "urgent"),
    ("Pavel", "user", "Called Dr. Reyes, nurse says watch her today and mention it Thursday.", "routine"),
    ("assistant", "assistant",
     "Noted: the nurse advised watching Mom today and raising the dizziness at Thursday's visit.",
     "routine"),
    ("Anna", "user", "Mom seemed cheerful on the phone tonight, she finished her puzzle.", "routine"),
    ("assistant", "assistant", "Lovely to hear. Nothing to record, just a good evening for Mom.", "routine"),
)

ALERT = (
    "Pavel",
    "Mom felt dizzy standing up this morning and sat back down.",
    "Pavel reported dizziness; nurse advised watching Mom today.",
)

SUGGESTED_QUESTIONS: tuple[dict[str, str], ...] = (
    {"text": "Who is driving Mom today?", "kind": "logistics"},
    {"text": "Did Mom take her metformin today?", "kind": "routine"},
    {"text": "What appointments does Mom have this week?", "kind": "logistics"},
    {"text": "Give me today's digest.", "kind": "routine"},
    {"text": "Book Mom's physio for next Tuesday, Anna drives.", "kind": "logistics"},
    {"text": "Mom had a fall in the kitchen this morning.", "kind": "urgent"},
)


def _stamp(hours_ago: float) -> str:
    moment = datetime.now(UTC) - timedelta(hours=hours_ago)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def is_seeded(store: CareStore) -> bool:
    """True once the first demo appointment is on file."""
    db = store._connect()
    try:
        row = db.execute(
            "SELECT 1 FROM appointments WHERE title = ? LIMIT 1", (APPOINTMENTS[0][0],)
        ).fetchone()
        return row is not None
    finally:
        db.close()


def seed_demo(store: CareStore) -> dict[str, Any]:
    """Insert the demo week once; return what was added."""
    if is_seeded(store):
        return {"seeded": False, "reason": "demo data already loaded"}
    for med, time, taken in MEDICATIONS:
        store.log_med(med, time, taken)
    for title, when in APPOINTMENTS:
        store.add_appointment(title, when)
    for sibling, task in SIBLING_TASKS:
        store.assign_sibling_task(sibling, task)
    # Spread the thread over the last day so the digest window picks it up.
    hours = 20.0
    for sender, role, content, message_class in THREAD:
        row = store.add_message(sender, role, content, message_class)
        _backdate("messages", store, row["id"], _stamp(hours))
        hours -= 1.5
    alert = store.add_alert(*ALERT)
    _backdate("alerts", store, alert["id"], _stamp(12.0))
    store.acknowledge_alert(alert["id"])
    return {
        "seeded": True,
        "medications": len(MEDICATIONS),
        "appointments": len(APPOINTMENTS),
        "tasks": len(SIBLING_TASKS),
        "messages": len(THREAD),
        "alerts": 1,
    }


def _backdate(table: str, store: CareStore, identifier: int, stamp: str) -> None:
    db = store._connect()
    try:
        with db:
            db.execute(
                f"UPDATE {table} SET created_at = ? WHERE id = ?", (stamp, identifier)
            )
    finally:
        db.close()
