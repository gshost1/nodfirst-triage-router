"""Demo data for Mom's care record: a realistic week, no model calls.

`seed_demo(store)` fills medications, appointments, sibling tasks, a short
sibling thread, and one open urgent alert so the digest, the summary tools,
and the summary questions have something to talk about. It is idempotent:
a second call is a no-op and reports `seeded: False`.
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
    ("metformin", "today 8:00 am (refused, stomach upset)", False),
    ("lisinopril", "today 8:00 am", True),
)

APPOINTMENTS: tuple[tuple[str, str], ...] = (
    ("Cardiology follow-up with Dr. Reyes", "Thursday 10:30 am"),
    ("Pharmacy refill delivery (metformin, lisinopril)", "Friday afternoon"),
    ("Physiotherapy, balance exercises", "next Tuesday 2:00 pm"),
    ("Eye exam, new reading glasses", "October 24, 11:00 am"),
)

# All open at the start of the demo: the family assigns them in the chat.
SIBLING_TASKS: tuple[tuple[str, str], ...] = (
    ("unassigned", "Drive Mom to cardiology on Thursday at 10:30 am"),
    ("unassigned", "Call the pharmacy to confirm Friday's refill delivery"),
    ("unassigned", "Take Mom to physiotherapy next Tuesday at 2 pm"),
    ("unassigned", "Drive Mom to the eye exam on October 24"),
    ("unassigned", "Weekend visit and check the bathroom grab bar"),
    ("unassigned", "Replace the kitchen night light"),
    ("unassigned", "Groceries and Sunday meal prep"),
    ("unassigned", "Sort the weekly pill organizer on Sunday evening"),
    ("unassigned", "Call Mom every evening this week"),
)

URGENT_REPLY = (
    "If this could be an emergency, call 911 now. Otherwise call Mom's doctor "
    "right away about the dizziness; please do not wait for a reply from me."
)

# (sender, role, content, class): Mom's day so far, as the siblings reported it.
THREAD: tuple[tuple[str, str, str, str], ...] = (
    ("Anna", "user", "Morning check: Mom slept well, had oatmeal and tea at 7:30.", "routine"),
    ("assistant", "assistant", "Good start to the day. Noted the breakfast, nothing to record.", "routine"),
    ("Anna", "user", "She took her lisinopril at 8 but refused the metformin, said her stomach was off.", "routine"),
    ("assistant", "assistant",
     "Recorded: lisinopril taken at 8 am, metformin at 8 am marked missed. Worth a gentle retry with lunch.",
     "routine"),
    ("Pavel", "user", "Mom felt dizzy standing up this morning and sat back down.", "urgent"),
    ("assistant", "assistant", URGENT_REPLY, "urgent"),
    ("Pavel", "user", "Called Dr. Reyes, nurse says watch her today and mention it Thursday.", "routine"),
    ("assistant", "assistant",
     "Noted: the nurse advised watching Mom today and raising the dizziness at Thursday's visit.",
     "routine"),
    ("Anna", "user", "Vitamin D taken at 9. She did her balance exercises with the neighbor, Rita, for 15 minutes.", "routine"),
    ("assistant", "assistant", "Recorded vitamin D at 9 am. Nice that Rita came by for the exercises.", "routine"),
    ("Pavel", "user", "Book Mom's cardiology follow-up for Thursday, we still need a driver.", "logistics"),
    ("assistant", "assistant",
     "Booked the cardiology follow-up for Thursday 10:30 am. Driving Mom there is an open task.",
     "logistics"),
    ("Anna", "user", "Can you schedule the pharmacy refill delivery for Friday?", "logistics"),
    ("assistant", "assistant",
     "Added the pharmacy refill delivery for Friday afternoon. Confirming it with the pharmacy is an open task.",
     "logistics"),
    ("Anna", "user", "Lunch was soup and bread, she ate most of it and took a nap after. Mood is cheerful.", "routine"),
    ("assistant", "assistant", "Lovely. Lunch, nap and a cheerful mood noted; the metformin is still the one to retry.", "routine"),
)

ALERT = (
    "Pavel",
    "Mom felt dizzy standing up this morning and sat back down.",
    "Pavel reported dizziness; nurse advised watching Mom today.",
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
    hours = 9.0
    for sender, role, content, message_class in THREAD:
        row = store.add_message(sender, role, content, message_class)
        _backdate("messages", store, row["id"], _stamp(hours))
        hours -= 0.55
    alert = store.add_alert(*ALERT)
    _backdate("alerts", store, alert["id"], _stamp(7.0))
    # Left open on purpose so the red card and "Acknowledge" are part of the demo.
    store.record_alert_email(alert["id"], "skipped", "seeded demo data, no email sent", [])
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
