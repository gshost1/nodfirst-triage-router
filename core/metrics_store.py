"""SQLite persistence for per-inference-call metrics.

Records live beside the demo data in the same file-backed SQLite store so the
stats stay consistent with the caregiving data that produced them. Tests pass
an isolated temporary file; connections are short lived like `CareStore`.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from decimal import Decimal, InvalidOperation
from pathlib import Path

from core.metrics import USAGE_UNAVAILABLE, InferenceCall, StatsReport, aggregate_calls
from core.pricing import PricingTable

SCHEMA = """
CREATE TABLE IF NOT EXISTS inference_calls (
    id INTEGER PRIMARY KEY,
    tier TEXT NOT NULL,
    route TEXT NOT NULL,
    model TEXT,
    response_model TEXT,
    latency_ms REAL NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    usage_source TEXT NOT NULL,
    estimated_cost_usd TEXT
);
"""

_COLUMNS = (
    "tier, route, model, response_model, latency_ms, prompt_tokens, "
    "completion_tokens, total_tokens, usage_source, estimated_cost_usd"
)


def _as_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


class MetricsStore:
    """Append-only record of every inference invocation."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with closing(self._connect()) as db, db:
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def record(self, call: InferenceCall) -> None:
        db = self._connect()
        try:
            with db:
                db.execute(
                    f"INSERT INTO inference_calls ({_COLUMNS}) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        call.tier,
                        call.route,
                        call.model,
                        call.response_model,
                        float(call.latency_ms),
                        call.prompt_tokens,
                        call.completion_tokens,
                        call.total_tokens,
                        call.usage_source,
                        None
                        if call.estimated_cost_usd is None
                        else str(call.estimated_cost_usd),
                    ),
                )
        finally:
            db.close()

    def calls(self) -> list[InferenceCall]:
        db = self._connect()
        try:
            rows = db.execute(
                f"SELECT {_COLUMNS} FROM inference_calls ORDER BY id"
            ).fetchall()
        finally:
            db.close()
        return [
            InferenceCall(
                tier=row["tier"],
                route=row["route"],
                model=row["model"],
                response_model=row["response_model"],
                latency_ms=float(row["latency_ms"]),
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                total_tokens=row["total_tokens"],
                usage_source=row["usage_source"] or USAGE_UNAVAILABLE,
                estimated_cost_usd=_as_decimal(row["estimated_cost_usd"]),
            )
            for row in rows
        ]

    def clear(self) -> None:
        db = self._connect()
        try:
            with db:
                db.execute("DELETE FROM inference_calls")
        finally:
            db.close()

    def stats(
        self, pricing: PricingTable | None = None, large_model: str | None = None
    ) -> StatsReport:
        return aggregate_calls(self.calls(), pricing, large_model)
