"""Five-message MOCK rehearsal for the Nod First Triage Router.

`scripts/demo.py` replays the approved five-message script (one routine, two
logistics, two urgent) through the same `core.chat.ChatService` the local API
uses and prints a results table plus the replies.

The demo is offline by design: it always runs `ROUTER_MODE=mock`, builds the
canned client directly, never reads `.env`, and never issues a provider
request. Export `MODEL_SMALL`, `MODEL_MID`, and `MODEL_LARGE` to show the
configured tier IDs in the table; otherwise MOCK routing labels each call
`mock`.

Run it with:

    uv run python scripts/demo.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.chat import ChatService

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = "nodfirst.sqlite3"
MESSAGE_WIDTH = 44
TIER_VARIABLES = ("MODEL_SMALL", "MODEL_MID", "MODEL_LARGE")

# The approved demo script, shared with the UI's Replay control
# (`static/index.html`, `<script id="demo-script">`).
DEMO_MESSAGES: tuple[tuple[str, str], ...] = (
    ("routine", "How did Mom's visit with her neighbor go yesterday?"),
    ("logistics", "Book Mom's cardiology for Thursday, Pavel drives"),
    ("logistics", "Can you schedule Mom's pharmacy refill delivery for Friday?"),
    ("urgent", "Mom had a fall in the kitchen this morning."),
    ("urgent", "Mom says she has chest pain and I can't reach her doctor."),
)


@dataclass(frozen=True)
class DemoRow:
    """One scripted message and the routing the chat service produced for it."""

    number: int
    expected: str
    message_class: str
    route: str
    model: str
    latency_ms: float
    estimated_cost_usd: float | None
    tools: tuple[str, ...]
    message: str
    reply: str

    @property
    def matches_script(self) -> bool:
        """Whether the runtime classification matches the scripted expectation."""
        return self.message_class == self.expected


def ensure_repo_root_on_path() -> None:
    """Make `core`/`router` importable when this file is run as a script."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def build_mock_service(db_path: str) -> ChatService:
    """Build the offline MOCK service used by the CLI and its tests.

    The canned client is constructed directly, so no `.env` value, credential,
    or provider endpoint can change this path.
    """
    ensure_repo_root_on_path()
    from core.chat import ChatService
    from core.llm import MOCK_BASE_URL, MOCK_MODEL, LLMClient
    from core.metrics_store import MetricsStore
    from core.store import CareStore
    from router.config import RouterConfig

    return ChatService(
        client=LLMClient(base_url=MOCK_BASE_URL, api_key="", model=MOCK_MODEL),
        config=RouterConfig.from_env(
            {
                "ROUTER_MODE": "mock",
                "MODEL_SMALL": os.getenv("MODEL_SMALL", "").strip(),
                "MODEL_MID": os.getenv("MODEL_MID", "").strip(),
                "MODEL_LARGE": os.getenv("MODEL_LARGE", "").strip(),
            }
        ),
        store=CareStore(db_path),
        metrics=MetricsStore(db_path),
    )


def run_demo(
    service: ChatService, messages: Sequence[tuple[str, str]] = DEMO_MESSAGES
) -> list[DemoRow]:
    """Replay the scripted messages through an already-built chat service."""
    rows: list[DemoRow] = []
    for number, (expected, message) in enumerate(messages, start=1):
        turn = service.handle(message)
        rows.append(
            DemoRow(
                number=number,
                expected=expected,
                message_class=turn.decision.message_class.value,
                route=turn.decision.route,
                model=turn.model or turn.decision.model or "unreported",
                latency_ms=float(turn.latency_ms),
                estimated_cost_usd=(
                    None
                    if turn.estimated_cost_usd is None
                    else float(turn.estimated_cost_usd)
                ),
                tools=tuple(action.name for action in turn.tool_actions),
                message=message,
                reply=turn.reply,
            )
        )
    return rows


def tier_status() -> str:
    """Describe the configured tier IDs without reading `.env`."""
    values = {name: os.getenv(name, "").strip() for name in TIER_VARIABLES}
    if not any(values.values()):
        return "MODEL_SMALL/MODEL_MID/MODEL_LARGE unset (MOCK labels calls 'mock')"
    return " ".join(f"{name}={value or 'unset'}" for name, value in values.items())


def banner(db_path: str) -> list[str]:
    """The header lines printed above the demo table."""
    lines = [
        "Nod First Triage Router - five-message demo",
        "Mode:  MOCK (offline rehearsal; canned client, no provider calls)",
        f"Store: {db_path}",
        f"Tiers: {tier_status()}",
    ]
    requested = os.getenv("ROUTER_MODE", "").strip().lower()
    if requested and requested != "mock":
        lines.append(
            f"Note:  ROUTER_MODE={requested} is ignored here; this demo always "
            "rehearses offline in MOCK mode."
        )
    return lines


def format_table(rows: Sequence[DemoRow]) -> str:
    """Render the demo rows as a fixed-width plain-text table."""
    headers = (
        "#",
        "Class",
        "Route",
        "Model",
        "Latency",
        "Est. cost",
        "Tools",
        "Message",
    )
    body = [
        (
            str(row.number),
            row.message_class,
            row.route,
            row.model,
            f"{row.latency_ms:.3f} ms",
            _money(row.estimated_cost_usd),
            ", ".join(row.tools) or "-",
            _truncate(row.message, MESSAGE_WIDTH),
        )
        for row in rows
    ]
    widths = [
        max([len(headers[index])] + [len(line[index]) for line in body])
        for index in range(len(headers))
    ]
    lines = ["  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))).rstrip()]
    lines.append("  ".join("-" * width for width in widths))
    for line in body:
        lines.append("  ".join(line[i].ljust(widths[i]) for i in range(len(headers))).rstrip())
    return "\n".join(lines)


def format_summary(rows: Sequence[DemoRow]) -> str:
    """The class-count summary line, with a warning when the mix is unexpected."""
    counts = Counter(row.message_class for row in rows)
    mix = ", ".join(
        f"{name} {counts.get(name, 0)}" for name in ("routine", "logistics", "urgent")
    )
    line = f"Counts: {mix} ({len(rows)} messages)"
    mismatches = [row for row in rows if not row.matches_script]
    if not mismatches:
        return line
    detail = "; ".join(
        f"#{row.number} expected {row.expected}, got {row.message_class}"
        for row in mismatches
    )
    return f"{line}\nWARNING: scripted expectation mismatch: {detail}"


def format_replies(rows: Sequence[DemoRow]) -> str:
    """List each scripted message with its canned reply and executed tools."""
    lines = ["Messages and replies:"]
    for row in rows:
        lines.append(f"[{row.number}] {row.message_class}: {_collapse(row.message)}")
        lines.append(f"    reply: {_collapse(row.reply)}")
        lines.append(f"    tools: {', '.join(row.tools) or 'none'}")
    return "\n".join(lines)


def _money(value: float | None) -> str:
    """Render an estimated cost, keeping unavailable values honest."""
    if value is None:
        return "n/a"
    if value == 0:
        return "$0.00"
    if value < 0.0000005:
        return "<$0.000001"
    return f"${value:.6f}".rstrip("0").rstrip(".")


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, width: int) -> str:
    collapsed = _collapse(text)
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: width - 3] + "..."


def main(argv: Sequence[str] | None = None) -> int:
    """Print the five-message MOCK demo; exit non-zero on a script mismatch."""
    parser = argparse.ArgumentParser(
        description="Replay the approved five-message script offline in MOCK mode."
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"SQLite store path (default: $CARE_DB or {DEFAULT_DB})",
    )
    args = parser.parse_args(argv)
    db_path = args.db or os.getenv("CARE_DB", "").strip() or DEFAULT_DB

    for line in banner(db_path):
        print(line)
    print()
    rows = run_demo(build_mock_service(db_path))
    print(format_table(rows))
    print()
    print(format_summary(rows))
    print()
    print(
        "Note: est. cost uses the published DigitalOcean rates in core/pricing.py "
        "(source/date in MODEL_SELECTION.md); MOCK token counts are estimated."
    )
    print()
    print(format_replies(rows))
    return 0 if all(row.matches_script for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
