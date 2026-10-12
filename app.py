"""Local FastAPI surface: the static UI, `POST /chat`, `GET /stats`,
`GET /thread`, and the urgent `/alerts` endpoints.

Bind this application to loopback only. It coordinates the classifier,
routing, and caregiver agent through `core.chat.ChatService` and never talks to
a model provider directly.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from core.chat import ChatService
from core.llm import create_llm_client
from core.metrics_store import MetricsStore
from core.store import CareStore
from router.config import RouterConfig

REPO_ROOT = Path(__file__).resolve().parent
STATIC_INDEX = REPO_ROOT / "static" / "index.html"
DEFAULT_DB_PATH = "nodfirst.sqlite3"

PLACEHOLDER_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Nod First Triage Router</title>
  </head>
  <body>
    <h1>Nod First Triage Router</h1>
    <p>
      The chat page is not built yet. The local API is available at
      <code>POST /chat</code> and <code>GET /stats</code>.
    </p>
    <p>Not medical advice.</p>
  </body>
</html>
"""


class ChatRequest(BaseModel):
    """A single caregiver message from the local UI."""

    message: str = Field(min_length=1)
    sender: str | None = Field(default=None, max_length=40)


def build_service() -> ChatService:
    """Build the chat service from environment configuration.

    Called lazily on the first request so importing this module never loads
    credentials or constructs a provider client.
    """
    db_path = os.getenv("CARE_DB", "").strip() or DEFAULT_DB_PATH
    return ChatService(
        client=create_llm_client(),
        config=RouterConfig.from_env(),
        store=CareStore(db_path),
        metrics=MetricsStore(db_path),
    )


def create_app(
    *,
    service: ChatService | None = None,
    service_factory: Callable[[], ChatService] | None = None,
) -> FastAPI:
    """Create the app; tests inject a `service` built from doubles."""
    api = FastAPI(title="Nod First Triage Router", version="0.1.0")
    cache: dict[str, ChatService] = {}

    def get_service() -> ChatService:
        if service is not None:
            return service
        if "service" not in cache:
            cache["service"] = (service_factory or build_service)()
        return cache["service"]

    @api.get("/", response_class=HTMLResponse)
    def index() -> Any:
        """Serve the static chat UI when it exists, else an honest placeholder."""
        headers = {"Cache-Control": "no-store"}
        if STATIC_INDEX.is_file():
            return FileResponse(STATIC_INDEX, media_type="text/html", headers=headers)
        return HTMLResponse(PLACEHOLDER_HTML, headers=headers)

    @api.post("/chat")
    def chat(request: ChatRequest) -> dict[str, Any]:
        """Classify, route, run the caregiver agent, and report the metrics."""
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail="message must not be blank")
        return get_service().handle(message, sender=request.sender).as_dict()

    @api.get("/thread")
    def thread() -> dict[str, Any]:
        """The shared sibling thread and roster, so a reload keeps the history."""
        return get_service().thread()

    @api.get("/alerts")
    def alerts() -> dict[str, Any]:
        """Urgent alerts recorded for the sibling group, newest first."""
        return {"alerts": get_service().alerts()}

    @api.post("/alerts/{alert_id}/ack")
    def acknowledge_alert(alert_id: int) -> dict[str, Any]:
        """Mark an alert as seen by a sibling."""
        alert = get_service().acknowledge_alert(alert_id)
        if alert is None:
            raise HTTPException(status_code=404, detail="alert not found")
        return alert

    @api.get("/stats")
    def stats() -> dict[str, Any]:
        """Report per-inference-call counts, latency, estimated spend and savings."""
        return get_service().stats().as_dict()

    return api


app = create_app()
