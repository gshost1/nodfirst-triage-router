"""Local FastAPI surface: the static UI, `POST /chat`, `POST /inbound`,
`GET /stats`, `GET /thread`, the urgent `/alerts` endpoints, and `/demo`.

Bind this application to loopback only. It coordinates the classifier,
routing, and caregiver agent through `core.chat.ChatService` and never talks to
a model provider directly.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from core.chat import ChatService
from core.demo_data import SUGGESTED_QUESTIONS, is_seeded, seed_demo
from core.inbound import (
    InboundError,
    contacts_from_env,
    parse_inbound,
    token_is_valid,
    twiml_reply,
)
from core.llm import create_llm_client
from core.metrics_store import MetricsStore
from core.notify import notifier_from_env
from core.speech import SpeechError, speech_from_env
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
    voice: bool = False


class SpeakRequest(BaseModel):
    """Text to read aloud, normally a reply the page just received."""

    text: str = Field(min_length=1, max_length=2000)


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
        notifier=notifier_from_env(),
        speech=speech_from_env(),
    )


def create_app(
    *,
    service: ChatService | None = None,
    service_factory: Callable[[], ChatService] | None = None,
    inbound_contacts: dict[str, str] | None = None,
    inbound_secret: str | None = None,
) -> FastAPI:
    """Create the app; tests inject a `service` built from doubles.

    `inbound_contacts` maps phone numbers or email addresses to sibling names
    and `inbound_secret` gates `POST /inbound`; both default to the
    `INBOUND_CONTACTS` and `INBOUND_SECRET` environment variables.
    """
    api = FastAPI(title="Nod First Triage Router", version="0.1.0")
    cache: dict[str, ChatService] = {}
    contacts = inbound_contacts if inbound_contacts is not None else contacts_from_env()
    secret = inbound_secret if inbound_secret is not None else os.getenv(
        "INBOUND_SECRET", ""
    ).strip()

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
        return get_service().handle(
            message, sender=request.sender, voice=request.voice
        ).as_dict()

    @api.post("/speak")
    def speak(request: SpeakRequest) -> Response:
        """Any text as WAV from text-to-speech; 503 when speech is not configured."""
        try:
            clip = get_service().speak(request.text)
        except SpeechError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from error
        return Response(
            clip.audio, media_type=clip.media_type,
            headers={"Cache-Control": "no-store", "X-Speech-Model": clip.model},
        )

    @api.post("/inbound")
    async def inbound(
        request: Request,
        x_inbound_token: str | None = Header(default=None),
    ) -> Response:
        """Accept a text or email from a gateway and triage it like a typed turn.

        JSON bodies (`from`, `text`) get the chat turn as JSON. Form bodies in
        the Twilio shape (`From`, `Body`) get a TwiML reply so the gateway can
        text the answer back. When `INBOUND_SECRET` is set, the request must
        carry it in `X-Inbound-Token`.
        """
        if not token_is_valid(x_inbound_token, secret):
            raise HTTPException(status_code=401, detail="invalid inbound token")
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise HTTPException(status_code=422, detail="JSON body must be an object")
            channel = "json"
        else:
            # Parse urlencoded forms directly: no python-multipart dependency.
            raw = (await request.body()).decode("utf-8", errors="replace")
            payload = {key: values[-1] for key, values in parse_qs(raw).items()}
            channel = "form"
        try:
            message = parse_inbound(payload, channel=channel, contacts=contacts)
        except InboundError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        turn = get_service().handle(message.text, sender=message.sender).as_dict()
        turn["inbound"] = {
            "channel": channel, "origin": message.origin, "sender": message.sender,
        }
        if channel == "form":
            return Response(twiml_reply(turn["reply"]), media_type="application/xml")
        return JSONResponse(turn)

    @api.get("/thread")
    def thread() -> dict[str, Any]:
        """The shared sibling thread and roster, so a reload keeps the history."""
        return get_service().thread()

    @api.get("/demo")
    def demo_state() -> dict[str, Any]:
        """Suggested questions and whether the demo week is loaded."""
        return {
            "seeded": is_seeded(get_service().store),
            "suggestions": list(SUGGESTED_QUESTIONS),
        }

    @api.post("/demo/seed")
    def demo_seed() -> dict[str, Any]:
        """Load the demo week into the store (no model call); idempotent."""
        return seed_demo(get_service().store)

    @api.get("/digest")
    def digest() -> dict[str, Any]:
        """The store-grounded daily digest; no model call, no cost."""
        return get_service().digest()

    @api.post("/digest/speak")
    def speak_digest() -> Response:
        """The digest as WAV audio from DigitalOcean text-to-speech.

        503 in MOCK mode or without a key: the page then uses the browser's
        own speech synthesis and says so.
        """
        try:
            clip = get_service().speak_digest()
        except SpeechError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from error
        return Response(
            clip.audio, media_type=clip.media_type,
            headers={"Cache-Control": "no-store", "X-Speech-Model": clip.model},
        )

    @api.post("/digest/send")
    def send_digest() -> dict[str, Any]:
        """Email the digest to ALERT_EMAIL_TO through Botmail, if configured."""
        return get_service().email_digest()

    @api.get("/alerts")
    def alerts() -> dict[str, Any]:
        """Urgent alerts recorded for the sibling group, newest first."""
        service = get_service()
        return {"alerts": service.alerts(), "delivery": service.alert_delivery()}

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
