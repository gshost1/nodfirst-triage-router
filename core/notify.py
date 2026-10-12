"""Outbound urgent-alert email through Botmail (https://botmail.pro).

Delivery is best effort and never blocks or fails a chat turn: every outcome
is returned as a `Notification` and stored on the alert. The API key comes
from `BOTMAIL_API_KEY` or a file named by `BOTMAIL_KEY_FILE`; it is never
logged, stored, or echoed in an error message.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BOTMAIL_BASE_URL = "https://botmail.pro/v1"
SEND_TIMEOUT_SECONDS = 10.0
STATUS_SENT = "sent"
STATUS_FAILED = "failed"
STATUS_DISABLED = "disabled"

# Returns the parsed JSON body; raises on transport or HTTP errors.
Transport = Callable[[str, dict[str, str], bytes], dict[str, Any]]


@dataclass(frozen=True)
class Notification:
    status: str
    recipients: list[str]
    detail: str
    message_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "recipients": list(self.recipients),
            "detail": self.detail,
            "message_id": self.message_id,
        }


class NullNotifier:
    """No outbound mail: the default when no recipients or key are configured."""

    def __init__(self, reason: str = "outbound email not configured") -> None:
        self.reason = reason

    def send_alert(self, alert: dict[str, Any]) -> Notification:
        return Notification(STATUS_DISABLED, [], self.reason)

    def send_digest(self, digest: dict[str, Any]) -> Notification:
        return Notification(STATUS_DISABLED, [], self.reason)


class BotmailNotifier:
    """Email each urgent alert to the sibling group through Botmail."""

    def __init__(
        self,
        *,
        api_key: str,
        recipients: list[str],
        base_url: str = BOTMAIL_BASE_URL,
        transport: Transport | None = None,
        app_url: str = "http://127.0.0.1:8000",
    ) -> None:
        if not api_key or not recipients:
            raise ValueError("BotmailNotifier needs an API key and recipients")
        self._api_key = api_key
        self.recipients = recipients
        self.base_url = base_url.rstrip("/")
        self.transport = transport or _urllib_transport
        self.app_url = app_url

    def send_alert(self, alert: dict[str, Any]) -> Notification:
        return self._send(
            subject=f"Urgent about Mom from {alert['sender']}",
            text=alert_email_text(alert, self.app_url),
            idempotency_key=f"nodfirst-alert-{alert['id']}",
        )

    def send_digest(self, digest: dict[str, Any]) -> Notification:
        return self._send(
            subject="Daily digest about Mom",
            text=digest["summary"] + "\n\nNot medical advice. "
            f"Open the shared thread: {self.app_url}\n",
            idempotency_key=None,
        )

    def _send(
        self, *, subject: str, text: str, idempotency_key: str | None
    ) -> Notification:
        payload = {
            "to": ", ".join(self.recipients) if len(self.recipients) > 1
            else self.recipients[0],
            "subject": subject,
            "text": text,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        try:
            body = self.transport(
                f"{self.base_url}/send", headers, json.dumps(payload).encode()
            )
        except urllib.error.HTTPError as error:
            return Notification(
                STATUS_FAILED, self.recipients, f"Botmail returned HTTP {error.code}"
            )
        except Exception as error:  # noqa: BLE001 - delivery must never raise
            return Notification(
                STATUS_FAILED, self.recipients,
                f"Botmail request failed: {type(error).__name__}",
            )
        message_id = body.get("id") or body.get("message_id")
        return Notification(
            STATUS_SENT, self.recipients, "accepted by Botmail (202)",
            str(message_id) if message_id else None,
        )


def alert_email_text(alert: dict[str, Any], app_url: str) -> str:
    return (
        f"{alert['sender']} wrote about Mom:\n\n"
        f"    {alert['message']}\n\n"
        f"{alert['note']}\n\n"
        "This message was routed as urgent-health by the Nod First triage "
        "router. Not medical advice: if this could be an emergency, call 911 now.\n\n"
        f"Acknowledge it in the shared thread: {app_url}\n"
    )


def _urllib_transport(url: str, headers: dict[str, str], body: bytes) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=SEND_TIMEOUT_SECONDS) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def _read_api_key() -> str:
    key = os.getenv("BOTMAIL_API_KEY", "").strip()
    if key:
        return key
    key_file = os.getenv("BOTMAIL_KEY_FILE", "").strip()
    if key_file:
        path = Path(key_file).expanduser()
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return ""


def notifier_from_env(**overrides: Any) -> NullNotifier | BotmailNotifier:
    """Build the notifier from `ALERT_EMAIL_TO` plus a Botmail key.

    With no recipients, or recipients but no key, return a `NullNotifier`
    whose reason says which is missing, so the UI can show it honestly.
    """
    recipients = [
        address.strip()
        for address in os.getenv("ALERT_EMAIL_TO", "").split(",")
        if address.strip()
    ]
    if not recipients:
        return NullNotifier("ALERT_EMAIL_TO is not set; alerts stay local")
    api_key = _read_api_key()
    if not api_key:
        return NullNotifier(
            "ALERT_EMAIL_TO is set but no Botmail key was found "
            "(BOTMAIL_API_KEY or BOTMAIL_KEY_FILE)"
        )
    port = os.getenv("PORT", "8000").strip() or "8000"
    return BotmailNotifier(
        api_key=api_key, recipients=recipients,
        app_url=f"http://127.0.0.1:{port}", **overrides,
    )
