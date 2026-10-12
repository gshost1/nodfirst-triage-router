"""Inbound messages from SMS or email gateways: parsing and sender mapping.

`POST /inbound` accepts either JSON (`{"from": ..., "text": ...}`) or a
Twilio-style form (`From`, `Body`). Nothing here trusts the content; it is a
caregiver message like any typed one and goes through the same classifier.
"""

from __future__ import annotations

import hmac
import os
import re
from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape

MAX_MESSAGE_LENGTH = 2000
MAX_SENDER_LENGTH = 40
UNKNOWN_SENDER = "Unknown sender"


@dataclass(frozen=True)
class InboundMessage:
    origin: str          # the raw From value: a phone number or email address
    sender: str          # the sibling name shown in the thread
    text: str
    channel: str         # "json" or "form"


class InboundError(ValueError):
    """The request did not carry a usable message."""


def parse_contacts(raw: str) -> dict[str, str]:
    """`INBOUND_CONTACTS="+15551234567=Anna,pavel@example.com=Pavel"`."""
    contacts: dict[str, str] = {}
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        origin, name = pair.split("=", 1)
        if origin.strip() and name.strip():
            contacts[_normalize_origin(origin)] = name.strip()[:MAX_SENDER_LENGTH]
    return contacts


def contacts_from_env() -> dict[str, str]:
    return parse_contacts(os.getenv("INBOUND_CONTACTS", ""))


def _normalize_origin(origin: str) -> str:
    origin = origin.strip().lower()
    # "Anna <anna@example.com>" -> anna@example.com
    match = re.search(r"<([^>]+)>", origin)
    if match:
        origin = match.group(1).strip()
    if "@" not in origin:
        origin = re.sub(r"[^\d+]", "", origin)  # phone: keep digits and +
    return origin


def resolve_sender(origin: str, contacts: dict[str, str]) -> str:
    key = _normalize_origin(origin)
    if key in contacts:
        return contacts[key]
    if not key:
        return UNKNOWN_SENDER
    # Unknown number or address: show it, truncated, so nobody is impersonated.
    return key[:MAX_SENDER_LENGTH]


def parse_inbound(
    payload: dict[str, Any], *, channel: str, contacts: dict[str, str]
) -> InboundMessage:
    origin = str(
        payload.get("from") or payload.get("From") or payload.get("sender") or ""
    ).strip()
    text = str(
        payload.get("text") or payload.get("Body") or payload.get("body")
        or payload.get("message") or ""
    ).strip()
    if not text:
        raise InboundError("message text must not be blank")
    if len(text) > MAX_MESSAGE_LENGTH:
        raise InboundError(f"message text longer than {MAX_MESSAGE_LENGTH} characters")
    return InboundMessage(
        origin=origin, sender=resolve_sender(origin, contacts), text=text,
        channel=channel,
    )


def token_is_valid(presented: str | None, expected: str) -> bool:
    """Constant-time check of the shared secret; an empty secret disables the gate."""
    if not expected:
        return True
    return presented is not None and hmac.compare_digest(presented, expected)


def twiml_reply(text: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Message>{escape(text)}</Message></Response>"
    )
