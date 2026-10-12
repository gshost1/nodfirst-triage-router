"""Read the digest aloud through DigitalOcean text-to-speech.

DigitalOcean's serverless inference exposes `POST /v1/audio/speech` and the
`qwen3-tts-voicedesign` model (verified 2026-10-12: a `voice` string is
required and overwritten to "default" by the gateway, `instructions` carries
the voice design, and `response_format` "wav" returns 24 kHz mono PCM). There
is no speech-to-text route on DigitalOcean, so voice input stays in the
browser. The API key is never logged or echoed.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_TTS_MODEL = "qwen3-tts-voicedesign"
DEFAULT_VOICE_INSTRUCTIONS = "a warm, calm adult woman speaking clearly and slowly"
MAX_SPEECH_CHARS = 320  # longer inputs time out on the gateway (verified 2026-10-12)
SPEECH_TIMEOUT_SECONDS = 180.0

# Returns (status_code, content_type, body bytes); raises on transport errors.
Transport = Callable[[str, dict[str, str], dict[str, Any]], tuple[int, str, bytes]]


class SpeechError(RuntimeError):
    """Synthesis failed; `status_code` is what the API should return."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class SpeechClip:
    audio: bytes
    media_type: str
    model: str


class NullSpeech:
    """No synthesis: MOCK mode or no credential. The UI falls back to the browser."""

    def __init__(self, reason: str = "text-to-speech not configured") -> None:
        self.reason = reason

    def status(self) -> dict[str, Any]:
        return {"enabled": False, "provider": None, "model": None, "reason": self.reason}

    def synthesize(self, text: str) -> SpeechClip:
        raise SpeechError(self.reason, status_code=503)


class DOSpeech:
    """Text-to-speech on the DigitalOcean OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = DEFAULT_TTS_MODEL,
        instructions: str = DEFAULT_VOICE_INSTRUCTIONS,
        transport: Transport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("DOSpeech needs an API key")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.instructions = instructions
        self.transport = transport or _httpx_transport

    def status(self) -> dict[str, Any]:
        return {"enabled": True, "provider": "digitalocean", "model": self.model}

    def synthesize(self, text: str) -> SpeechClip:
        text = text.strip()
        if not text:
            raise SpeechError("nothing to read", status_code=422)
        payload = {
            "model": self.model,
            "input": text[:MAX_SPEECH_CHARS],
            "voice": "default",
            "instructions": self.instructions,
            "response_format": "wav",
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            status, _content_type, body = self.transport(
                f"{self.base_url}/audio/speech", headers, payload
            )
        except Exception as error:  # noqa: BLE001 - never leak the request
            raise SpeechError(
                f"text-to-speech request failed: {type(error).__name__}"
            ) from None
        if status != 200:
            raise SpeechError(f"text-to-speech returned HTTP {status}")
        if not body.startswith(b"RIFF"):
            raise SpeechError("text-to-speech returned something other than WAV audio")
        return SpeechClip(body, "audio/wav", self.model)


def _httpx_transport(
    url: str, headers: dict[str, str], payload: dict[str, Any]
) -> tuple[int, str, bytes]:
    response = httpx.post(url, headers=headers, json=payload, timeout=SPEECH_TIMEOUT_SECONDS)
    return response.status_code, response.headers.get("content-type", ""), response.content


def speech_from_env() -> NullSpeech | DOSpeech:
    """Reuse the LLM endpoint and key; MOCK mode or a blank key disables speech."""
    base_url = os.getenv("LLM_BASE_URL", "mock").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if base_url.lower() == "mock":
        return NullSpeech("LLM_BASE_URL=mock: text-to-speech stays in the browser")
    if not api_key:
        return NullSpeech("LLM_API_KEY is not set")
    return DOSpeech(
        base_url=base_url,
        api_key=api_key,
        model=os.getenv("TTS_MODEL", "").strip() or DEFAULT_TTS_MODEL,
        instructions=os.getenv("TTS_VOICE", "").strip() or DEFAULT_VOICE_INSTRUCTIONS,
    )
