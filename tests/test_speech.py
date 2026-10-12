"""Voice: DigitalOcean text-to-speech for the digest, browser fallback otherwise."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import create_app
from core.chat import ChatService
from core.metrics_store import MetricsStore
from core.speech import DOSpeech, NullSpeech, SpeechError, speech_from_env
from core.store import CareStore
from router.config import RouterConfig
from tests.test_api import MOCK_ENV, mock_agent_client

WAV = b"RIFF" + b"\x00" * 40


def service(tmp_path: Path, speech: object = None) -> ChatService:
    db = tmp_path / "v.sqlite3"
    kwargs = {"speech": speech} if speech is not None else {}
    return ChatService(
        client=mock_agent_client(), config=RouterConfig.from_env(MOCK_ENV),
        store=CareStore(db), metrics=MetricsStore(db), **kwargs,  # type: ignore[arg-type]
    )


def fake_transport(calls: list[dict[str, object]], *, status: int = 200, body: bytes = WAV):
    def transport(url: str, headers: dict[str, str], payload: dict[str, object]):
        calls.append({"url": url, "headers": headers, "payload": payload})
        return status, "application/octet-stream", body
    return transport


def test_digest_speak_returns_wav_from_digitalocean_without_leaking_the_key(
    tmp_path: Path, outbound_connections: list[object],
) -> None:
    calls: list[dict[str, object]] = []
    speech = DOSpeech(
        base_url="https://inference.do-ai.run/v1", api_key="doo-v1-secret",
        transport=fake_transport(calls),
    )
    api = TestClient(create_app(service=service(tmp_path, speech)))

    response = api.post("/digest/speak")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-speech-model"] == "qwen3-tts-voicedesign"
    assert response.content == WAV
    call = calls[0]
    assert call["url"] == "https://inference.do-ai.run/v1/audio/speech"
    assert call["headers"]["Authorization"] == "Bearer doo-v1-secret"  # type: ignore[index]
    payload = call["payload"]
    assert payload["model"] == "qwen3-tts-voicedesign"  # type: ignore[index]
    assert payload["voice"] == "default"  # type: ignore[index]
    assert payload["response_format"] == "wav"  # type: ignore[index]
    assert str(payload["input"]).startswith("Daily digest for Mom")  # type: ignore[index]
    assert outbound_connections == []
    assert api.get("/digest").json()["speech"] == {
        "enabled": True, "provider": "digitalocean", "model": "qwen3-tts-voicedesign",
    }


def test_speech_failures_map_to_api_errors_without_the_key(tmp_path: Path) -> None:
    bad_status = DOSpeech(base_url="https://x", api_key="doo-v1-secret",
                          transport=fake_transport([], status=503))
    with pytest.raises(SpeechError) as excinfo:
        bad_status.synthesize("hi")
    assert str(excinfo.value) == "text-to-speech returned HTTP 503"
    assert excinfo.value.status_code == 502

    not_wav = DOSpeech(base_url="https://x", api_key="doo-v1-secret",
                       transport=fake_transport([], body=b'{"error":"x"}'))
    with pytest.raises(SpeechError, match="other than WAV"):
        not_wav.synthesize("hi")

    def boom(url: str, headers: dict[str, str], payload: dict[str, object]):
        raise TimeoutError("doo-v1-secret must not appear")
    slow = DOSpeech(base_url="https://x", api_key="doo-v1-secret", transport=boom)
    with pytest.raises(SpeechError) as timed_out:
        slow.synthesize("hi")
    assert "doo-v1" not in str(timed_out.value)
    assert str(timed_out.value) == "text-to-speech request failed: TimeoutError"

    api = TestClient(create_app(service=service(tmp_path, bad_status)))
    assert api.post("/digest/speak").status_code == 502


def test_mock_mode_reports_503_so_the_page_uses_browser_speech(tmp_path: Path) -> None:
    api = TestClient(create_app(service=service(tmp_path)))
    response = api.post("/digest/speak")
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]
    assert api.get("/digest").json()["speech"]["enabled"] is False


def test_speech_from_env_follows_the_llm_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "TTS_MODEL", "TTS_VOICE"):
        monkeypatch.delenv(name, raising=False)
    assert isinstance(speech_from_env(), NullSpeech)
    monkeypatch.setenv("LLM_BASE_URL", "https://inference.do-ai.run/v1")
    assert "LLM_API_KEY" in speech_from_env().reason  # type: ignore[union-attr]
    monkeypatch.setenv("LLM_API_KEY", "doo-v1-x")
    monkeypatch.setenv("TTS_VOICE", "a calm narrator")
    built = speech_from_env()
    assert isinstance(built, DOSpeech)
    assert built.instructions == "a calm narrator"
    assert built.status()["model"] == "qwen3-tts-voicedesign"


def test_ui_has_mic_and_read_aloud_controls(tmp_path: Path) -> None:
    html = TestClient(create_app(service=service(tmp_path))).get("/").text
    for required in (
        'id="mic-control"', 'id="digest-speak"', 'id="digest-audio"',
        "fetch('/digest/speak'", "webkitSpeechRecognition", "speechSynthesis",
    ):
        assert required in html, required
