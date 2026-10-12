"""Local speech-to-text: whisper.cpp wrapper, routes, and the UI hook."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import create_app
from core.chat import ChatService
from core.metrics_store import MetricsStore
from core.store import CareStore
from core.transcribe import (
    DEFAULT_PROMPT,
    NullTranscriber,
    TranscribeError,
    WhisperCLI,
    transcriber_from_env,
)
from router.config import RouterConfig
from tests.test_api import MOCK_ENV, mock_agent_client


def _fake_runner(calls: list[list[str]], stdout: str = " Did Mom take her metformin?\n"):
    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        if args[0] == "ffmpeg":
            Path(args[-1]).write_bytes(b"RIFF....WAVE")
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 0, stdout, "")
    return run


def test_whisper_cli_converts_then_decodes_with_the_family_prompt() -> None:
    calls: list[list[str]] = []
    stt = WhisperCLI("whisper-cli", "/models/ggml-base.en.bin", runner=_fake_runner(calls))
    transcript = stt.transcribe(b"\x1aE\xdf\xa3 webm bytes", "audio/webm;codecs=opus")
    assert transcript.text == "Did Mom take her metformin?"
    assert transcript.engine == "whisper.cpp" and transcript.model == "ggml-base.en.bin"
    ffmpeg, whisper = calls
    assert ffmpeg[0] == "ffmpeg" and ffmpeg[ffmpeg.index("-i") + 1].endswith(".webm")
    assert ffmpeg[ffmpeg.index("-ar") + 1] == "16000"
    assert whisper[:3] == ["whisper-cli", "-m", "/models/ggml-base.en.bin"]
    assert whisper[whisper.index("--prompt") + 1] == DEFAULT_PROMPT
    assert "Pavel" in DEFAULT_PROMPT and "metformin" in DEFAULT_PROMPT
    assert stt.status() == {"enabled": True, "engine": "whisper.cpp", "model": "ggml-base.en.bin"}


def test_whisper_cli_rejects_empty_and_oversized_clips() -> None:
    stt = WhisperCLI("whisper-cli", "m.bin", runner=_fake_runner([]))
    with pytest.raises(TranscribeError) as empty:
        stt.transcribe(b"")
    assert empty.value.status_code == 422
    with pytest.raises(TranscribeError) as huge:
        stt.transcribe(b"x" * (9 * 1024 * 1024))
    assert huge.value.status_code == 413


def test_env_without_binary_or_model_gives_null(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STT_WHISPER_BIN", "")
    monkeypatch.setenv("STT_WHISPER_MODEL", str(tmp_path / "missing.bin"))
    monkeypatch.setattr("core.transcribe.shutil.which", lambda name: None)
    stt = transcriber_from_env()
    assert isinstance(stt, NullTranscriber)
    assert stt.status()["enabled"] is False
    with pytest.raises(TranscribeError) as off:
        stt.transcribe(b"abc")
    assert off.value.status_code == 503


def _service(tmp_path: Path, transcriber: object) -> ChatService:
    db = tmp_path / "t.sqlite3"
    return ChatService(
        client=mock_agent_client(), config=RouterConfig.from_env(MOCK_ENV),
        store=CareStore(db), metrics=MetricsStore(db),
        transcriber=transcriber,  # type: ignore[arg-type]
    )


def test_transcribe_routes(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    stt = WhisperCLI("whisper-cli", "m.bin", runner=_fake_runner(calls))
    api = TestClient(create_app(service=_service(tmp_path, stt)))
    assert api.get("/transcribe").json()["enabled"] is True
    response = api.post(
        "/transcribe", content=b"webm bytes", headers={"Content-Type": "audio/webm"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "text": "Did Mom take her metformin?", "engine": "whisper.cpp", "model": "m.bin",
    }
    assert api.post("/transcribe", content=b"", headers={"Content-Type": "audio/webm"}).status_code == 422

    off = TestClient(create_app(service=_service(tmp_path, NullTranscriber("off"))))
    assert off.get("/transcribe").json()["enabled"] is False
    assert off.post("/transcribe", content=b"abc", headers={"Content-Type": "audio/webm"}).status_code == 503


def test_ui_records_locally_when_the_server_offers_it(tmp_path: Path) -> None:
    html = TestClient(create_app(service=_service(tmp_path, NullTranscriber("off")))).get("/").text
    for required in ("fetch('/transcribe'", "MediaRecorder", "function listenLocal", "RECORD_SILENCE_MS"):
        assert required in html, f"the served page is missing {required}"
