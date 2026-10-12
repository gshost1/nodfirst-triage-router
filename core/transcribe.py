"""Local speech-to-text with whisper.cpp (`whisper-cli`), no provider involved.

The inference provider has no speech-to-text model (verified 2026-10-12) and
the browser's built-in recognition cut sentences short and misheard names, so
voice input runs through whisper.cpp on this machine: the page records a
clip, `POST /transcribe` converts it to 16 kHz mono WAV with ffmpeg and runs
`whisper-cli` with a prompt that primes the family's names and Mom's
medications. Measured here (Apple M2, `ggml-base.en`): an 11 s clip in
under 1 s.

Configuration (all optional, see `.env.example`):
  STT_WHISPER_BIN    path to whisper-cli (default: found on PATH)
  STT_WHISPER_MODEL  path to a ggml model (default ~/.cache/whisper-cpp/ggml-base.en.bin)
  STT_FFMPEG         path to ffmpeg (default: found on PATH)
  STT_PROMPT         words to prime the decoder with
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = Path.home() / ".cache" / "whisper-cpp" / "ggml-base.en.bin"
DEFAULT_PROMPT = (
    "Mom, Ludmila, Anna, Pavel, metformin, lisinopril, vitamin D, cardiology, "
    "Dr. Reyes, physiotherapy, pharmacy refill, eye exam, digest."
)
MAX_AUDIO_BYTES = 8 * 1024 * 1024  # about 60 s of browser audio
TRANSCRIBE_TIMEOUT_SECONDS = 60.0
ENGINE = "whisper.cpp"


class TranscribeError(RuntimeError):
    def __init__(self, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class Transcript:
    text: str
    engine: str
    model: str


class NullTranscriber:
    """Speech-to-text is off; the page falls back to browser recognition."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def status(self) -> dict[str, Any]:
        return {"enabled": False, "engine": None, "model": None, "reason": self.reason}

    def transcribe(self, audio: bytes, media_type: str = "audio/webm") -> Transcript:
        raise TranscribeError(self.reason, status_code=503)


class WhisperCLI:
    """Run whisper.cpp's command line on a clip; ffmpeg does the resampling."""

    def __init__(
        self,
        binary: str,
        model: str | Path,
        *,
        ffmpeg: str = "ffmpeg",
        prompt: str = DEFAULT_PROMPT,
        runner: Any = subprocess.run,
    ) -> None:
        self.binary = binary
        self.model = str(model)
        self.ffmpeg = ffmpeg
        self.prompt = prompt
        self._run = runner

    def status(self) -> dict[str, Any]:
        return {"enabled": True, "engine": ENGINE, "model": Path(self.model).name}

    def transcribe(self, audio: bytes, media_type: str = "audio/webm") -> Transcript:
        if not audio:
            raise TranscribeError("no audio received", status_code=422)
        if len(audio) > MAX_AUDIO_BYTES:
            raise TranscribeError("clip too long", status_code=413)
        with tempfile.TemporaryDirectory(prefix="nodfirst-stt-") as folder:
            source = Path(folder) / ("clip" + _extension(media_type))
            wav = Path(folder) / "clip.wav"
            source.write_bytes(audio)
            self._convert(source, wav)
            text = self._decode(wav)
        return Transcript(text=text, engine=ENGINE, model=Path(self.model).name)

    def _convert(self, source: Path, wav: Path) -> None:
        result = self._run(
            [self.ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
             "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)],
            capture_output=True, timeout=TRANSCRIBE_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise TranscribeError("could not decode the audio clip", status_code=422)

    def _decode(self, wav: Path) -> str:
        result = self._run(
            [self.binary, "-m", self.model, "-f", str(wav), "-l", "en",
             "-nt", "-np", "--prompt", self.prompt],
            capture_output=True, timeout=TRANSCRIBE_TIMEOUT_SECONDS, text=True,
        )
        if result.returncode != 0:
            raise TranscribeError("speech-to-text failed", status_code=503)
        lines = [line.strip() for line in str(result.stdout).splitlines()]
        return " ".join(line for line in lines if line and not line.startswith("[")).strip()


def _extension(media_type: str) -> str:
    base = media_type.split(";", 1)[0].strip().lower()
    return {
        "audio/webm": ".webm", "video/webm": ".webm", "audio/ogg": ".ogg",
        "audio/mp4": ".m4a", "audio/mpeg": ".mp3", "audio/wav": ".wav",
        "audio/x-wav": ".wav", "audio/wave": ".wav",
    }.get(base, ".bin")


def transcriber_from_env() -> WhisperCLI | NullTranscriber:
    binary = os.getenv("STT_WHISPER_BIN", "").strip() or shutil.which("whisper-cli") or ""
    ffmpeg = os.getenv("STT_FFMPEG", "").strip() or shutil.which("ffmpeg") or ""
    model = Path(os.getenv("STT_WHISPER_MODEL", "").strip() or DEFAULT_MODEL).expanduser()
    if not binary:
        return NullTranscriber("whisper-cli not installed")
    if not ffmpeg:
        return NullTranscriber("ffmpeg not installed")
    if not model.is_file():
        return NullTranscriber(f"model missing: {model.name}")
    prompt = os.getenv("STT_PROMPT", "").strip() or DEFAULT_PROMPT
    return WhisperCLI(binary, model, ffmpeg=ffmpeg, prompt=prompt)
