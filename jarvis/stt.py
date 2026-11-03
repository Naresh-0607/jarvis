"""Speech-to-text with a swappable backend.

Primary backend: Groq's OpenAI-compatible /audio/transcriptions endpoint
(whisper-large-v3-turbo). Optional local backend: faster-whisper, disabled
unless `voice.stt_backend: local` is set in config.yaml.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Dict, Optional, Protocol

import httpx

from jarvis.db import PROJECT_ROOT

_logger = logging.getLogger("jarvis.stt")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class TranscriptionError(Exception):
    """Speech-to-text failed in a way the caller should report."""


class Transcriber(Protocol):
    """Anything that can turn a WAV blob into text."""

    name: str

    def transcribe(self, wav: bytes) -> str: ...


def _api_key(explicit: Optional[str], env_name: str) -> Optional[str]:
    key = explicit or os.environ.get(env_name, "")
    if not key.strip():
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
            key = os.environ.get(env_name, "")
        except Exception:  # noqa: BLE001 - optional
            _logger.debug("Could not load .env for %s", env_name)
    return key.strip() or None


class GroqTranscriber:
    """Groq whisper transcription."""

    name = "groq"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_key_env: str = "GROQ_API_KEY",
        base_url: str = GROQ_BASE_URL,
        model: str = "whisper-large-v3-turbo",
        language: str = "en",
        prompt: str = "",
        timeout: float = 30.0,
        retries: int = 1,
        client: Optional[httpx.Client] = None,
    ):
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.language = language
        self.prompt = prompt
        self.timeout = timeout
        self.retries = max(0, retries)
        self._client = client or httpx.Client()
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def transcribe(self, wav: bytes) -> str:
        key = _api_key(self.api_key, self.api_key_env)
        if not key:
            raise TranscriptionError(
                f"{self.api_key_env} is not set, so speech cannot be transcribed"
            )

        data: Dict[str, str] = {
            "model": self.model,
            "response_format": "json",
            "temperature": "0",
        }
        if self.language:
            data["language"] = self.language
        if self.prompt:
            data["prompt"] = self.prompt

        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                response = self._client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}"},
                    files={"file": ("speech.wav", wav, "audio/wav")},
                    data=data,
                    timeout=self.timeout,
                )

                if response.status_code in (401, 403):
                    raise TranscriptionError(
                        "Groq rejected the API key for speech-to-text"
                    )
                if response.status_code >= 400:
                    # Bad/too-short audio is permanent; retrying will not help.
                    raise TranscriptionError(
                        f"Groq transcription failed: HTTP {response.status_code} "
                        f"{response.text[:200]}"
                    )

                payload = response.json()
                text = payload.get("text")
                return (text if text is not None else response.text).strip()

            except TranscriptionError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt < self.retries:
                    _logger.warning("Transcription attempt %s failed: %s", attempt + 1, exc)
                    time.sleep(0.6)

        raise TranscriptionError(f"Could not reach Groq for transcription: {last_error}")


class LocalWhisperTranscriber:
    """faster-whisper, offline. Off by default; needs the optional package."""

    name = "local"

    def __init__(self, model_size: str = "base.en", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError(
                "faster-whisper is not installed; run "
                "'pip install faster-whisper' or set voice.stt_backend: groq"
            ) from exc

        _logger.info("Loading local whisper model %s", self.model_size)
        self._model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )
        return self._model

    def transcribe(self, wav: bytes) -> str:
        model = self._load()
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            handle.write(wav)
            handle.close()
            segments, _info = model.transcribe(handle.name, language="en")
            return " ".join(segment.text.strip() for segment in segments).strip()
        finally:
            try:
                os.unlink(handle.name)
            except OSError:
                pass


def build_transcriber(cfg: Optional[Dict] = None, client: Optional[httpx.Client] = None):
    """Pick a backend from `voice` config (flat or nested under "voice")."""
    full = cfg or {}
    voice = full.get("voice") if isinstance(full.get("voice"), dict) else full
    backend = str(voice.get("stt_backend", "groq")).lower()

    if backend == "local":
        return LocalWhisperTranscriber(
            model_size=voice.get("local_model", "base.en"),
            device=voice.get("local_device", "cpu"),
            compute_type=voice.get("local_compute_type", "int8"),
        )

    return GroqTranscriber(
        api_key_env=voice.get("stt_api_key_env", "GROQ_API_KEY"),
        base_url=voice.get("stt_base_url", GROQ_BASE_URL),
        model=voice.get("stt_model", "whisper-large-v3-turbo"),
        language=voice.get("language", "en"),
        prompt=voice.get("stt_prompt", ""),
        client=client,
    )
