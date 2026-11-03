"""Tests for voice input helpers, STT backends and the capture pipeline."""

import io
import json
import wave

import httpx
import numpy as np
import pytest

from jarvis import audio as audio_utils
from jarvis.stt import (
    GroqTranscriber,
    LocalWhisperTranscriber,
    TranscriptionError,
    build_transcriber,
)
from jarvis.voice_input import HoldToTalk, TranscriptQueue, parse_hotkey


# ---------------------------------------------------------------------------
# Hotkey parsing
# ---------------------------------------------------------------------------
def test_parse_default_hotkey():
    modifiers, main = parse_hotkey("<ctrl>+<alt>+<space>")
    assert modifiers == {"ctrl", "alt"}
    assert main == "space"


def test_parse_hotkey_with_function_key():
    modifiers, main = parse_hotkey("<ctrl>+<shift>+<f9>")
    assert modifiers == {"ctrl", "shift"}
    assert main == "f9"


def test_parse_empty_hotkey_falls_back_to_default():
    modifiers, main = parse_hotkey("")
    assert modifiers == {"ctrl", "alt"}
    assert main == "space"


# ---------------------------------------------------------------------------
# Audio maths
# ---------------------------------------------------------------------------
def test_wav_roundtrip_is_16khz_mono_16bit():
    samples = np.zeros(1600, dtype=np.float32)
    wav = audio_utils.to_wav_bytes(samples, 16000)

    with wave.open(io.BytesIO(wav), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        assert handle.getnframes() == 1600

    assert audio_utils.wav_duration(wav) == pytest.approx(0.1, abs=0.001)


def test_trim_silence_removes_quiet_edges():
    rate = 16000
    quiet = np.zeros(rate, dtype=np.float32)
    loud = np.ones(rate, dtype=np.float32) * 0.2
    samples = np.concatenate([quiet, loud, quiet])

    trimmed = audio_utils.trim_silence(samples, rate, threshold=0.01, pad_ms=100)

    assert trimmed.size < samples.size
    assert trimmed.size >= loud.size
    assert audio_utils.rms(trimmed) > 0.1


def test_trim_silence_returns_empty_for_silence():
    samples = np.zeros(16000, dtype=np.float32)
    assert audio_utils.trim_silence(samples).size == 0


def test_adaptive_threshold_keeps_soft_speech_on_a_quiet_mic():
    rate = 16000
    rng = np.random.default_rng(7)
    floor = (rng.normal(0, 0.0002, rate)).astype(np.float32)
    t = np.linspace(0, 0.5, rate // 2, endpoint=False)
    soft_speech = (0.006 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    samples = np.concatenate([floor, soft_speech, floor])

    adaptive = audio_utils.trim_silence(samples, rate, threshold=0.01)
    fixed = audio_utils.trim_silence(samples, rate, threshold=0.01, adaptive=False)

    assert adaptive.size > 0
    assert fixed.size == 0


def test_resample_changes_length_but_not_duration():
    source = np.zeros(44100, dtype=np.float32)
    out = audio_utils.resample(source, 44100, 16000)
    assert out.size == 16000


# ---------------------------------------------------------------------------
# Groq transcription
# ---------------------------------------------------------------------------
def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_groq_transcriber_sends_expected_fields():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.content
        return httpx.Response(200, json={"text": "  hello there  "})

    transcriber = GroqTranscriber(
        api_key="test-key",
        language="en",
        prompt="Naresh, Kaggle",
        client=make_client(handler),
    )
    text = transcriber.transcribe(b"RIFFfake")

    assert text == "hello there"
    assert seen["path"].endswith("/audio/transcriptions")
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert b"whisper-large-v3-turbo" in body
    assert b'"en"' in body or b"en" in body
    assert b"Naresh, Kaggle" in body


def test_groq_transcriber_rejects_bad_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "nope"})

    transcriber = GroqTranscriber(api_key="bad", client=make_client(handler))
    with pytest.raises(TranscriptionError, match="rejected"):
        transcriber.transcribe(b"RIFFfake")


def test_groq_transcriber_retries_once_on_network_error():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"text": "second try"})

    transcriber = GroqTranscriber(
        api_key="k", retries=1, client=make_client(handler)
    )
    assert transcriber.transcribe(b"RIFFfake") == "second try"
    assert calls["count"] == 2


def test_groq_transcriber_does_not_retry_bad_audio():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, json={"error": "file too short"})

    transcriber = GroqTranscriber(api_key="k", client=make_client(handler))
    with pytest.raises(TranscriptionError):
        transcriber.transcribe(b"short")
    assert calls["count"] == 1


def test_build_transcriber_defaults_to_groq():
    assert isinstance(build_transcriber({"voice": {}}), GroqTranscriber)
    assert isinstance(build_transcriber({}), GroqTranscriber)
    assert isinstance(build_transcriber({"stt_backend": "groq"}), GroqTranscriber)


def test_build_transcriber_local_backend_is_opt_in():
    nested = build_transcriber({"voice": {"stt_backend": "local"}})
    flat = build_transcriber({"stt_backend": "local"})
    assert isinstance(nested, LocalWhisperTranscriber)
    assert isinstance(flat, LocalWhisperTranscriber)


# ---------------------------------------------------------------------------
# Capture -> transcript pipeline
# ---------------------------------------------------------------------------
class FakeTranscriber:
    name = "fake"

    def __init__(self, text: str = "hello there"):
        self.text = text
        self.received = b""

    def transcribe(self, wav: bytes) -> str:
        self.received = wav
        return self.text


def make_voice(transcriber, *messages):
    statuses = []
    reports = TranscriptQueue()
    voice = HoldToTalk(
        {"voice": {"min_seconds": 0.4, "trim_silence": True}},
        on_transcript=reports.put,
        status=statuses.append,
        transcriber=transcriber,
    )
    return voice, statuses, reports


def test_pipeline_produces_a_transcript():
    fake = FakeTranscriber("hello there")
    voice, statuses, reports = make_voice(fake)

    voice._transcribe(np.full(16000 * 2, 0.05, dtype=np.float32), 16000)

    assert reports.get_nowait() == "hello there"
    assert audio_utils.wav_duration(fake.received) > 1.0
    assert any("transcript in" in s for s in statuses)


def test_pipeline_ignores_too_short_audio():
    fake = FakeTranscriber()
    voice, statuses, reports = make_voice(fake)

    voice._transcribe(np.full(1600, 0.05, dtype=np.float32), 16000)

    assert reports.get_nowait() is None
    assert fake.received == b""
    assert any("too short" in s for s in statuses)


def test_pipeline_ignores_silence():
    fake = FakeTranscriber()
    voice, statuses, reports = make_voice(fake)

    voice._transcribe(np.zeros(16000 * 2, dtype=np.float32), 16000)

    assert reports.get_nowait() is None
    assert any("silence" in s for s in statuses)


def test_pipeline_reports_transcription_errors():
    class Broken:
        name = "broken"

        def transcribe(self, wav: bytes) -> str:
            raise TranscriptionError("network down")

    voice, statuses, reports = make_voice(Broken())

    voice._transcribe(np.full(16000 * 2, 0.05, dtype=np.float32), 16000)

    assert reports.get_nowait() is None
    assert any("could not hear you" in s for s in statuses)


def test_transcript_queue_is_fifo():
    queue_ = TranscriptQueue()
    queue_.put("one")
    queue_.put("two")
    assert queue_.get_nowait() == "one"
    assert queue_.get_nowait() == "two"
    assert queue_.get_nowait() is None
