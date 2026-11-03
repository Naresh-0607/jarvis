"""Audio helpers: resampling, silence trimming and WAV encoding."""

from __future__ import annotations

import io
import wave
from typing import Tuple

import numpy as np

TARGET_RATE = 16000


def resample(samples: np.ndarray, source_rate: int, target_rate: int = TARGET_RATE) -> np.ndarray:
    """Linear resample. Good enough for speech at 16 kHz."""
    if source_rate == target_rate or samples.size == 0:
        return samples
    duration = samples.size / float(source_rate)
    target_count = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, samples.size, endpoint=False)
    target_x = np.linspace(0.0, duration, target_count, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def trim_silence(
    samples: np.ndarray,
    rate: int = TARGET_RATE,
    threshold: float = 0.01,
    pad_ms: int = 200,
    window_ms: int = 20,
    adaptive: bool = True,
) -> np.ndarray:
    """Drop leading and trailing quiet audio, keeping a small pad.

    With ``adaptive`` the threshold follows the clip's own noise floor, so a
    low-gain microphone still keeps soft speech while a noisy room still
    trims. Returns an empty array when the whole clip is quiet.
    """
    if samples.size == 0:
        return samples

    window = max(1, int(rate * window_ms / 1000))
    usable = samples.size - (samples.size % window)
    if usable == 0:
        return samples

    frames = samples[:usable].reshape(-1, window)
    levels = np.sqrt(np.mean(np.square(frames), axis=1))

    effective = threshold
    if adaptive and levels.size:
        floor = float(np.percentile(levels, 20))
        effective = max(0.002, min(threshold, floor * 2.5))

    loud = levels > effective
    if not loud.any():
        return np.empty(0, dtype=samples.dtype)

    first = int(np.argmax(loud))
    last = int(len(loud) - np.argmax(loud[::-1]))
    pad = int(rate * pad_ms / 1000)
    start = max(0, first * window - pad)
    end = min(samples.size, last * window + pad)
    return samples[start:end]


def to_wav_bytes(samples: np.ndarray, rate: int = TARGET_RATE) -> bytes:
    """Encode float32 [-1, 1] mono samples as 16-bit PCM WAV."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def wav_duration(wav: bytes) -> float:
    """Duration of a WAV blob in seconds (0.0 if unreadable)."""
    try:
        with wave.open(io.BytesIO(wav), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate() or 1
            return frames / float(rate)
    except Exception:  # noqa: BLE001 - only used for logging/tests
        return 0.0


def rms(samples: np.ndarray) -> float:
    """Root-mean-square level, handy for logs and tests."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def prepare(samples: np.ndarray, rate: int, threshold: float, trim: bool) -> Tuple[np.ndarray, int]:
    """Resample to 16 kHz and optionally trim silence."""
    audio = resample(samples, rate)
    if trim:
        audio = trim_silence(audio, TARGET_RATE, threshold=threshold)
    return audio, TARGET_RATE
