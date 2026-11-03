"""Audio DSP effects for voice presets (Phase 9 - Shadow Voice).

Provides pitch shifting, low-shelf bass boost, high-cut filtering, stateful
multi-tap echo/reverb, and soft-limiting using numpy and scipy only.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import signal as sp_signal
    _SCIPY_AVAILABLE = True
except ImportError:
    sp_signal = None  # type: ignore
    _SCIPY_AVAILABLE = False

_logger = logging.getLogger("jarvis.voice_fx")


@dataclass
class VoiceFXConfig:
    """Configuration for audio FX chain."""

    enabled: bool = False
    semitones: float = -2.5
    bass_boost_db: float = 4.5
    bass_cutoff_hz: float = 200.0
    high_cut_hz: float = 7000.0
    echo_delays_ms: List[float] = field(default_factory=lambda: [85.0, 150.0, 230.0])
    echo_decays: List[float] = field(default_factory=lambda: [0.35, 0.22, 0.12])
    wet_mix: float = 0.16
    breath_enabled: bool = False
    breath_gain_db: float = -36.0
    limiter_threshold: float = 0.95

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> VoiceFXConfig:
        if not data or not isinstance(data, dict):
            return cls(enabled=False)

        def _safe_float(val, default):
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        def _safe_float_list(vals, default):
            if not isinstance(vals, (list, tuple)):
                return default
            res = []
            for v in vals:
                try:
                    res.append(float(v))
                except (TypeError, ValueError):
                    pass
            return res or default

        return cls(
            enabled=bool(data.get("enabled", False)),
            semitones=_safe_float(data.get("semitones", -2.5), -2.5),
            bass_boost_db=_safe_float(data.get("bass_boost_db", 4.5), 4.5),
            bass_cutoff_hz=_safe_float(data.get("bass_cutoff_hz", 200.0), 200.0),
            high_cut_hz=_safe_float(data.get("high_cut_hz", 7000.0), 7000.0),
            echo_delays_ms=_safe_float_list(
                data.get("echo_delays_ms"), [85.0, 150.0, 230.0]
            ),
            echo_decays=_safe_float_list(
                data.get("echo_decays"), [0.35, 0.22, 0.12]
            ),
            wet_mix=_safe_float(data.get("wet_mix", 0.16), 0.16),
            breath_enabled=bool(data.get("breath_enabled", False)),
            breath_gain_db=_safe_float(data.get("breath_gain_db", -36.0), -36.0),
            limiter_threshold=_safe_float(data.get("limiter_threshold", 0.95), 0.95),
        )


class VoiceFXChain:
    """Stateful audio FX chain for real-time speech transformation.

    Maintains filter states and multi-tap echo buffers across sentence chunks
    so acoustic tails are never truncated at boundaries, with instant zeroing
    on barge-in.
    """

    def __init__(self, config: Optional[VoiceFXConfig] = None):
        self.config = config or VoiceFXConfig(enabled=False)
        self._delay_buffers: List[np.ndarray] = []
        self._delay_indices: List[int] = []
        self._rate: int = 0
        self._bass_zi: Optional[np.ndarray] = None
        self._highcut_zi: Optional[np.ndarray] = None
        self._last_semitones: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def reset(self) -> None:
        """Reset stateful echo buffers and filter history (called on barge-in/turn start)."""
        self._delay_buffers.clear()
        self._delay_indices.clear()
        self._bass_zi = None
        self._highcut_zi = None

    def _init_buffers_if_needed(self, rate: int) -> None:
        if self._rate != rate or not self._delay_buffers:
            self._rate = rate
            self._delay_buffers.clear()
            self._delay_indices.clear()
            for delay_ms in self.config.echo_delays_ms:
                delay_samples = max(1, int(round((delay_ms / 1000.0) * rate)))
                self._delay_buffers.append(np.zeros(delay_samples, dtype=np.float32))
                self._delay_indices.append(0)

    def process(self, samples: np.ndarray, rate: int) -> Tuple[np.ndarray, int]:
        """Apply audio FX chain to mono float32 PCM samples.

        Returns processed samples and output sample rate.
        """
        if not self.config.enabled or samples.size == 0:
            return samples, rate

        audio = np.asarray(samples, dtype=np.float32).copy()

        # 1. Pitch-shift via high-quality resampling + playback rate
        if abs(self.config.semitones) > 1e-3:
            audio, rate = self._pitch_shift(audio, rate, self.config.semitones)

        self._init_buffers_if_needed(rate)

        # 2. Low-shelf bass boost & gentle high cut
        if _SCIPY_AVAILABLE and rate > 0:
            audio = self._apply_filters(audio, rate)

        # 3. Optional envelope-shaped breath layer
        if self.config.breath_enabled:
            audio = self._apply_breath_layer(audio)

        # 4. Stateful multi-tap echo / reverb
        if self.config.wet_mix > 1e-4 and self._delay_buffers:
            audio = self._apply_echo(audio)

        # 5. Soft-knee limiter (guarantees no clipping/distortion)
        audio = self._soft_limit(audio, self.config.limiter_threshold)

        # Ensure no NaN or Inf
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.98, neginf=-0.98)
        return audio.astype(np.float32), rate

    def _pitch_shift(
        self, samples: np.ndarray, rate: int, semitones: float
    ) -> Tuple[np.ndarray, int]:
        """Pitch shift by semitones using precise resampling.

        Resampling by ratio 2^(semitones/12) and playing back at original rate
        drops pitch cleanly and deepens formant resonance without phase-vocoder
        metallic artifacts. Combined with Piper's length_scale, this produces
        an organic, ominous delivery.
        """
        ratio = float(2.0 ** (semitones / 12.0))
        if abs(ratio - 1.0) < 1e-4 or samples.size == 0:
            return samples, rate

        target_length = max(1, int(round(samples.size / ratio)))
        source_x = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        target_x = np.linspace(0.0, 1.0, target_length, endpoint=False)
        resampled = np.interp(target_x, source_x, samples).astype(np.float32)
        return resampled, rate

    def _apply_filters(self, samples: np.ndarray, rate: int) -> np.ndarray:
        """Apply 2nd-order low-shelf boost and Butterworth high-cut."""
        audio = samples

        # Low-shelf filter for warm, dark bass resonance
        if abs(self.config.bass_boost_db) > 1e-3 and self.config.bass_cutoff_hz < rate * 0.45:
            b_shelf, a_shelf = self._design_low_shelf(
                self.config.bass_cutoff_hz, self.config.bass_boost_db, rate
            )
            if self._bass_zi is None or self._bass_zi.size != max(len(a_shelf), len(b_shelf)) - 1:
                self._bass_zi = sp_signal.lfilter_zi(b_shelf, a_shelf) * audio[0] if audio.size else None
            if self._bass_zi is not None:
                audio, self._bass_zi = sp_signal.lfilter(b_shelf, a_shelf, audio, zi=self._bass_zi)
            else:
                audio = sp_signal.lfilter(b_shelf, a_shelf, audio)

        # High-cut filter for dark, intimate timbre
        if self.config.high_cut_hz < (rate * 0.5 - 200):
            nyq = rate * 0.5
            norm_cutoff = min(0.95, max(0.01, self.config.high_cut_hz / nyq))
            b_cut, a_cut = sp_signal.butter(2, norm_cutoff, btype="low")
            if self._highcut_zi is None or self._highcut_zi.size != max(len(a_cut), len(b_cut)) - 1:
                self._highcut_zi = sp_signal.lfilter_zi(b_cut, a_cut) * audio[0] if audio.size else None
            if self._highcut_zi is not None:
                audio, self._highcut_zi = sp_signal.lfilter(b_cut, a_cut, audio, zi=self._highcut_zi)
            else:
                audio = sp_signal.lfilter(b_cut, a_cut, audio)

        return np.asarray(audio, dtype=np.float32)

    def _design_low_shelf(
        self, cutoff_hz: float, gain_db: float, rate: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Audio EQ Cookbook biquad low-shelf filter."""
        a_gain = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * cutoff_hz / rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * 0.707)  # Q = 0.707
        two_sqrt_a_alpha = 2.0 * math.sqrt(a_gain) * alpha

        b0 = a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * cos_w0 + two_sqrt_a_alpha)
        b1 = 2.0 * a_gain * ((a_gain - 1.0) - (a_gain + 1.0) * cos_w0)
        b2 = a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * cos_w0 - two_sqrt_a_alpha)
        a0 = (a_gain + 1.0) + (a_gain - 1.0) * cos_w0 + two_sqrt_a_alpha
        a1 = -2.0 * ((a_gain - 1.0) + (a_gain + 1.0) * cos_w0)
        a2 = (a_gain + 1.0) + (a_gain - 1.0) * cos_w0 - two_sqrt_a_alpha

        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float32)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float32)
        return b, a

    def _apply_breath_layer(self, samples: np.ndarray) -> np.ndarray:
        """Add low-level white noise modulated by the speech envelope."""
        if samples.size == 0:
            return samples

        # Envelope follower
        envelope = np.abs(samples)
        if _SCIPY_AVAILABLE and envelope.size > 16:
            kernel_len = min(64, envelope.size)
            kernel = np.ones(kernel_len, dtype=np.float32) / kernel_len
            envelope = np.convolve(envelope, kernel, mode="same")

        noise = np.random.uniform(-1.0, 1.0, size=samples.size).astype(np.float32)
        gain = 10.0 ** (self.config.breath_gain_db / 20.0)
        breath = noise * envelope * gain
        return samples + breath

    def _apply_echo(self, samples: np.ndarray) -> np.ndarray:
        """Stateful multi-tap delay line with smooth cross-chunk feedback."""
        n_samples = samples.size
        out = samples.copy()
        wet_mix = max(0.0, min(1.0, self.config.wet_mix))
        dry_mix = 1.0 - wet_mix * 0.5

        for buf, decay in zip(self._delay_buffers, self.config.echo_decays):
            buf_len = buf.size
            if buf_len == 0:
                continue

            # Process samples through circular ring buffer
            delay_out = np.zeros(n_samples, dtype=np.float32)
            if n_samples <= buf_len:
                delay_out = buf[:n_samples].copy()
                buf[:-n_samples] = buf[n_samples:]
                buf[-n_samples:] = samples + delay_out * (decay * 0.4)
            else:
                # Chunk longer than buffer: process in slices
                curr = 0
                while curr < n_samples:
                    chunk_sz = min(buf_len, n_samples - curr)
                    delay_out[curr : curr + chunk_sz] = buf[:chunk_sz]
                    buf[:-chunk_sz] = buf[chunk_sz:]
                    buf[-chunk_sz:] = (
                        samples[curr : curr + chunk_sz]
                        + delay_out[curr : curr + chunk_sz] * (decay * 0.4)
                    )
                    curr += chunk_sz

            out += delay_out * (decay * wet_mix)

        return out * dry_mix

    def _soft_limit(self, samples: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """Soft-knee saturation limiter to guarantee zero digital clipping."""
        thresh = max(0.5, min(0.95, threshold))
        ceiling = 0.99
        abs_s = np.abs(samples)
        over = abs_s > thresh

        if not np.any(over):
            return samples

        out = samples.copy()
        margin = ceiling - thresh
        scaled = (abs_s[over] - thresh) / max(1e-4, margin)
        sat = thresh + margin * np.tanh(scaled)
        out[over] = np.sign(samples[over]) * sat
        return np.clip(out, -ceiling, ceiling)
