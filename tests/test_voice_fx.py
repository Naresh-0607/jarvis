"""Unit tests for voice_fx module (Phase 9 - Voice Effects & Shadow Voice)."""

import numpy as np
import pytest

from jarvis.voice_fx import VoiceFXChain, VoiceFXConfig


def test_disabled_chain_is_identity():
    """Disabled chain should return original samples and rate untouched."""
    fx = VoiceFXChain(VoiceFXConfig(enabled=False))
    samples = np.array([0.1, -0.2, 0.5, -0.4], dtype=np.float32)
    rate = 22050
    out_samples, out_rate = fx.process(samples, rate)
    np.testing.assert_array_equal(out_samples, samples)
    assert out_rate == rate


def test_empty_samples_handled_gracefully():
    """Empty arrays should not raise errors."""
    fx = VoiceFXChain(VoiceFXConfig(enabled=True))
    samples = np.array([], dtype=np.float32)
    out_samples, out_rate = fx.process(samples, 22050)
    assert out_samples.size == 0
    assert out_rate == 22050


def test_soft_limiter_bounds():
    """Soft limiter must ensure output stays strictly within [-1.0, 1.0] even with extreme input."""
    cfg = VoiceFXConfig(
        enabled=True,
        semitones=0.0,
        bass_boost_db=12.0,
        limiter_threshold=0.95,
        wet_mix=0.0,
    )
    fx = VoiceFXChain(cfg)
    rate = 22050
    # Overdriven signal well beyond 1.0
    overshot = np.sin(np.linspace(0, 100 * np.pi, 2000, dtype=np.float32)) * 5.0
    out_samples, _ = fx.process(overshot, rate)

    assert not np.any(np.isnan(out_samples))
    assert not np.any(np.isinf(out_samples))
    assert np.all(out_samples <= 1.0)
    assert np.all(out_samples >= -1.0)
    assert np.max(np.abs(out_samples)) <= 0.99


def test_echo_tail_continuity_across_chunks():
    """Stateful echo should carry over delay tail into subsequent silent chunk."""
    cfg = VoiceFXConfig(
        enabled=True,
        semitones=0.0,
        bass_boost_db=0.0,
        high_cut_hz=20000.0,
        echo_delays_ms=[10.0],
        echo_decays=[0.5],
        wet_mix=0.5,
    )
    fx = VoiceFXChain(cfg)
    rate = 16000

    # Impulse chunk: single click followed by silence
    chunk1 = np.zeros(320, dtype=np.float32)  # 20ms
    chunk1[0] = 1.0

    out1, _ = fx.process(chunk1, rate)
    assert np.max(np.abs(out1)) > 0.0

    # Next chunk is pure silence
    chunk2 = np.zeros(320, dtype=np.float32)
    out2, _ = fx.process(chunk2, rate)

    # Chunk2 should contain the echo from Chunk1 (not silence!)
    assert np.max(np.abs(out2)) > 0.01


def test_reset_clears_echo_buffers():
    """Reset (called on barge-in) must instantly clear echoes so no tail leaks."""
    cfg = VoiceFXConfig(
        enabled=True,
        semitones=0.0,
        bass_boost_db=0.0,
        high_cut_hz=20000.0,
        echo_delays_ms=[50.0],
        echo_decays=[0.8],
        wet_mix=0.5,
    )
    fx = VoiceFXChain(cfg)
    rate = 16000

    # First feed loud impulse
    chunk1 = np.zeros(160, dtype=np.float32)
    chunk1[0] = 1.0
    fx.process(chunk1, rate)

    # Barge-in: reset()
    fx.reset()

    # Now silent chunk should produce pure silence
    chunk2 = np.zeros(160, dtype=np.float32)
    out2, _ = fx.process(chunk2, rate)
    assert np.all(np.abs(out2) < 1e-6)


def test_pitch_shift_preserves_audio_quality_and_finite_values():
    """Resampling pitch-shift should not produce NaNs or infs."""
    cfg = VoiceFXConfig(
        enabled=True,
        semitones=-2.5,
        bass_boost_db=4.5,
        high_cut_hz=7000.0,
        wet_mix=0.15,
    )
    fx = VoiceFXChain(cfg)
    rate = 22050
    t = np.linspace(0, 0.5, int(rate * 0.5), endpoint=False, dtype=np.float32)
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)

    out_samples, out_rate = fx.process(sine_wave, rate)
    assert out_rate == rate
    assert out_samples.size > 0
    assert not np.any(np.isnan(out_samples))
    assert not np.any(np.isinf(out_samples))


def test_config_from_dict():
    """Config parses safely from dict or missing keys."""
    cfg = VoiceFXConfig.from_dict({
        "enabled": True,
        "semitones": -3.0,
        "echo_delays_ms": [80, 160],
    })
    assert cfg.enabled is True
    assert cfg.semitones == -3.0
    assert cfg.echo_delays_ms == [80.0, 160.0]
    assert cfg.high_cut_hz == 7000.0  # default

