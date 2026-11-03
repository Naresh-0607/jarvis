"""Text-to-speech engines.

edge-tts is the primary (online, natural). Piper is the local backup; it is
only used when the binary and model are configured and present.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np

_logger = logging.getLogger("jarvis.tts")

Audio = Tuple[np.ndarray, int]

DEFAULT_VOICE = "en-GB-SoniaNeural"

# Calm, unhurried candidates for the audition script.
AUDITION_VOICES = [
    "en-IN-NeerjaNeural",
    "en-IN-NeerjaExpressiveNeural",
    "en-IN-PrabhatNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-US-AriaNeural",
    "en-AU-NatashaNeural",
]

SAMPLE_SENTENCE = (
    "Alright, the report is due at five. Shall I set a reminder for an hour before?"
)


class TTSError(Exception):
    """Synthesis failed; the caller may fall back or stay silent."""


class TTSEngine(Protocol):
    """Anything that turns text into mono float samples."""

    name: str

    async def synth(self, text: str) -> Audio: ...


class EdgeTTSEngine:
    """Microsoft Edge neural voices through the unofficial edge-tts package."""

    name = "edge"

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        rate: str = "-8%",
        pitch: str = "-2Hz",
        volume: str = "+0%",
        timeout: float = 20.0,
    ):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.volume = volume
        self.timeout = timeout

    def update_config(
        self,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
        volume: Optional[str] = None,
    ) -> None:
        """Update voice settings for runtime preset switching."""
        if voice:
            self.voice = voice
        if rate:
            self.rate = rate
        if pitch:
            self.pitch = pitch
        if volume:
            self.volume = volume

    async def synth(self, text: str) -> Audio:
        import edge_tts
        import soundfile as sf

        if not text.strip():
            raise TTSError("nothing to say")

        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            pitch=self.pitch,
            volume=self.volume,
        )

        buffer = bytearray()
        try:
            async with asyncio.timeout(self.timeout):
                async for chunk in communicate.stream():
                    if chunk.get("type") == "audio" and chunk.get("data"):
                        buffer.extend(chunk["data"])
        except TimeoutError as exc:
            raise TTSError("edge-tts timed out") from exc
        except Exception as exc:  # noqa: BLE001 - network/service problems
            raise TTSError(f"edge-tts failed: {exc}") from exc

        if not buffer:
            raise TTSError("edge-tts returned no audio")

        try:
            samples, rate = sf.read(
                io.BytesIO(bytes(buffer)), dtype="float32", always_2d=False
            )
        except Exception as exc:  # noqa: BLE001 - decode problems
            raise TTSError(f"could not decode edge-tts audio: {exc}") from exc

        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        return np.asarray(samples, dtype=np.float32), int(rate)


class PiperEngine:
    """Offline Piper, in-process through the `piper-tts` package.

    A medium voice synthesises a sentence in ~80-180 ms on CPU, so this is the
    fast path: no network handshake, no MP3 decode. The model is loaded once
    and reused; ``warm()`` can pay that ~2.5 s cost at startup instead of on
    the first reply.
    """

    name = "piper"

    def __init__(
        self,
        model: Optional[str] = None,
        config: Optional[str] = None,
        length_scale: float = 1.05,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        speaker: Optional[int] = None,
    ):
        self.model = model or ""
        self.config = config or ""
        self.length_scale = float(length_scale)
        self.noise_scale = float(noise_scale) if noise_scale is not None else None
        self.noise_w = float(noise_w) if noise_w is not None else None
        self.speaker = speaker
        self._voice = None
        self._lock = threading.Lock()
        self._warm_thread: Optional[threading.Thread] = None

    def update_config(
        self,
        model: Optional[str] = None,
        config: Optional[str] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        speaker: Optional[int] = None,
    ) -> None:
        """Update Piper model and parameters for runtime voice switching."""
        with self._lock:
            if model is not None and model != self.model:
                self.model = model
                self._voice = None
            if config is not None:
                self.config = config
            if length_scale is not None:
                self.length_scale = float(length_scale)
            if noise_scale is not None:
                self.noise_scale = float(noise_scale)
            if noise_w is not None:
                self.noise_w = float(noise_w)
            if speaker is not None:
                self.speaker = speaker

    @property
    def voice(self) -> str:
        """Short label for banners: the model's file name without extension."""
        return Path(self.model).stem if self.model else ""

    @property
    def available(self) -> bool:
        return bool(self.model) and Path(self.model).exists()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def _load(self):
        with self._lock:
            if self._voice is not None:
                return self._voice
            if not self.available:
                raise TTSError(
                    "Piper is not configured; set speech.piper_model to a .onnx file"
                )
            try:
                from piper import PiperVoice
            except ImportError as exc:
                raise TTSError(
                    "piper-tts is not installed; run 'pip install piper-tts'"
                ) from exc

            started = time.perf_counter()

            try:
                self._voice = PiperVoice.load(
                    self.model, config_path=self.config or None
                )
            except Exception as exc:  # noqa: BLE001 - bad/missing model
                raise TTSError(f"could not load Piper voice: {exc}") from exc
            _logger.debug(
                "Piper voice %s loaded in %.0f ms",
                self.voice,
                (time.perf_counter() - started) * 1000,
            )
            return self._voice

    def warm(self) -> None:
        """Load the model and run one tiny synthesis in the background.

        Loading alone is not enough: the first inference pays ~1.4 s of ONNX
        warm-up, which would otherwise land on the first reply.
        """
        if self._voice is not None or not self.available:
            return
        if self._warm_thread is not None and self._warm_thread.is_alive():
            return

        def _run() -> None:
            try:
                self._load()
                self._synth("Hello.")
            except TTSError as exc:
                _logger.warning("Piper warm-up failed: %s", exc)

        self._warm_thread = threading.Thread(
            target=_run, name="piper-warmup", daemon=True
        )
        self._warm_thread.start()

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    async def synth(self, text: str) -> Audio:
        if not text.strip():
            raise TTSError("nothing to say")
        return await asyncio.to_thread(self._synth, text)

    def _synth(self, text: str) -> Audio:
        voice = self._load()
        started = time.perf_counter()
        try:
            from piper import SynthesisConfig

            config_kwargs = {
                "speaker_id": self.speaker,
                "length_scale": self.length_scale,
            }
            if self.noise_scale is not None:
                config_kwargs["noise_scale"] = self.noise_scale
            if self.noise_w is not None:
                config_kwargs["noise_w_scale"] = self.noise_w

            config = SynthesisConfig(**config_kwargs)
            chunks = [
                chunk.audio_float_array
                for chunk in voice.synthesize(text, syn_config=config)
            ]
        except Exception as exc:  # noqa: BLE001 - phonemiser/ONNX problems
            raise TTSError(f"Piper failed: {exc}") from exc

        if not chunks:
            raise TTSError("Piper produced no audio")

        samples = np.concatenate([np.asarray(c) for c in chunks])
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        audio = samples.astype(np.float32)
        _logger.debug(
            "Piper synthesised %.2f s in %.0f ms",
            audio.size / voice.config.sample_rate,
            (time.perf_counter() - started) * 1000,
        )
        return audio, int(voice.config.sample_rate)


class FallbackTTS:
    """Try the primary engine, quietly fall back to the local one."""

    def __init__(self, primary: TTSEngine, backup: Optional[TTSEngine] = None):
        self.primary = primary
        self.backup = backup
        self.name = f"{primary.name}+{backup.name if backup else 'none'}"
        self.using_backup = False

    async def synth(self, text: str) -> Audio:
        try:
            audio = await self.primary.synth(text)
            if self.using_backup:
                _logger.info("Primary voice is back")
                self.using_backup = False
            return audio
        except TTSError as exc:
            if not self.backup:
                raise
            _logger.warning("Primary voice failed (%s); trying %s", exc, self.backup.name)
            self.using_backup = True
            return await self.backup.synth(text)

    def warm(self) -> None:
        for engine in (self.primary, self.backup):
            warm = getattr(engine, "warm", None)
            if callable(warm):
                warm()


def get_preset_config(cfg: Optional[dict], preset_name: Optional[str] = None) -> dict:
    """Extract preset configuration dict from root config."""
    full = cfg or {}
    voice_cfg = full.get("voice") if isinstance(full.get("voice"), dict) else {}
    presets = voice_cfg.get("presets", {})
    if not preset_name:
        preset_name = voice_cfg.get("default_preset", "calm")
    if isinstance(presets, dict):
        preset = presets.get(preset_name)
        if isinstance(preset, dict):
            return dict(preset)
    return {}


def build_engine(
    cfg: Optional[dict] = None, preset: Optional[str] = None
) -> Optional[TTSEngine]:
    """Build the engine described by `speech` config or preset, or None when disabled.

    ``auto`` prefers the local Piper voice (fast, offline) and keeps edge-tts
    as the backup. ``edge`` forces the cloud voice; ``piper`` forces local and
    degrades to edge-tts with a warning if the model file is missing.
    """
    full = cfg or {}
    speech = full.get("speech") if isinstance(full.get("speech"), dict) else full

    if not speech.get("enabled", True):
        return None

    preset_data = get_preset_config(full, preset)

    # Edge configuration (preset overrides if present)
    edge_voice = preset_data.get("edge_voice") or speech.get("voice", DEFAULT_VOICE)
    edge_rate = str(preset_data.get("edge_rate") or speech.get("rate", "-8%"))
    edge_pitch = str(preset_data.get("edge_pitch") or speech.get("pitch", "-2Hz"))
    edge_volume = str(preset_data.get("edge_volume") or speech.get("volume", "+0%"))

    # Piper configuration (preset overrides if present)
    piper_model = preset_data.get("piper_model") or speech.get("piper_model", "")
    piper_config = preset_data.get("piper_config") or speech.get("piper_config", "")
    length_scale = float(
        preset_data.get("length_scale") or speech.get("piper_length_scale", 1.05)
    )
    noise_scale = preset_data.get("noise_scale")
    noise_w = preset_data.get("noise_w")

    backend = str(speech.get("engine", "auto")).lower()
    edge = EdgeTTSEngine(
        voice=edge_voice,
        rate=edge_rate,
        pitch=edge_pitch,
        volume=edge_volume,
    )
    piper = PiperEngine(
        model=piper_model,
        config=piper_config,
        length_scale=length_scale,
        noise_scale=float(noise_scale) if noise_scale is not None else None,
        noise_w=float(noise_w) if noise_w is not None else None,
    )
    local = piper if piper.available else None

    if backend == "edge":
        return edge
    if backend == "piper":
        if local is None:
            _logger.warning(
                "speech.engine is piper but no model was found at %r; using edge-tts",
                piper.model or "(unset)",
            )
            return edge
        return FallbackTTS(local, edge)
    return FallbackTTS(local, edge) if local else edge
