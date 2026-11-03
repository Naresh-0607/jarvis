"""Hold-to-talk voice input: global hotkey, recording and transcription."""

from __future__ import annotations

import ctypes
import logging
import queue
import threading
import time
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np

from jarvis import audio as audio_utils
from jarvis.stt import Transcriber, TranscriptionError, build_transcriber

_logger = logging.getLogger("jarvis.voice")

DEFAULT_HOTKEY = "<ctrl>+<alt>+<space>"

_MODIFIER_ALIASES = {
    "ctrl": "ctrl",
    "ctrl_l": "ctrl",
    "ctrl_r": "ctrl",
    "alt": "alt",
    "alt_l": "alt",
    "alt_r": "alt",
    "alt_gr": "alt",
    "shift": "shift",
    "shift_l": "shift",
    "shift_r": "shift",
    "cmd": "cmd",
    "cmd_l": "cmd",
    "cmd_r": "cmd",
}


def parse_hotkey(value: str) -> Tuple[Set[str], str]:
    """Split "<ctrl>+<alt>+<space>" into (modifiers, main key)."""
    tokens = [
        token.strip().strip("<>").lower()
        for token in (value or "").split("+")
        if token.strip()
    ]
    if not tokens:
        tokens = ["ctrl", "alt", "space"]

    modifiers = {_MODIFIER_ALIASES[t] for t in tokens[:-1] if t in _MODIFIER_ALIASES}
    main = tokens[-1]
    return modifiers, main


def key_name(key) -> str:
    """Normalise a pynput key into a short name."""
    name = getattr(key, "name", None)
    if name:
        return _MODIFIER_ALIASES.get(name, name)
    char = getattr(key, "char", None)
    if char:
        return char.lower()
    return ""


class HoldToTalk:
    """Records while the hotkey is held and transcribes on release."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        on_transcript: Optional[Callable[[str], None]] = None,
        status: Optional[Callable[[str], None]] = None,
        transcriber: Optional[Transcriber] = None,
        on_record_start: Optional[Callable[[], None]] = None,
        on_level: Optional[Callable[[float], None]] = None,
        on_state: Optional[Callable[[str], None]] = None,
    ):
        full = config or {}
        self.voice_cfg = full.get("voice") or full
        self.hotkey = self.voice_cfg.get("hotkey") or full.get("hotkey") or DEFAULT_HOTKEY
        self.device = self.voice_cfg.get("device") or None
        self.min_seconds = float(self.voice_cfg.get("min_seconds", 0.4))
        self.max_seconds = float(self.voice_cfg.get("max_seconds", 60))
        self.trim = bool(self.voice_cfg.get("trim_silence", True))
        self.threshold = float(self.voice_cfg.get("silence_threshold", 0.01))
        self.set_priority = bool(self.voice_cfg.get("below_normal_priority", True))

        self.transcriber = transcriber or build_transcriber(self.voice_cfg)
        self.status = status or (lambda message: print(f"[voice] {message}"))
        self.on_transcript = on_transcript or (lambda text: None)
        self.on_record_start = on_record_start or (lambda: None)
        self.on_level = on_level
        self.on_state = on_state

        self.required, self.main_key = parse_hotkey(self.hotkey)
        self.required = set(self.required) | {self.main_key}

        self._pressed: Set[str] = set()
        self._recording = False
        self._listener = None
        self._stream = None
        self._frames = []
        self._rate = audio_utils.TARGET_RATE
        self._lock = threading.Lock()
        self._started_at = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.set_priority:
            self._lower_priority()

        from pynput import keyboard

        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()
        self.status(f"ready - hold {self.hotkey} and speak")

    def stop(self) -> None:
        if self._recording:
            self._stop_stream()
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        close = getattr(self.transcriber, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _lower_priority() -> None:
        try:
            kernel32 = ctypes.windll.kernel32
            # BELOW_NORMAL_PRIORITY_CLASS
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000)
        except Exception:  # noqa: BLE001 - not critical
            _logger.debug("Could not lower process priority")

    # ------------------------------------------------------------------
    # Hotkey handling
    # ------------------------------------------------------------------
    def _on_press(self, key) -> None:
        name = key_name(key)
        if not name:
            return
        self._pressed.add(name)
        if not self._recording and self.required.issubset(self._pressed):
            self._start_stream()

    def _on_release(self, key) -> None:
        name = key_name(key)
        if self._recording and name in self.required:
            self._stop_stream()
        self._pressed.discard(name)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _pick_rate(self) -> int:
        import sounddevice as sd

        for rate in (audio_utils.TARGET_RATE, 48000, 44100):
            try:
                sd.check_input_settings(
                    device=self.device, samplerate=rate, channels=1, dtype="float32"
                )
                return rate
            except Exception:  # noqa: BLE001 - try the next rate
                continue
        return audio_utils.TARGET_RATE

    def _start_stream(self) -> None:
        import sounddevice as sd

        with self._lock:
            if self._recording:
                return
            # Barge-in: stop any speech before the mic opens.
            try:
                self.on_record_start()
            except Exception:  # noqa: BLE001 - never block recording
                _logger.debug("on_record_start failed", exc_info=True)

            self._frames = []
            self._rate = self._pick_rate()
            try:
                self._stream = sd.InputStream(
                    samplerate=self._rate,
                    channels=1,
                    dtype="float32",
                    device=self.device,
                    callback=self._audio_callback,
                )
                self._stream.start()
            except Exception as exc:  # noqa: BLE001 - mic problems are reported
                self._stream = None
                self._recording = False
                self.status(f"cannot record: {exc}")
                return

            self._recording = True
            self._started_at = time.perf_counter()
            self.status("listening...")
            if self.on_state:
                self.on_state("listening")

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            _logger.debug("Audio status: %s", status)
        with self._lock:
            self._frames.append(indata.copy())
        if self.on_level:
            try:
                self.on_level(audio_utils.rms(indata))
            except Exception:  # noqa: BLE001
                pass
        if time.perf_counter() - self._started_at > self.max_seconds:
            # Guard against a stuck key.
            threading.Thread(target=self._stop_stream, daemon=True).start()

    def _stop_stream(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            stream, self._stream = self._stream, None
            frames = self._frames
            self._frames = []
            rate = self._rate

        if self.on_level:
            self.on_level(0.0)
        if self.on_state:
            self.on_state("thinking")

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                _logger.debug("Error closing input stream", exc_info=True)

        if not frames:
            self.status("heard nothing")
            if self.on_state:
                self.on_state("idle")
            return

        samples = np.concatenate([frame.reshape(-1) for frame in frames]).astype(np.float32)
        threading.Thread(
            target=self._transcribe, args=(samples, rate), daemon=True
        ).start()

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------
    def _transcribe(self, samples: np.ndarray, rate: int) -> None:
        duration = samples.size / float(rate) if rate else 0.0
        if duration < self.min_seconds:
            self.status(f"too short ({duration:.1f}s), ignored")
            if self.on_state:
                self.on_state("idle")
            return

        prepared, out_rate = audio_utils.prepare(
            samples, rate, threshold=self.threshold, trim=self.trim
        )
        if prepared.size == 0:
            self.status(
                f"only silence (level {audio_utils.rms(samples):.4f}), ignored"
            )
            if self.on_state:
                self.on_state("idle")
            return

        wav = audio_utils.to_wav_bytes(prepared, out_rate)
        started = time.perf_counter()
        try:
            text = self.transcriber.transcribe(wav)
        except TranscriptionError as exc:
            self.status(f"could not hear you: {exc}")
            if self.on_state:
                self.on_state("idle")
            return
        except Exception as exc:  # noqa: BLE001 - never kill the listener thread
            _logger.exception("Unexpected transcription error")
            self.status(f"could not hear you: {exc}")
            if self.on_state:
                self.on_state("idle")
            return

        elapsed = time.perf_counter() - started
        _logger.info(
            "Speech: %.1fs recorded, %.1fs sent, %.0f ms to transcribe",
            duration,
            prepared.size / float(out_rate),
            elapsed * 1000,
        )

        if not text:
            self.status("no words caught")
            if self.on_state:
                self.on_state("idle")
            return

        self.status(f"{duration:.1f}s recorded, transcript in {elapsed * 1000:.0f} ms")
        self.on_transcript(text)


class TranscriptQueue:
    """Thread-safe hand-out between the hotkey thread and the chat loop."""

    def __init__(self):
        self._queue: "queue.Queue[str]" = queue.Queue()

    def put(self, text: str) -> None:
        self._queue.put(text)

    def get_nowait(self) -> Optional[str]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None
