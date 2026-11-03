"""Streaming speech output: clean text, sentence splitting, queued playback."""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import re
import threading
import time
from datetime import date
from typing import Callable, List, Optional, Tuple

import numpy as np

from jarvis.tts import (
    DEFAULT_VOICE,
    EdgeTTSEngine,
    FallbackTTS,
    PiperEngine,
    TTSError,
    TTSEngine,
    build_engine,
    get_preset_config,
)
from jarvis.voice_fx import VoiceFXChain, VoiceFXConfig

_logger = logging.getLogger("jarvis.speech")

FILLERS = [
    "Let me take a look.",
    "One moment.",
    "Checking that now.",
    "Give me a second.",
]

CODE_NOTICE = "I've put the code on your screen."

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "vs", "etc", "approx", "no",
    "st", "jr", "sr", "fig", "eg", "ie", "e.g", "i.e",
}

_EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF" "]+"
)
_SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s)")


def strip_code_blocks(text: str) -> Tuple[str, bool]:
    """Remove fenced code before sentence splitting.

    Called on the *whole* buffer, because a fence can be split across streamed
    chunks and must never be read aloud.
    """
    had_code = "```" in text
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"```.*$", " ", text, flags=re.DOTALL)  # unterminated tail
    return text, had_code


def clean_for_speech(text: str) -> Tuple[str, bool]:
    """Strip markdown, code and links. Returns (text, had_code_block)."""
    had_code = "```" in text

    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "the link on screen", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("__", "")
    text = text.replace("*", "").replace("_", "")
    text = text.replace("|", " ")
    text = re.sub(r"<tool_call>[\s\S]*?</tool_call>", " ", text)
    text = re.sub(r"<function=[a-zA-Z0-9_\-]+>[\s\S]*?</function>", " ", text)
    text = re.sub(r"</?[a-zA-Z0-9_\-]+(=[^>]*)?>", " ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, had_code


def normalise_numbers(text: str) -> str:
    """Make dates, times, money and percentages read naturally."""

    def _iso_date(match: re.Match) -> str:
        try:
            parsed = date.fromisoformat(match.group(0))
        except ValueError:
            return match.group(0)
        return parsed.strftime("%d %B %Y")

    def _time(match: re.Match) -> str:
        hour, minute = int(match.group(1)), int(match.group(2))
        suffix = "AM" if hour < 12 else "PM"
        hour12 = hour % 12 or 12
        return f"{hour12}:{minute:02d} {suffix}" if minute else f"{hour12} {suffix}"

    def _money(unit: str):
        def _replace(match: re.Match) -> str:
            return f"{match.group(1).replace(',', '')} {unit}"

        return _replace

    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", _iso_date, text)
    text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?!\s*[AaPp]\.?[Mm])", _time, text)
    text = re.sub(r"₹\s*(\d[\d,]*)", _money("rupees"), text)
    text = re.sub(r"\$\s*(\d[\d,]*)", _money("dollars"), text)
    text = re.sub(r"(\d[\d,]*)\s*%", _money("percent"), text)
    text = re.sub(r"\bRs\.?\s*(\d[\d,]*)", _money("rupees"), text)
    return text


def split_sentences(buffer: str, max_chars: int = 180) -> Tuple[List[str], str]:
    """Pull complete sentences off the front of a streamed buffer.

    A sentence needs its terminator followed by whitespace, so decimals like
    "3.5" and half-received tokens are never cut in half. Very long runs with
    no punctuation are split on a word boundary.
    """
    sentences: List[str] = []
    remaining = buffer

    while True:
        boundary = _find_boundary(remaining)
        if boundary is None:
            break
        end = boundary
        part = remaining[:end].strip()
        if part:
            sentences.append(part)
        remaining = remaining[end:].lstrip()

    while len(remaining) > max_chars:
        cut = remaining.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        piece = remaining[:cut].strip()
        if piece:
            sentences.append(piece)
        remaining = remaining[cut:].lstrip()

    return sentences, remaining


def _find_boundary(text: str) -> Optional[int]:
    """Index just past a terminator (and the whitespace that follows)."""
    for match in _SENTENCE_END_RE.finditer(text):
        before = text[: match.start()]
        word = before.split()[-1].lower().rstrip(".") if before.split() else ""
        if word in _ABBREVIATIONS:
            continue
        return match.end() + len(text[match.end():]) - len(text[match.end():].lstrip())

    newline = text.find("\n")
    if newline != -1:
        return newline + 1
    return None


def _takes_stop_event(player: Callable[..., None]) -> bool:
    """True when the player accepts a third ``stop_event`` argument."""
    try:
        parameters = inspect.signature(player).parameters
    except (TypeError, ValueError):
        return True

    for parameter in parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return True
    positional = [
        p
        for p in parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 3


def play_audio(
    samples: np.ndarray,
    rate: int,
    stop_event: Optional[threading.Event] = None,
    on_level: Optional[Callable[[float], None]] = None,
) -> None:
    """Blocking playback with a cooperative stop."""
    import sounddevice as sd

    if samples.size == 0:
        return

    if stop_event is not None and stop_event.is_set():
        return

    sd.play(samples, rate)
    stream = sd.get_stream()
    start_t = time.perf_counter()
    chunk_size = max(1, int(rate * 0.05))

    while stream is not None and stream.active:
        if stop_event is not None and stop_event.is_set():
            sd.stop()
            if on_level:
                on_level(0.0)
            return
        if on_level:
            elapsed = time.perf_counter() - start_t
            idx = int(elapsed * rate)
            if idx < samples.size:
                chunk = samples[idx : min(samples.size, idx + chunk_size)]
                level = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
                on_level(level)
            else:
                on_level(0.0)
        time.sleep(0.02)
    if on_level:
        on_level(0.0)
class SpeechOutput:
    """Consumes streamed text and speaks it sentence by sentence."""

    def __init__(
        self,
        config: Optional[dict] = None,
        engine: Optional[TTSEngine] = None,
        player: Optional[Callable[..., None]] = None,
        log: Optional[Callable[[str], None]] = None,
        on_level: Optional[Callable[[float], None]] = None,
        on_state: Optional[Callable[[str], None]] = None,
        preset: Optional[str] = None,
    ):
        full = config or {}
        speech_cfg = full.get("speech") if isinstance(full.get("speech"), dict) else full
        self._config = full

        voice_cfg = full.get("voice") if isinstance(full.get("voice"), dict) else {}
        self.preset_name = preset or voice_cfg.get("default_preset", "calm")
        preset_cfg = get_preset_config(full, self.preset_name)

        self.engine = (
            engine
            if engine is not None
            else build_engine(full, preset=self.preset_name)
        )
        self.enabled = self.engine is not None
        self.max_chars = int(speech_cfg.get("max_sentence_chars", 180))
        self.filler_enabled = bool(speech_cfg.get("filler", True))
        self.player = player or play_audio
        self.log = log or (lambda message: print(f"[voice out] {message}"))
        self.on_level = on_level
        self.on_state = on_state
        self._player_takes_event = _takes_stop_event(self.player)

        # Voice FX chain for post-synthesis audio processing
        fx_dict = preset_cfg.get("fx")
        self.fx: Optional[VoiceFXChain] = VoiceFXChain(VoiceFXConfig.from_dict(fx_dict))

        self._muted = False
        self._buffer = ""
        self._held = ""
        self._queue: Optional[asyncio.Queue] = None
        self._worker: Optional[asyncio.Task] = None
        self._stop_event = threading.Event()
        self._recent_fillers: List[str] = []
        self._code_notice_sent = False
        self._turn_started: Optional[float] = None
        self._first_audio_ms: Optional[float] = None

    def set_preset(self, preset_name: str, config: Optional[dict] = None) -> None:
        """Switch voice preset mid-session.

        Stops active speech, resets FX tail, reconfigures Piper/Edge engines,
        and starts warming the model in the background.
        """
        self.stop()
        if config is not None:
            self._config = config
        self.preset_name = preset_name
        preset_cfg = get_preset_config(self._config, preset_name)

        # Update FX chain
        fx_dict = preset_cfg.get("fx")
        self.fx = VoiceFXChain(VoiceFXConfig.from_dict(fx_dict))
        self.fx.reset()

        # Update engines in-place
        piper_model = preset_cfg.get("piper_model", "")
        piper_config = preset_cfg.get("piper_config", "")
        length_scale = float(preset_cfg.get("length_scale", 1.05))
        noise_scale = preset_cfg.get("noise_scale")
        noise_w = preset_cfg.get("noise_w")

        edge_voice = preset_cfg.get("edge_voice", DEFAULT_VOICE)
        edge_rate = str(preset_cfg.get("edge_rate", "-8%"))
        edge_pitch = str(preset_cfg.get("edge_pitch", "-2Hz"))
        edge_volume = str(preset_cfg.get("edge_volume", "+0%"))

        def _collect_engines(eng: Any) -> List[Any]:
            if isinstance(eng, FallbackTTS):
                return _collect_engines(eng.primary) + _collect_engines(eng.backup)
            elif eng is not None:
                return [eng]
            return []

        for eng in _collect_engines(self.engine):
            if eng is None:
                continue
            if isinstance(eng, PiperEngine):
                eng.update_config(
                    model=piper_model,
                    config=piper_config,
                    length_scale=length_scale,
                    noise_scale=float(noise_scale) if noise_scale is not None else None,
                    noise_w=float(noise_w) if noise_w is not None else None,
                )
                eng.warm()
            elif isinstance(eng, EdgeTTSEngine):
                eng.update_config(
                    voice=edge_voice,
                    rate=edge_rate,
                    pitch=edge_pitch,
                    volume=edge_volume,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def first_audio_ms(self) -> Optional[float]:
        return self._first_audio_ms

    def mute(self) -> None:
        self._muted = True
        self.stop()

    def unmute(self) -> None:
        self._muted = False

    def warm(self) -> None:
        """Start loading a local voice model before the first reply needs it."""
        warm = getattr(self.engine, "warm", None)
        if callable(warm):
            warm()

    def start_turn(self, started_at: Optional[float] = None) -> None:
        """Begin buffering text for one assistant reply."""
        self._buffer = ""
        self._held = ""
        self._code_notice_sent = False
        self._first_audio_ms = None
        self._turn_started = started_at if started_at is not None else time.perf_counter()
        # A fresh event per turn, so a late barge-in cannot silence the next one.
        self._stop_event = threading.Event()

        if not self.enabled:
            return

        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._run(self._queue))

    def feed(self, text: str) -> None:
        """Add streamed text; complete sentences are queued immediately."""
        if not self.enabled or self._muted or self._queue is None or not text:
            return

        self._buffer += text

        # Half a code fence is in the buffer: speak the prose before it now and
        # wait for the closing fence before touching anything after it.
        if self._buffer.count("```") % 2 == 1:
            opening = self._buffer.find("```")
            head, self._buffer = self._buffer[:opening], self._buffer[opening:]
            sentences, leftover = split_sentences(head, self.max_chars)
            for sentence in sentences:
                self._enqueue(sentence)
            if leftover:
                self._held = f"{self._held} {leftover}".strip()
            return

        self._buffer = self._join(self._held, self._buffer)
        self._held = ""

        self._buffer, had_code = strip_code_blocks(self._buffer)
        if had_code:
            self._note_code()

        sentences, self._buffer = split_sentences(self._buffer, self.max_chars)
        for sentence in sentences:
            self._enqueue(sentence)

    async def flush(self, timeout: float = 40.0) -> None:
        """Speak whatever is left and wait for playback to finish."""
        if not self.enabled or self._queue is None:
            return

        self._buffer = self._join(self._held, self._buffer)
        self._held = ""
        self._buffer, had_code = strip_code_blocks(self._buffer)
        if had_code:
            self._note_code()

        if not self._muted and self._buffer.strip():
            self._enqueue(self._buffer)
        self._buffer = ""

        queue = self._queue
        await queue.put(None)

        worker = self._worker
        self._worker = None
        if worker is not None:
            try:
                await asyncio.wait_for(worker, timeout=timeout)
            except asyncio.TimeoutError:
                _logger.warning("Speech playback timed out")
                worker.cancel()
            except asyncio.CancelledError:
                pass
        if self._queue is queue:
            self._queue = None

    def stop(self) -> None:
        """Barge-in: drop queued speech and stop the current clip."""
        self._stop_event.set()
        self._buffer = ""
        self._held = ""
        self._queue = None
        if self.fx is not None:
            self.fx.reset()
        if self.on_level:
            self.on_level(0.0)
        if self.on_state:
            self.on_state("idle")

        worker, self._worker = self._worker, None
        if worker is not None and not worker.done():
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # Called from the hotkey thread; the worker notices the
                # detached queue and retires on its next poll.
                pass
            else:
                worker.cancel()

        try:
            import sounddevice as sd

            sd.stop()
        except Exception:  # noqa: BLE001 - audio may not be initialised
            _logger.debug("Could not stop playback", exc_info=True)

    def maybe_filler(self) -> None:
        """Say a short holding line while a slow tool runs."""
        if not self.enabled or self._muted or not self.filler_enabled:
            return
        if self._queue is None:
            return

        choices = [line for line in FILLERS if line not in self._recent_fillers]
        if not choices:
            choices = list(FILLERS)

        line = random.choice(choices)
        self._recent_fillers = (self._recent_fillers + [line])[-3:]
        self._enqueue(line)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _join(first: str, second: str) -> str:
        return f"{first} {second}".strip() if first else second

    def _note_code(self) -> None:
        if self._code_notice_sent or self._queue is None:
            return
        self._code_notice_sent = True
        self._queue.put_nowait(CODE_NOTICE)

    def _enqueue(self, text: str) -> None:
        if self._queue is None:
            return

        cleaned, _had_code = clean_for_speech(text)
        cleaned = normalise_numbers(cleaned)

        if cleaned:
            self._queue.put_nowait(cleaned)

    async def _run(self, queue: "asyncio.Queue") -> None:
        try:
            while True:
                try:
                    text = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # A barge-in detaches the queue; let this worker retire.
                    if self._queue is not queue:
                        return
                    continue

                if text is None or self._queue is not queue:
                    return

                try:
                    samples, rate = await self.engine.synth(text)
                except TTSError as exc:
                    self.log(f"could not speak: {exc}")
                    return
                except Exception as exc:  # noqa: BLE001 - a custom engine may misbehave
                    _logger.exception("Unexpected TTS error")
                    self.log("could not speak: " + str(exc))
                    return

                if self._queue is not queue:
                    return

                # Process samples through FX chain if configured
                if self.fx is not None:
                    try:
                        samples, rate = self.fx.process(samples, rate)
                    except Exception as exc:  # noqa: BLE001
                        _logger.exception("Voice FX processing error: %s", exc)

                await asyncio.to_thread(self._play, samples, rate, self._stop_event)
        finally:
            if self.on_state:
                self.on_state("idle")
            if self.on_level:
                self.on_level(0.0)

    def _play(
        self, samples: np.ndarray, rate: int, stop_event: Optional[threading.Event]
    ) -> None:
        if self._first_audio_ms is None and self._turn_started is not None:
            self._first_audio_ms = (time.perf_counter() - self._turn_started) * 1000
            self.log(f"first audio after {self._first_audio_ms:.0f} ms")

        if self.on_state:
            self.on_state("speaking")

        try:
            params = inspect.signature(self.player).parameters
            if "on_level" in params or any(
                p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
                for p in params.values()
            ):
                self.player(samples, rate, stop_event, on_level=self.on_level)
            elif self._player_takes_event:
                self.player(samples, rate, stop_event)
            else:
                self.player(samples, rate)
        except Exception as exc:  # noqa: BLE001 - speakers can fail
            self.log(f"playback failed: {exc}")
        finally:
            if self.on_level:
                self.on_level(0.0)
