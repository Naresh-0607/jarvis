"""Full-screen terminal interface (TUI) for Jarvis assistant (Phase 9-UI).

Provides an in-terminal, full-screen dashboard powered by Textual with a glowing
ASCII orb, responsive layout, startup online/local mode selection, status indicator,
and quiet terminal logging.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static

from jarvis.chat import ChatSession
from jarvis.db import get_setting, set_setting
from jarvis.speech import play_audio
from jarvis.tts import build_engine, get_preset_config
from jarvis.voice_fx import VoiceFXChain, VoiceFXConfig

_logger = logging.getLogger("jarvis.tui")

# ==============================================================================
# Configurable constants (Colors, Ramp, FPS, Orb Geometry)
# ==============================================================================
DEFAULT_ACCENT_COLOR = "cyan"
DEFAULT_USER_COLOR = "cyan"
DEFAULT_JARVIS_COLOR = "#38bdf8"
DEFAULT_IDLE_COLOR = "#0284c7"
DEFAULT_THINKING_COLOR = "#fbbf24"

DEFAULT_ASCII_RAMP = " .:-=+*#%@"
DEFAULT_FPS = 28
IDLE_FPS = 12

DEFAULT_ORB_BASE_RADIUS = 7.0
DEFAULT_ORB_GAIN = 6.0


# ==============================================================================
# Pure Logic: State Machine, Level Smoother, Ripple Manager, Connectivity, Orb
# ==============================================================================
class UIStateMachine:
    """Manages UI audio/interaction states with valid transitions."""

    VALID_STATES = ("idle", "listening", "thinking", "speaking")

    def __init__(self, initial_state: str = "idle"):
        self._state = initial_state if initial_state in self.VALID_STATES else "idle"

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_idle(self) -> bool:
        return self._state == "idle"

    @property
    def is_listening(self) -> bool:
        return self._state == "listening"

    @property
    def is_thinking(self) -> bool:
        return self._state == "thinking"

    @property
    def is_speaking(self) -> bool:
        return self._state == "speaking"

    def transition(self, new_state: str) -> bool:
        """Attempt to transition to new_state. Returns True if accepted."""
        if new_state not in self.VALID_STATES:
            return False

        # Any state can return to idle
        if new_state == "idle":
            self._state = "idle"
            return True

        # Barge-in: speaking can transition directly to listening
        if self._state == "speaking" and new_state == "listening":
            self._state = "listening"
            return True

        self._state = new_state
        return True


class LevelSmoother:
    """Filters audio amplitude levels with fast attack and smooth decay."""

    def __init__(self, attack: float = 0.50, decay: float = 0.16):
        self.attack = attack
        self.decay = decay
        self._current: float = 0.0

    @property
    def current(self) -> float:
        return self._current

    def update(self, raw_level: float) -> float:
        raw_level = max(0.0, float(raw_level))
        if raw_level > self._current:
            self._current += (raw_level - self._current) * self.attack
        else:
            self._current -= self._current * self.decay
        if self._current < 1e-4:
            self._current = 0.0
        return self._current

    def reset(self) -> None:
        self._current = 0.0


class RippleManager:
    """Manages expanding ripple rings spawned by audio amplitude peaks."""

    def __init__(self, max_ripples: int = 3, throttle_s: float = 0.28):
        self.max_ripples = max_ripples
        self.throttle_s = throttle_s
        self.ripples: List[Dict[str, float]] = []
        self._last_spawn_time = 0.0

    def trigger(self, base_radius: float, now: Optional[float] = None) -> bool:
        """Trigger a ripple if within throttle limit and max ripples."""
        current_time = time.monotonic() if now is None else now
        if current_time - self._last_spawn_time < self.throttle_s:
            return False
        if len(self.ripples) >= self.max_ripples:
            return False

        self._last_spawn_time = current_time
        self.ripples.append({
            "radius": max(1.5, base_radius * 0.8),
            "max_radius": base_radius * 2.4,
            "speed": 8.0,
            "age": 0.0,
        })
        return True

    def update(self, dt: float) -> None:
        """Advance ripple positions and ages, pruning expired ripples."""
        survivors = []
        for r in self.ripples:
            r["age"] += dt
            r["radius"] += r["speed"] * dt
            if r["age"] < 1.0 and r["radius"] < r["max_radius"]:
                survivors.append(r)
        self.ripples = survivors

    def clear(self) -> None:
        self.ripples.clear()

    @property
    def count(self) -> int:
        return len(self.ripples)


def probe_internet(
    hosts: Sequence[Tuple[str, int]] = (("1.1.1.1", 53), ("8.8.8.8", 53)),
    timeout: float = 2.0,
) -> bool:
    """Fast socket probe to check internet connectivity."""
    for host, port in hosts:
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
            sock.close()
            return True
        except OSError:
            continue
    return False


def determine_connectivity(
    is_online_reachable: bool, local_available: bool, has_error: bool = False
) -> Tuple[str, str]:
    """Pure logic to compute status dot color and label."""
    if has_error:
        return "error", "error"
    if is_online_reachable:
        return "online", "online"
    if local_available:
        return "local", "local"
    return "error", "error"


def generate_orb_grid(
    width: int,
    height: int,
    radius: float,
    level: float = 0.0,
    gain: float = DEFAULT_ORB_GAIN,
    ramp: str = DEFAULT_ASCII_RAMP,
    state: str = "idle",
    phase: float = 0.0,
    ripples: Optional[Sequence[Dict[str, float]]] = None,
    aspect_ratio: float = 0.5,
) -> List[str]:
    """Render a round glowing ASCII orb with density falloff and ripples.

    Guarantees that every character in the returned strings is strictly within
    ASCII code points 32-126.
    """
    if width <= 0 or height <= 0:
        return []

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    # Dynamic radius modulated by audio level and breathing / shimmer
    if state == "idle":
        effective_radius = max(1.5, radius + 0.45 * math.sin(phase))
    elif state == "listening":
        effective_radius = max(1.5, radius + min(1.0, max(0.0, level)) * gain)
    elif state == "speaking":
        effective_radius = max(1.5, radius + min(1.0, max(0.0, level)) * gain)
    elif state == "thinking":
        effective_radius = max(1.5, radius + 0.85 * math.sin(phase * 2.5))
    else:
        effective_radius = max(1.5, radius)

    ramp_len = len(ramp)
    lines: List[str] = []

    for y in range(height):
        dy = y - cy
        row_chars: List[str] = []
        for x in range(width):
            dx = (x - cx) * aspect_ratio
            dist = math.hypot(dx, dy)

            # Check for peak ripples
            ripple_ch: Optional[str] = None
            if ripples:
                for rip in ripples:
                    rip_r = rip.get("radius", 0.0)
                    if abs(dist - rip_r) < 0.65:
                        age = rip.get("age", 0.0)
                        ripple_ch = "o" if age < 0.35 else (":" if age < 0.70 else ".")
                        break

            if dist <= effective_radius:
                # 3D spherical raycast volume
                norm_d = dist / effective_radius
                sphere_z = math.sqrt(max(0.0, 1.0 - norm_d * norm_d))
                plasma = 0.08 * math.sin(dx * 1.4 + phase * 2.0) + 0.06 * math.cos(dy * 1.8 - phase * 1.5)

                if state == "thinking":
                    angle = math.atan2(dy, dx)
                    plasma += 0.20 * math.sin(angle * 3.0 + phase * 3.5)
                elif state == "speaking":
                    plasma += 0.12 * math.sin(phase * 4.0 + dist * 1.5)
                elif state == "listening":
                    plasma += 0.10 * math.sin(phase * 3.0 + dist)
                elif state == "idle":
                    plasma += 0.04 * math.sin(phase)

                ratio = max(0.0, min(1.0, 0.20 * (1.0 - norm_d) + 0.80 * sphere_z + plasma))
                idx = int(ratio * (ramp_len - 1))
                idx = max(1, min(ramp_len - 1, idx))
                orb_ch = ramp[idx]

                # Outer edge blending with ripples
                if ripple_ch and dist > effective_radius * 0.75:
                    ch = ripple_ch
                else:
                    ch = orb_ch
            elif dist <= effective_radius + 1.3:
                # Soft atmospheric corona/halo
                halo = 1.0 - (dist - effective_radius) / 1.3
                if ripple_ch:
                    ch = ripple_ch
                elif halo > 0.55:
                    ch = ":"
                elif halo > 0.15:
                    ch = "."
                else:
                    ch = " "
            else:
                ch = ripple_ch if ripple_ch else " "

            # Safety check: enforce strict ASCII 32-126
            if not (32 <= ord(ch) <= 126):
                ch = " "

            row_chars.append(ch)
        lines.append("".join(row_chars))

    return lines


def render_ascii_waveform(
    buffer: Sequence[float],
    width: int,
    height: int,
    ramp: str = DEFAULT_ASCII_RAMP,
    center_char: str = "-",
) -> List[str]:
    """Retained for test backward compatibility."""
    if width <= 0 or height <= 0:
        return []

    buf = list(buffer)
    if len(buf) < width:
        buf = [0.0] * (width - len(buf)) + buf
    elif len(buf) > width:
        buf = buf[-width:]

    cy = height // 2
    lines: List[str] = []

    for r in range(height):
        dist = abs(r - cy)
        if dist == 0:
            lines.append(center_char * width)
        else:
            row_chars = []
            for x in range(width):
                amp = max(0.0, min(1.0, float(buf[x])))
                bar_h = int(amp * cy)
                if dist <= bar_h:
                    ratio = 1.0 - (dist / (bar_h + 1))
                    idx = min(len(ramp) - 1, max(1, int(ratio * (len(ramp) - 1))))
                    ch = ramp[idx]
                else:
                    ch = " "
                row_chars.append(ch)
            lines.append("".join(row_chars))

    return lines


# ==============================================================================
# Quiet Logging & Stdout Redirection
# ==============================================================================
class LogFileWriter:
    """Redirects print() and stderr calls directly to logs/jarvis.log."""

    def __init__(self, log_path: Path):
        self._file = log_path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        with self._lock:
            self._file.write(s)
            self._file.flush()
        return len(s)

    def flush(self) -> None:
        with self._lock:
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


# ==============================================================================
# TUI Widgets
# ==============================================================================
class NotificationPanel(Static):
    """Top-right reserved notification space stub (Phase 9).

    Hidden by default. Ready for later notifications via push().
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.display = False

    def push(self, title: str, body: str, level: str = "info") -> None:
        self.update(f"[{title}] {body}")
        self.display = True


class StatusIndicator(Static):
    """Bottom-right status indicator: (o) online / <voice>, (o) local / <voice>, (o) error."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._status = "online"
        self._voice = "calm"
        self._label = "online"
        self._reason = ""

    def on_mount(self) -> None:
        self.set_status("online", "online", voice=self._voice)

    def set_status(
        self, status: str, label: str, reason: str = "", voice: Optional[str] = None
    ) -> None:
        self._status = status
        self._label = label
        self._reason = reason
        if voice is not None:
            self._voice = voice

        suffix = f" / {self._voice}" if self._voice else ""
        if status == "online":
            self.update(Text(f"(o) online{suffix}", style="bold green"))
        elif status == "local":
            self.update(Text(f"(o) local{suffix}", style="bold yellow"))
        else:
            text = f"(o) error: {reason}" if reason else "(o) error"
            self.update(Text(text, style="bold red"))


class OrbWidget(Static):
    """Center dominant ASCII glowing orb widget."""

    def __init__(
        self,
        ramp: str = DEFAULT_ASCII_RAMP,
        fps: int = DEFAULT_FPS,
        base_radius: float = DEFAULT_ORB_BASE_RADIUS,
        gain: float = DEFAULT_ORB_GAIN,
        colors: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ramp = ramp
        self.fps = fps
        self.base_radius = base_radius
        self.gain = gain
        self.orb_colors = colors or {
            "listening": "#00f0ff",
            "speaking": "#38bdf8",
            "thinking": "#fbbf24",
            "idle": "#0284c7",
        }

        self.state_machine = UIStateMachine("idle")
        self.smoother = LevelSmoother(attack=0.55, decay=0.15)
        self.ripples = RippleManager()
        self._raw_mic_level = 0.0
        self._raw_tts_level = 0.0
        self._phase = 0.0
        self._timer = None

    @property
    def state(self) -> str:
        return self.state_machine.state

    def on_mount(self) -> None:
        self._update_timer("idle")

    def set_state(self, new_state: str) -> None:
        if self.state_machine.transition(new_state):
            self._update_timer(new_state)
            self.refresh()

    def set_mic_level(self, level: float) -> None:
        self._raw_mic_level = level

    def set_tts_level(self, level: float) -> None:
        self._raw_tts_level = level

    def _update_timer(self, state: str) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        if state == "idle":
            # Low 2 fps timer during idle to maintain << 1% CPU
            interval = 1.0 / float(IDLE_FPS)
            self.smoother.reset()
            self.ripples.clear()
        else:
            # Active 28 fps timer during listening/thinking/speaking
            interval = 1.0 / float(self.fps)

        self._timer = self.set_interval(interval, self._tick, pause=False)

    def _tick(self) -> None:
        state = self.state_machine.state
        dt = 1.0 / float(self.fps if state != "idle" else IDLE_FPS)

        if state == "idle":
            self._phase += 0.08
            self.ripples.clear()
        elif state == "listening":
            self._phase += 0.22
            current = self.smoother.update(self._raw_mic_level)
            if self._raw_mic_level > 0.35 and current > 0.25:
                self.ripples.trigger(self.base_radius)
            self.ripples.update(dt)
        elif state == "thinking":
            self._phase += 0.18
            self.ripples.update(dt)
        elif state == "speaking":
            self._phase += 0.26
            current = self.smoother.update(self._raw_tts_level)
            if self._raw_tts_level > 0.35 and current > 0.25:
                self.ripples.trigger(self.base_radius)
            self.ripples.update(dt)

        self.refresh()

    def render(self) -> Text:
        width = max(10, self.size.width)
        height = max(5, self.size.height)

        state = self.state_machine.state
        level = 0.0
        if state == "listening":
            level = self.smoother.current
        elif state == "speaking":
            level = self.smoother.current

        # Ensure base radius scales nicely to fill container proportionally
        target_r = max(self.base_radius, min(12.0, (height / 2.0) - 1.5))
        max_r = max(2.5, min((height / 2.0) - 1.0, ((width / 2.0) - 2.0) * 0.5))
        effective_base = min(target_r, max_r)

        lines = generate_orb_grid(
            width=width,
            height=height,
            radius=effective_base,
            level=level,
            gain=self.gain,
            ramp=self.ramp,
            state=state,
            phase=self._phase,
            ripples=self.ripples.ripples,
            aspect_ratio=0.5,
        )

        style_color = self.orb_colors.get(state, DEFAULT_IDLE_COLOR)
        text = Text()
        for i, line in enumerate(lines):
            for ch in line:
                if ch in ("@", "%"):
                    text.append(ch, style="bold bright_white")
                elif ch == "#":
                    text.append(ch, style=f"bold {style_color}")
                elif ch in ("*", "+", "="):
                    text.append(ch, style=style_color)
                elif ch in (":", "-", "."):
                    text.append(ch, style=f"dim {style_color}")
                elif ch == "o":
                    text.append(ch, style="bold bright_white")
                else:
                    text.append(ch)
            if i < len(lines) - 1:
                text.append("\n")

        return text


class StartupSelectorModal(ModalScreen[Any]):
    """Startup modal for selecting thinking mode and voice preset."""

    DEFAULT_CSS = """
    StartupSelectorModal {
        align: center middle;
        background: rgba(8, 10, 15, 0.90);
    }
    #selector-box {
        width: 68;
        height: auto;
        border: solid cyan;
        background: #0f131c;
        padding: 1 2;
    }
    #selector-title {
        text-style: bold;
        color: #06b6d4;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    #mode-row, #voice-row {
        width: 100%;
        height: 1;
        content-align: center middle;
        text-align: center;
        margin-bottom: 1;
    }
    #selector-hint {
        color: #94a3b8;
        text-align: center;
        width: 100%;
        margin-top: 1;
    }
    #preview-status {
        color: #fbbf24;
        text-align: center;
        width: 100%;
        height: 1;
    }
    """

    BINDINGS = [
        Binding("up", "prev_row", "Up", show=False),
        Binding("down", "next_row", "Down", show=False),
        Binding("tab", "next_row", "Tab", show=False),
        Binding("left", "prev_opt", "Left", show=False),
        Binding("right", "next_opt", "Right", show=False),
        Binding("h", "prev_opt", "Left", show=False),
        Binding("l", "next_opt", "Right", show=False),
        Binding("p", "preview_voice", "Preview", show=False),
        Binding("P", "preview_voice", "Preview", show=False),
        Binding("enter", "confirm", "Confirm", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
        # Legacy shortcuts
        Binding("1", "select_online", "Online", show=False),
        Binding("2", "select_local", "Local", show=False),
    ]

    def __init__(
        self,
        default_mode: str = "online",
        default_voice: Optional[str] = None,
        cfg: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.default_mode = "local" if default_mode == "local" else "online"
        self.current_mode = self.default_mode
        self.cfg = cfg or {}
        self.active_row = 0  # 0: MODE, 1: VOICE
        self._preview_stop = threading.Event()
        self._preview_thread: Optional[threading.Thread] = None

        # Discover voice presets from config
        self.voice_presets = self._discover_voice_presets()

        # Preselect default voice
        target_voice = default_voice or self.cfg.get("voice", {}).get("default_preset", "calm")
        match_idx = 0
        for i, (k, _, avail) in enumerate(self.voice_presets):
            if k == target_voice and avail:
                match_idx = i
                break
        self.voice_index = match_idx
        self.current_voice = self.voice_presets[self.voice_index][0]
        self.default_voice = self.current_voice

    def _discover_voice_presets(self) -> List[Tuple[str, str, bool]]:
        """Return list of (preset_key, display_label, is_selectable)."""
        voice_cfg = self.cfg.get("voice") if isinstance(self.cfg.get("voice"), dict) else {}
        presets = voice_cfg.get("presets", {})
        if not isinstance(presets, dict) or not presets:
            return [("calm", "calm", True), ("shadow", "shadow", True)]

        items = []
        project_root = Path(__file__).resolve().parents[1]
        for key, p_data in presets.items():
            if not isinstance(p_data, dict):
                continue

            model_path_str = p_data.get("piper_model", "")
            model_exists = False
            if model_path_str:
                p = Path(model_path_str)
                if not p.is_absolute():
                    p = project_root / p
                model_exists = p.exists()
            else:
                model_exists = True

            if model_exists:
                items.append((key, key, True))
            else:
                items.append((key, f"{key} (model not downloaded)", False))
        return items or [("calm", "calm", True)]

    def compose(self) -> ComposeResult:
        with Vertical(id="selector-box"):
            yield Static("J A R V I S   C O N F I G U R A T I O N", id="selector-title")
            yield Static("", id="mode-row")
            yield Static("", id="voice-row")
            yield Static("", id="preview-status")
            yield Static(
                "up/down: row   left/right: change   P: preview   Enter: start",
                id="selector-hint",
            )

    def on_mount(self) -> None:
        self._render_display()

    def _render_display(self) -> None:
        try:
            mode_widget = self.query_one("#mode-row", Static)
            voice_widget = self.query_one("#voice-row", Static)
        except Exception:
            return

        is_mode_active = self.active_row == 0
        prefix_mode = "   " if not is_mode_active else "▶  "
        mode_text = Text()
        if is_mode_active:
            mode_text.append(prefix_mode, style="bold cyan")
            mode_text.append("MODE    <  ", style="bold #06b6d4")
            mode_text.append(f"{self.current_mode}", style="bold bright_white")
            mode_text.append("  >", style="bold #06b6d4")
        else:
            mode_text.append(prefix_mode, style="dim #64748b")
            mode_text.append(f"MODE    <  {self.current_mode}  >", style="#94a3b8")
        mode_widget.update(mode_text)

        is_voice_active = self.active_row == 1
        prefix_voice = "   " if not is_voice_active else "▶  "
        v_key, v_label, v_avail = self.voice_presets[self.voice_index]
        voice_text = Text()
        if is_voice_active:
            voice_text.append(prefix_voice, style="bold cyan")
            if v_avail:
                voice_text.append("VOICE   <  ", style="bold #06b6d4")
                voice_text.append(f"{v_key}", style="bold bright_white")
                voice_text.append("  >", style="bold #06b6d4")
            else:
                voice_text.append("VOICE   <  ", style="dim #64748b")
                voice_text.append(f"{v_label}", style="dim #ef4444")
                voice_text.append("  >", style="dim #64748b")
        else:
            voice_text.append(prefix_voice, style="dim #64748b")
            if v_avail:
                voice_text.append(f"VOICE   <  {v_key}  >", style="#94a3b8")
            else:
                voice_text.append(f"VOICE   <  {v_label}  >", style="dim #64748b")
        voice_widget.update(voice_text)

    def on_key(self, event: events.Key) -> None:
        if event.key.lower() not in ("p",):
            self._stop_preview()

    def action_prev_row(self) -> None:
        self.active_row = (self.active_row - 1) % 2
        self._stop_preview()
        self._render_display()

    def action_next_row(self) -> None:
        self.active_row = (self.active_row + 1) % 2
        self._stop_preview()
        self._render_display()

    def action_prev_opt(self) -> None:
        if self.active_row == 0:
            self.current_mode = "local" if self.current_mode == "online" else "online"
        else:
            n = len(self.voice_presets)
            self.voice_index = (self.voice_index - 1) % n
            self.current_voice = self.voice_presets[self.voice_index][0]
        self._stop_preview()
        self._render_display()

    def action_next_opt(self) -> None:
        if self.active_row == 0:
            self.current_mode = "local" if self.current_mode == "online" else "online"
        else:
            n = len(self.voice_presets)
            self.voice_index = (self.voice_index + 1) % n
            self.current_voice = self.voice_presets[self.voice_index][0]
        self._stop_preview()
        self._render_display()

    def action_preview_voice(self) -> None:
        v_key, v_label, v_avail = self.voice_presets[self.voice_index]
        if not v_avail:
            self._set_status_msg(f"{v_key} model not downloaded")
            return

        self._stop_preview()
        self._preview_stop = threading.Event()
        self._set_status_msg(f"Previewing {v_key} voice...")

        def _worker():
            try:
                engine = build_engine(self.cfg, preset=v_key)
                if engine is None:
                    return
                loop = asyncio.new_event_loop()
                samples, rate = loop.run_until_complete(
                    engine.synth("The system is primed. All channels nominal.")
                )
                loop.close()
                if self._preview_stop.is_set():
                    return
                preset_cfg = get_preset_config(self.cfg, v_key)
                fx = VoiceFXChain(VoiceFXConfig.from_dict(preset_cfg.get("fx")))
                samples, rate = fx.process(samples, rate)
                if self._preview_stop.is_set():
                    return
                play_audio(samples, rate, stop_event=self._preview_stop)
            except Exception as exc:
                _logger.warning("Voice preview error: %s", exc)
            finally:
                self.post_message(self._ClearStatusMsg())

        self._preview_thread = threading.Thread(target=_worker, daemon=True)
        self._preview_thread.start()

    class _ClearStatusMsg(events.Event):
        pass

    def on_startup_selector_modal__clear_status_msg(self, event: _ClearStatusMsg) -> None:
        self._set_status_msg("")

    def _set_status_msg(self, msg: str) -> None:
        try:
            status_widget = self.query_one("#preview-status", Static)
            status_widget.update(Text(msg, style="italic #fbbf24"))
        except Exception:
            pass

    def _stop_preview(self) -> None:
        self._preview_stop.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self._set_status_msg("")

    def action_confirm(self) -> None:
        v_key, v_label, v_avail = self.voice_presets[self.voice_index]
        if not v_avail:
            self._set_status_msg(f"Cannot select {v_key}: model not downloaded")
            return
        self._stop_preview()
        self.dismiss((self.current_mode, self.current_voice))

    def action_cancel(self) -> None:
        self._stop_preview()
        self.dismiss((self.default_mode, self.default_voice))

    def action_select_online(self) -> None:
        self.current_mode = "online"
        self.action_confirm()

    def action_select_local(self) -> None:
        self.current_mode = "local"
        self.action_confirm()

    def action_select_active(self) -> None:
        self.action_confirm()

    def action_select_default(self) -> None:
        self.action_cancel()


StartupModeModal = StartupSelectorModal


# ==============================================================================
# Full-Screen Textual App
# ==============================================================================
class JarvisTUIApp(App):
    """Full-screen terminal interface for Jarvis assistant."""

    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+m", "toggle_mode", "Switch Mode", show=False),
        Binding("ctrl+v", "toggle_voice", "Switch Voice", show=False),
    ]

    CSS = """
    Screen {
        layout: vertical;
        background: #080a0f;
        color: #e2e8f0;
    }
    #top-bar {
        dock: top;
        height: 2;
        padding: 0 2;
        background: transparent;
    }
    #app-title {
        color: #06b6d4;
        text-style: bold;
    }
    #notifications-slot {
        dock: right;
        display: none;
    }
    #orb-container {
        height: 1fr;
        min-height: 8;
        width: 100%;
        align: center middle;
        content-align: center middle;
    }
    #orb {
        width: 100%;
        height: 100%;
        content-align: center middle;
    }
    #response-container {
        height: auto;
        min-height: 5;
        max-height: 16;
        width: 100%;
        padding: 0 2;
        margin: 0 0 1 0;
        background: transparent;
        overflow-y: auto;
    }
    #response-text {
        width: 100%;
        height: auto;
        border-left: thick #06b6d4;
        background: #0e121a;
        padding: 0 1;
        color: #f1f5f9;
    }
    #bottom-bar {
        dock: bottom;
        height: 3;
        padding: 0 2;
        background: #080a0f;
        border-top: solid #1e293b;
        align: left middle;
    }
    #prompt-symbol {
        width: 2;
        color: #06b6d4;
        text-style: bold;
        height: 1;
        margin-right: 1;
    }
    Input {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        height: 1 !important;
    }
    Input:focus {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    #user-input {
        width: 1fr;
        height: 1;
        border: none !important;
        background: transparent !important;
        color: #f8fafc;
        padding: 0;
    }
    #user-input:focus {
        border: none !important;
        background: transparent !important;
    }
    #status-indicator {
        width: auto;
        height: 1;
        margin-left: 2;
    }
    """

    def __init__(
        self,
        skip_prompt: Optional[bool] = None,
        init_services: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.skip_prompt_override = skip_prompt
        self.init_services = init_services
        self.session = ChatSession()
        self.cfg: Dict[str, Any] = {}
        self.turn_queue: Optional[asyncio.Queue] = None
        self.selected_mode: str = "online"
        self.selected_voice: str = "calm"
        self._orig_stdout = None
        self._orig_stderr = None
        self._log_stream = None

    def post_to_app(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Thread-safe dispatch that works from both worker threads and the main thread."""
        try:
            if threading.current_thread() is threading.main_thread():
                callback(*args, **kwargs)
            else:
                self.call_from_thread(callback, *args, **kwargs)
        except Exception:  # noqa: BLE001
            pass

    def compose(self) -> ComposeResult:
        with Horizontal(id="top-bar"):
            yield Static("J A R V I S", id="app-title")
            yield NotificationPanel(id="notifications-slot")

        with Vertical(id="orb-container"):
            yield OrbWidget(id="orb")

        with Vertical(id="response-container"):
            yield Static("Jarvis online. Type a message or hold Ctrl+Alt+Space to talk.", id="response-text")

        with Horizontal(id="bottom-bar"):
            yield Static("❯ ", id="prompt-symbol")
            yield Input(placeholder="Ask Jarvis anything, or hold Ctrl+Alt+Space to speak...", id="user-input")
            yield StatusIndicator(id="status-indicator")

    async def on_mount(self) -> None:
        self.turn_queue = asyncio.Queue()
        self.orb: OrbWidget = self.query_one("#orb")
        self.response_container = self.query_one("#response-container")
        self.response_text: Static = self.query_one("#response-text")
        self.status_indicator: StatusIndicator = self.query_one("#status-indicator")
        self.notification_panel: NotificationPanel = self.query_one("#notifications-slot")
        self.user_input: Input = self.query_one("#user-input")

        self.user_input.focus()

        # Silence stdout/stderr and redirect prints to logs/jarvis.log in live mode
        if self.init_services:
            self._redirect_stdio()

        # Initialize session and background services
        if self.init_services:
            await self._init_session()
        else:
            self.session.cfg = self.session.load_config()
            self.cfg = self.session.cfg

        # Check startup mode selector configuration and persisted settings
        ui_cfg = self.cfg.get("ui") or {}
        if self.skip_prompt_override is not None:
            skip_prompt = self.skip_prompt_override
        else:
            skip_prompt = bool(ui_cfg.get("skip_mode_prompt", False))

        # Retrieve last-used mode and voice from SQLite settings
        saved_mode: Optional[str] = None
        saved_voice: Optional[str] = None
        try:
            saved_mode = get_setting("mode")
            saved_voice = get_setting("voice")
        except Exception as exc:
            _logger.warning("Could not read settings from database: %s", exc)

        default_mode = saved_mode or str(ui_cfg.get("default_mode", "online")).lower()
        if default_mode not in ("online", "local"):
            default_mode = "online"

        voice_cfg = self.cfg.get("voice") if isinstance(self.cfg.get("voice"), dict) else {}
        default_voice = (
            saved_voice or str(voice_cfg.get("default_preset", "calm")).lower()
        )

        if skip_prompt:
            self.selected_voice = default_voice
            self.session.set_voice_preset(default_voice)
            await self.apply_mode(default_mode)
        else:
            def _on_selection_chosen(chosen: Any) -> None:
                if isinstance(chosen, tuple):
                    mode_to_use, voice_to_use = chosen
                elif isinstance(chosen, str):
                    mode_to_use, voice_to_use = chosen, default_voice
                else:
                    mode_to_use, voice_to_use = default_mode, default_voice

                try:
                    set_setting("mode", mode_to_use)
                    set_setting("voice", voice_to_use)
                except Exception as exc:
                    _logger.warning("Could not persist settings to database: %s", exc)

                self.selected_voice = voice_to_use
                self.session.set_voice_preset(voice_to_use)
                asyncio.create_task(self.apply_mode(mode_to_use))

            self.push_screen(
                StartupSelectorModal(
                    default_mode=default_mode,
                    default_voice=default_voice,
                    cfg=self.cfg,
                ),
                _on_selection_chosen,
            )

        # Start background workers
        self._start_turn_worker()
        if self.init_services:
            self._start_connectivity_worker()

    def _redirect_stdio(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "jarvis.log"

        # Remove any existing console StreamHandlers
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)

        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root.addHandler(file_handler)
        root.setLevel(logging.INFO)

        for noisy in ("apscheduler", "httpx", "httpcore", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._log_stream = LogFileWriter(log_file)
        sys.stdout = self._log_stream
        sys.stderr = self._log_stream

    def on_unmount(self) -> None:
        """Clean shutdown of background services and restore terminal stdio."""
        if self._log_stream is not None:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
            self._log_stream.close()

        if self.session.speech:
            self.session.speech.stop()
        if self.session.voice:
            self.session.voice.stop()
        if self.session.screen_hotkey:
            self.session.screen_hotkey.stop()
        if self.session.screen_history:
            self.session.screen_history.stop()
        if self.session.scheduler:
            self.session.scheduler.shutdown()
        if self.session.task_service:
            self.session.task_service.close()
        if self.session.provider_chain:
            asyncio.create_task(self.session.provider_chain.aclose())

    async def _init_session(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        from dotenv import load_dotenv

        load_dotenv(project_root / ".env")
        self.session.cfg = self.session.load_config()
        self.cfg = self.session.cfg

        # Configure orb widget from config
        ui_cfg = self.cfg.get("ui") or {}
        if "ascii_ramp" in ui_cfg:
            self.orb.ramp = str(ui_cfg["ascii_ramp"])
        if "orb_base_radius" in ui_cfg:
            self.orb.base_radius = float(ui_cfg["orb_base_radius"])
        if "orb_gain" in ui_cfg:
            self.orb.gain = float(ui_cfg["orb_gain"])
        if "fps" in ui_cfg:
            self.orb.fps = int(ui_cfg["fps"])

        # Wire up Voice Output
        speech_cfg = self.cfg.get("speech") or {}
        if speech_cfg.get("enabled", True):
            try:
                from jarvis.speech import SpeechOutput

                self.session.speech = SpeechOutput(
                    self.cfg,
                    on_level=lambda lvl: self.post_to_app(
                        self.orb.set_tts_level, lvl
                    ),
                    on_state=lambda st: self.post_to_app(
                        self.orb.set_state, st
                    ),
                    log=lambda msg: _logger.info("Speech: %s", msg),
                )
                if self.session.speech.enabled:
                    self.session.speech.warm()
            except Exception as exc:
                _logger.warning("Voice output unavailable: %s", exc)

        # Wire up Voice Input (Hold-to-talk)
        voice_cfg = self.cfg.get("voice") or {}
        if voice_cfg.get("enabled", True):
            try:
                from jarvis.voice_input import HoldToTalk

                self.session.voice = HoldToTalk(
                    self.cfg,
                    on_transcript=lambda txt: self.post_to_app(
                        self.submit_user_message, txt
                    ),
                    on_record_start=(
                        self.session.speech.stop if self.session.speech else None
                    ),
                    on_level=lambda lvl: self.post_to_app(
                        self.orb.set_mic_level, lvl
                    ),
                    on_state=lambda st: self.post_to_app(
                        self.orb.set_state, st
                    ),
                    status=lambda msg: _logger.info("Voice: %s", msg),
                )
                self.session.voice.start()
            except Exception as exc:
                _logger.warning("Voice input unavailable: %s", exc)

        # Wire up Screen Reading
        screen_cfg = self.cfg.get("screen") or {}
        if screen_cfg.get("enabled", True):
            try:
                from jarvis import screen_tools
                from jarvis.screen import PauseHotkey, ScreenHistory

                self.session.screen = screen_tools.configure(self.cfg)
                hotkey = str(screen_cfg.get("pause_hotkey", "<ctrl>+<alt>+p"))
                self.session.screen_hotkey = PauseHotkey(
                    self.session.screen,
                    hotkey=hotkey,
                )
                self.session.screen_hotkey.start()
                self.session.screen_history = ScreenHistory(
                    self.session.screen,
                    config={"history": (screen_cfg.get("history") or {})},
                )
                self.session.screen_history.start()
            except Exception as exc:
                _logger.warning("Screen reading unavailable: %s", exc)

        # Wire up Reminder Scheduler
        task_cfg = self.cfg.get("tasks") or {}
        if task_cfg.get("enabled", True):
            try:
                from jarvis import task_tools
                from jarvis.scheduler import ReminderScheduler

                self.session.task_service = task_tools.get_service()
                self.session.scheduler = ReminderScheduler(
                    self.session.task_service,
                    task_cfg,
                    deliver=lambda msg: self.post_to_app(
                        self.notification_panel.push, "Reminder", msg, "info"
                    ),
                )
                self.session.scheduler.start()
            except Exception as exc:
                _logger.warning("Reminders scheduler unavailable: %s", exc)

    async def apply_mode(self, mode: str) -> None:
        """Validate and configure providers for chosen mode (online or local)."""
        self.selected_mode = mode
        self.session.mode = mode

        # Persist mode selection in SQLite database
        try:
            set_setting("mode", mode)
        except Exception:
            pass

        # Ensure config and env are loaded
        if not self.session.cfg:
            self.session.cfg = self.session.load_config()

        ready, reason = self.session.validate_mode(mode)
        if not ready:
            _logger.warning("Mode %s validation failed: %s", mode, reason)
            self.status_indicator.set_status(
                "error", "error", reason[:24], voice=self.selected_voice
            )
            return

        try:
            self.session.setup_providers(mode)
            if mode == "online":
                self.status_indicator.set_status(
                    "online", "online", voice=self.selected_voice
                )
            else:
                self.status_indicator.set_status(
                    "local", "local", voice=self.selected_voice
                )
            _logger.info("Successfully activated %s mode", mode)
        except Exception as exc:
            _logger.exception("Failed to setup %s providers: %s", mode, exc)
            self.status_indicator.set_status(
                "error", "error", str(exc)[:24], voice=self.selected_voice
            )

    async def action_toggle_mode(self) -> None:
        """Switch between online and local modes when Ctrl+M is pressed."""
        current = self.selected_mode
        new_mode = "local" if current == "online" else "online"
        await self.apply_mode(new_mode)

    def _get_presets_info(self) -> List[Tuple[str, str, bool]]:
        """Return list of (preset_key, display_label, is_available) for voice presets."""
        voice_cfg = self.cfg.get("voice") if isinstance(self.cfg.get("voice"), dict) else {}
        presets = voice_cfg.get("presets", {})
        if not isinstance(presets, dict) or not presets:
            return [("calm", "calm", True), ("shadow", "shadow", True)]

        items = []
        project_root = Path(__file__).resolve().parents[1]
        for key, p_data in presets.items():
            if not isinstance(p_data, dict):
                continue

            model_path_str = p_data.get("piper_model", "")
            model_exists = False
            if model_path_str:
                p = Path(model_path_str)
                if not p.is_absolute():
                    p = project_root / p
                model_exists = p.exists()
            else:
                model_exists = True

            if model_exists:
                items.append((key, key, True))
            else:
                items.append((key, f"{key} (model not downloaded)", False))
        return items or [("calm", "calm", True)]

    def apply_voice(self, voice: str) -> None:
        """Switch voice preset mid-session without losing chat history."""
        self.selected_voice = voice
        try:
            set_setting("voice", voice)
        except Exception:
            pass

        self.session.set_voice_preset(voice)
        self.status_indicator.set_status(
            self.selected_mode, self.selected_mode, voice=self.selected_voice
        )
        _logger.info("Activated voice preset: %s", voice)

    def action_toggle_voice(self) -> None:
        """Cycle through selectable voice presets when Ctrl+V is pressed."""
        selectable = [k for k, _, ok in self._get_presets_info() if ok]
        if not selectable:
            return
        curr = self.selected_voice
        idx = selectable.index(curr) if curr in selectable else 0
        new_voice = selectable[(idx + 1) % len(selectable)]
        self.apply_voice(new_voice)

    def _update_response(
        self, user_msg: str, jarvis_msg: str, status_msg: str = ""
    ) -> None:
        """Update response display and scroll to bottom so latest text is visible."""
        content = Text()
        if user_msg:
            content.append("You: ", style="bold cyan")
            content.append(f"{user_msg}\n", style="italic #94a3b8")
        if jarvis_msg or status_msg:
            content.append("Jarvis: ", style="bold magenta")
            if jarvis_msg:
                content.append(jarvis_msg, style="#f1f5f9")
            if status_msg:
                content.append(f" {status_msg}", style="dim yellow")
        self.response_text.update(content)
        self.call_after_refresh(self._scroll_response_to_end)

    def _scroll_response_to_end(self) -> None:
        try:
            self.response_container.scroll_end(animate=False)
        except Exception:
            pass

    @work(group="turns", exclusive=True)
    async def _start_turn_worker(self) -> None:
        """Processes messages sequentially from typed input and voice transcripts."""
        while self.is_running:
            try:
                user_msg = await asyncio.wait_for(self.turn_queue.get(), timeout=0.4)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if not user_msg:
                continue

            # Check if this is a slash command
            if user_msg.startswith("/"):
                try:
                    res = self.session.handle_command(user_msg)
                    self.post_to_app(
                        self.response_text.update,
                        Text(f"Command '{user_msg}' executed.", style="dim cyan"),
                    )
                except Exception as exc:
                    self.post_to_app(
                        self.response_text.update,
                        Text(f"Command error: {exc}", style="red"),
                    )
                continue

            self.orb.set_state("thinking")
            self.post_to_app(self._update_response, user_msg, "", "thinking...")

            accumulated_response = ""
            had_speech = False
            turn_started = time.perf_counter()

            if self.session.speech:
                self.session.speech.start_turn(turn_started)

            try:
                async for chunk in self.session.get_response_stream(user_msg):
                    accumulated_response += chunk
                    self.post_to_app(
                        self._update_response, user_msg, accumulated_response, ""
                    )
                    if self.session.speech:
                        had_speech = True
                        self.session.speech.feed(chunk)

                final_text = (
                    accumulated_response.strip()
                    if accumulated_response.strip()
                    else "(Completed)"
                )
                self.post_to_app(self._update_response, user_msg, final_text, "")

                if self.session.speech and had_speech:
                    await self.session.speech.flush()

            except Exception as exc:
                _logger.exception("Error processing turn: %s", exc)
                self.post_to_app(
                    self._update_response,
                    user_msg,
                    f"Sorry, I could not answer that: {exc}",
                    "",
                )
                self.status_indicator.set_status("error", "error", str(exc)[:24])
            finally:
                if self.orb.state != "listening":
                    self.orb.set_state("idle")

    @work(group="connectivity", exclusive=True)
    async def _start_connectivity_worker(self) -> None:
        """Periodically probe connectivity and update status indicator."""
        while self.is_running:
            try:
                await asyncio.sleep(12)
            except asyncio.CancelledError:
                break

            if not self.is_running:
                break

            mode = self.selected_mode
            if mode == "online":
                is_online = await asyncio.to_thread(probe_internet)
                if not is_online:
                    self.post_to_app(
                        self.status_indicator.set_status,
                        "error",
                        "error",
                        "no internet",
                        self.selected_voice,
                    )
                elif self.session.provider_chain is None:
                    # Attempt reconnect
                    try:
                        ready, reason = self.session.validate_mode("online")
                        if ready:
                            self.session.setup_providers("online")
                            self.post_to_app(
                                self.status_indicator.set_status,
                                "online",
                                "online",
                                "",
                                self.selected_voice,
                            )
                        else:
                            self.post_to_app(
                                self.status_indicator.set_status,
                                "error",
                                "error",
                                reason[:24],
                                self.selected_voice,
                            )
                    except Exception as exc:
                        self.post_to_app(
                            self.status_indicator.set_status,
                            "error",
                            "error",
                            str(exc)[:24],
                            self.selected_voice,
                        )
                else:
                    self.post_to_app(
                        self.status_indicator.set_status,
                        "online",
                        "online",
                        "",
                        self.selected_voice,
                    )
            elif mode == "local":
                has_error = self.session.provider_chain is None
                if has_error:
                    self.post_to_app(
                        self.status_indicator.set_status,
                        "error",
                        "error",
                        "ollama down",
                        self.selected_voice,
                    )
                else:
                    self.post_to_app(
                        self.status_indicator.set_status,
                        "local",
                        "local",
                        "",
                        self.selected_voice,
                    )

    def submit_user_message(self, text: str) -> None:
        """Enqueue message from typed input or transcript."""
        if self.turn_queue is not None:
            self.turn_queue.put_nowait(text)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        self._update_response(text, "", "thinking...")
        self.submit_user_message(text)


def main() -> None:
    """Entry point for the terminal user interface."""
    app = JarvisTUIApp()
    app.run()


if __name__ == "__main__":
    main()
