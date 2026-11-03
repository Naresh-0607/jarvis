"""Desktop Window UI for Jarvis assistant (Phase 9-UI).

Provides a frameless, dark desktop window interface powered by PySide6.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from PySide6.QtCore import QObject, QPoint, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizeGrip,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from jarvis.chat import ChatSession

_logger = logging.getLogger("jarvis.ui")

# ==============================================================================
# UI Styling and Sizing Configuration
# (Colors and sizes kept in one place for easy tweaking)
# ==============================================================================
DEFAULT_WIDTH = 900
DEFAULT_HEIGHT = 520
MIN_WIDTH = 640
MIN_HEIGHT = 380

# Dark theme palette
BG_COLOR = "#0c0e12"
SURFACE_COLOR = "#12161f"
CARD_BG = "#161b24"
INPUT_BG = "#151922"
INPUT_BORDER = "#252e3d"
INPUT_BORDER_FOCUS = "#00d2c4"

# Accent & glow
ACCENT_COLOR = "#00d2c4"  # soft cyan/teal
ACCENT_RGB = (0, 210, 196)
TEXT_COLOR = "#e2e8f0"
MUTED_TEXT = "#64748b"

# Status colors
COLOR_ONLINE = "#22c55e"  # green
COLOR_LOCAL = "#f59e0b"   # amber
COLOR_ERROR = "#ef4444"   # red


# ==============================================================================
# Pure Logic: State Machine, Level Smoother, Connectivity Detection
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

        # Any state can go to idle on completion or error
        if new_state == "idle":
            self._state = "idle"
            return True

        # Barge-in: speaking can transition directly to listening
        if self._state == "speaking" and new_state == "listening":
            self._state = "listening"
            return True

        # Standard lifecycle: idle -> listening/thinking -> speaking -> idle
        self._state = new_state
        return True


class LevelSmoother:
    """Filters audio amplitude levels with fast attack and smooth decay."""

    def __init__(self, attack: float = 0.45, decay: float = 0.18):
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


# ==============================================================================
# Qt Signals
# ==============================================================================
class JarvisSignals(QObject):
    """Signals for thread-safe cross-thread communication."""

    audio_level = Signal(float)
    state_changed = Signal(str)
    status_changed = Signal(str, str)  # (status, label)
    transcript_received = Signal(str)
    response_chunk = Signal(str)
    turn_completed = Signal()
    notification_pushed = Signal(str, str, str)  # (title, body, level)
    error_occurred = Signal(str)


# ==============================================================================
# Widgets
# ==============================================================================
class NotificationPanel(QFrame):
    """Top-right reserved notification space stub (Phase 9).

    Hidden by default. Ready to display notifications via push().
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("notificationPanel")
        self.setVisible(False)
        self.setMinimumWidth(220)
        self.setMaximumWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self._title = QLabel("Notifications", self)
        self._title.setStyleSheet(
            f"color: {ACCENT_COLOR}; font-weight: 600; font-size: 12px;"
        )
        self._body = QLabel("", self)
        self._body.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 11px;")
        self._body.setWordWrap(True)

        layout.addWidget(self._title)
        layout.addWidget(self._body)

        self.setStyleSheet(
            f"""
            QFrame#notificationPanel {{
                background-color: {CARD_BG};
                border: 1px solid {INPUT_BORDER};
                border-radius: 8px;
            }}
            """
        )

    def push(self, title: str, body: str, level: str = "info") -> None:
        """Display an incoming notification."""
        self._title.setText(title)
        self._body.setText(body)
        color = COLOR_ONLINE if level == "success" else (COLOR_ERROR if level == "error" else ACCENT_COLOR)
        self._title.setStyleSheet(f"color: {color}; font-weight: 600; font-size: 12px;")
        self.setVisible(True)


class StatusDot(QWidget):
    """Small status indicator with dot and label beside it (matches sketch)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._status = "online"
        self._label_text = "online"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._label = QLabel("online", self)
        self._label.setStyleSheet(
            f"color: {MUTED_TEXT}; font-size: 12px; font-weight: 500;"
        )

        self._dot = QLabel(self)
        self._dot.setFixedSize(9, 9)

        # Sketch shows label then dot in bottom right
        layout.addWidget(self._label)
        layout.addWidget(self._dot)

        self.set_status("online", "online")

    def set_status(self, status: str, label: str) -> None:
        self._status = status
        self._label_text = label
        self._label.setText(label)
        color = COLOR_ONLINE if status == "online" else (COLOR_LOCAL if status == "local" else COLOR_ERROR)
        self._dot.setStyleSheet(
            f"background-color: {color}; border-radius: 4px; min-width: 9px; min-height: 9px;"
        )


class WaveformWidget(QWidget):
    """Dominant live audio waveform that spans the window.

    Renders organic curves with soft glow matching the hand-drawn sketch.
    Stays under 1% CPU when idle by using a slow 10 fps timer, switching
    to ~60 fps only during active states (listening, thinking, speaking).
    """

    def __init__(self, parent: Optional[QWidget] = None, accent_color: str = ACCENT_COLOR):
        super().__init__(parent)
        self.accent_color = accent_color
        self._state_machine = UIStateMachine("idle")
        self._smoother = LevelSmoother(attack=0.45, decay=0.18)
        self._raw_level: float = 0.0
        self._phase: float = 0.0

        # Animation timer runs only during non-idle states (listening, thinking, speaking)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        # In idle, timer is stopped for ~0% CPU usage
        self._timer.setInterval(16)

    @property
    def state(self) -> str:
        return self._state_machine.state

    def set_state(self, new_state: str) -> None:
        if self._state_machine.transition(new_state):
            if self._state_machine.is_idle:
                # In idle: stop timer completely for < 0.1% CPU usage
                self._timer.stop()
                self._smoother.reset()
                self._raw_level = 0.0
            else:
                # Non-idle: ~60 fps active animation timer
                if not self._timer.isActive():
                    self._timer.start(16)
            self.update()

    def set_level(self, level: float) -> None:
        self._raw_level = level

    def _on_tick(self) -> None:
        state = self._state_machine.state
        if state == "idle":
            self._phase += 0.05
        elif state == "listening":
            self._phase += 0.15
        elif state == "thinking":
            self._phase += 0.12
        elif state == "speaking":
            self._phase += 0.20

        if self._phase > 1000.0:
            self._phase = 0.0

        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        width = self.width()
        height = self.height()
        cy = height * 0.5

        smoothed = self._smoother.update(self._raw_level)
        state = self._state_machine.state
        phase = self._phase

        # Generate organic curve points
        n_points = max(60, min(140, int(width / 7)))
        points: List[Tuple[float, float]] = []

        for i in range(n_points):
            x = width * (i / float(n_points - 1))
            norm_x = (x / width) * 2.0 - 1.0  # -1.0 to 1.0
            envelope = math.exp(-2.2 * (norm_x**2))  # Taper edges smoothly

            if state == "idle":
                # Calm, nearly flat line with subtle breathing
                breath = 0.02 + 0.01 * math.sin(phase * 0.6)
                dy = (
                    math.sin(i * 0.12 + phase) * 0.7
                    + math.sin(i * 0.24 - phase * 0.5) * 0.3
                ) * (breath * height * 0.4) * envelope

            elif state == "listening":
                # Real mic RMS amplitude with responsive organic spikes
                amp = max(0.04, min(1.0, smoothed * 3.8))
                primary = math.sin(i * 0.35 + phase * 2.0)
                harmonic1 = math.sin(i * 0.75 - phase * 1.5) * 0.5
                harmonic2 = math.sin(i * 1.4 + phase * 3.0) * 0.25
                dy = (primary + harmonic1 + harmonic2) * (amp * height * 0.45) * envelope

            elif state == "thinking":
                # Gentle traveling harmonic pulse
                pulse = math.sin(i * 0.25 - phase * 2.5) * math.exp(
                    -3.5 * ((norm_x - math.sin(phase * 0.8) * 0.6) ** 2)
                )
                dy = pulse * (height * 0.32)

            elif state == "speaking":
                # Driven by actual TTS PCM chunks amplitude
                amp = max(0.06, min(1.0, smoothed * 3.5))
                # Organic spiky look matching the sketch
                w1 = math.sin(i * 0.30 + phase * 2.5)
                w2 = math.sin(i * 0.65 - phase * 1.8) * 0.6
                w3 = math.sin(i * 1.35 + phase * 4.0) * 0.35
                spike = math.sin(i * 2.1 - phase * 5.0) * 0.18
                dy = (w1 + w2 + w3 + spike) * (amp * height * 0.46) * envelope

            else:
                dy = 0.0

            points.append((x, cy + dy))

        # Build smooth path
        path = QPainterPath()
        path.moveTo(points[0][0], points[0][1])
        for i in range(1, len(points)):
            p0 = points[i - 1]
            p1 = points[i]
            mid_x = (p0[0] + p1[0]) * 0.5
            mid_y = (p0[1] + p1[1]) * 0.5
            path.quadTo(p0[0], p0[1], mid_x, mid_y)
        path.lineTo(points[-1][0], points[-1][1])

        # Layer 1: Soft glow
        glow_color = QColor(self.accent_color)
        glow_color.setAlpha(45)
        glow_pen = QPen(glow_color, 6.0, Qt.PenStyle.SolidLine)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        glow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(glow_pen)
        painter.drawPath(path)

        # Layer 2: Medium glow
        mid_color = QColor(self.accent_color)
        mid_color.setAlpha(110)
        mid_pen = QPen(mid_color, 3.5, Qt.PenStyle.SolidLine)
        mid_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(mid_pen)
        painter.drawPath(path)

        # Layer 3: Crisp main line
        core_pen = QPen(QColor(self.accent_color), 2.0, Qt.PenStyle.SolidLine)
        core_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(core_pen)
        painter.drawPath(path)


# ==============================================================================
# Main Desktop Window
# ==============================================================================
class JarvisWindow(QMainWindow):
    """Main desktop interface window matching the hand-drawn sketch."""

    def __init__(self, bridge: "JarvisBridge", config: Optional[Dict] = None):
        super().__init__()
        self.bridge = bridge
        self.cfg = config or {}
        ui_cfg = self.cfg.get("ui") or {}

        # Window dimensions
        w = int(ui_cfg.get("width", DEFAULT_WIDTH))
        h = int(ui_cfg.get("height", DEFAULT_HEIGHT))
        min_w = int(ui_cfg.get("min_width", MIN_WIDTH))
        min_h = int(ui_cfg.get("min_height", MIN_HEIGHT))
        self.resize(w, h)
        self.setMinimumSize(min_w, min_h)

        self.accent_color = str(ui_cfg.get("accent_color", ACCENT_COLOR))
        self.bg_color = str(ui_cfg.get("bg_color", BG_COLOR))
        self.show_caption = bool(ui_cfg.get("show_caption", False))

        self._quitting = False
        self._drag_pos: Optional[QPoint] = None

        if not self.show_caption:
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        self.setWindowTitle("Jarvis")
        self._setup_ui()
        self._setup_tray()
        self._connect_signals()

    def _setup_ui(self) -> None:
        central = QWidget(self)
        central.setObjectName("centralWidget")
        central.setStyleSheet(
            f"""
            QWidget#centralWidget {{
                background-color: {self.bg_color};
                border: 1px solid {INPUT_BORDER};
            }}
            """
        )
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(28, 20, 28, 20)
        main_layout.setSpacing(12)

        # 1. Top Bar: "Jarvis" title (left), NotificationPanel & Controls (right)
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel("Jarvis", self)
        title_font = QFont("Segoe UI", 22, QFont.Weight.Bold)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2.5)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet(
            f"color: {self.accent_color}; margin-top: 4px;"
        )

        top_bar.addWidget(self.title_label)
        top_bar.addStretch()

        # Notification panel stub (hidden by default)
        self.notification_panel = NotificationPanel(self)
        top_bar.addWidget(self.notification_panel)

        # Window controls for frameless mode
        if not self.show_caption:
            controls_layout = QHBoxLayout()
            controls_layout.setSpacing(8)

            btn_min = QPushButton("—", self)
            btn_min.setFixedSize(26, 26)
            btn_min.setToolTip("Minimize")
            btn_min.setStyleSheet(
                f"""
                QPushButton {{
                    background: transparent; color: {MUTED_TEXT};
                    border: 1px solid {INPUT_BORDER}; border-radius: 13px; font-size: 11px;
                }}
                QPushButton:hover {{
                    background: {INPUT_BG}; color: {TEXT_COLOR};
                }}
                """
            )
            btn_min.clicked.connect(self.showMinimized)

            btn_close = QPushButton("✕", self)
            btn_close.setFixedSize(26, 26)
            btn_close.setToolTip("Hide to Tray")
            btn_close.setStyleSheet(
                f"""
                QPushButton {{
                    background: transparent; color: {MUTED_TEXT};
                    border: 1px solid {INPUT_BORDER}; border-radius: 13px; font-size: 11px;
                }}
                QPushButton:hover {{
                    background: {INPUT_BG}; color: {COLOR_ERROR};
                }}
                """
            )
            btn_close.clicked.connect(self.hide)

            controls_layout.addWidget(btn_min)
            controls_layout.addWidget(btn_close)
            top_bar.addLayout(controls_layout)

        main_layout.addLayout(top_bar)

        # 2. Center: Live audio waveform spanning the width
        self.waveform = WaveformWidget(self, accent_color=self.accent_color)
        main_layout.addWidget(self.waveform, 1)

        # 3. Bottom Bar: Input bar (left) and Status Dot (right)
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(0, 0, 0, 0)
        bottom_bar.setSpacing(16)

        # Text input bar with `>` prompt
        self.input_container = QFrame(self)
        self.input_container.setObjectName("inputContainer")
        self.input_container.setFixedHeight(44)
        input_layout = QHBoxLayout(self.input_container)
        input_layout.setContentsMargins(14, 0, 14, 0)
        input_layout.setSpacing(8)

        self.prompt_label = QLabel(">", self.input_container)
        self.prompt_label.setStyleSheet(
            f"color: {self.accent_color}; font-size: 16px; font-weight: bold;"
        )

        self.input_field = QLineEdit(self.input_container)
        self.input_field.setObjectName("inputField")
        self.input_field.setPlaceholderText("type here...")
        self.input_field.setStyleSheet(
            f"""
            QLineEdit#inputField {{
                background: transparent;
                border: none;
                color: {TEXT_COLOR};
                font-size: 13px;
                selection-background-color: {self.accent_color};
            }}
            """
        )
        self.input_field.returnPressed.connect(self._on_submit_text)

        input_layout.addWidget(self.prompt_label)
        input_layout.addWidget(self.input_field)

        self.input_container.setStyleSheet(
            f"""
            QFrame#inputContainer {{
                background-color: {INPUT_BG};
                border: 1px solid {INPUT_BORDER};
                border-radius: 22px;
            }}
            QFrame#inputContainer:focus-within {{
                border: 1px solid {self.accent_color};
            }}
            """
        )

        bottom_bar.addWidget(self.input_container, 4)

        # Status dot with label beside it
        self.status_dot = StatusDot(self)
        bottom_bar.addWidget(self.status_dot, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Size grip in bottom corner for frameless window resizing
        if not self.show_caption:
            size_grip = QSizeGrip(self)
            size_grip.setStyleSheet("width: 14px; height: 14px; background: transparent;")
            bottom_bar.addWidget(size_grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

        main_layout.addLayout(bottom_bar)

    def _setup_tray(self) -> None:
        """Setup system tray icon for background hide/show."""
        self.tray = QSystemTrayIcon(self)

        # Generate a sleek circular icon
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setBrush(QBrush(QColor(self.accent_color)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(4, 4, 24, 24)
        p.end()

        self.tray.setIcon(QIcon(pixmap))
        self.tray.setToolTip("Jarvis Assistant")

        tray_menu = QMenu()
        self.action_toggle_window = tray_menu.addAction("Hide Jarvis")
        self.action_toggle_window.triggered.connect(self._toggle_visibility)

        self.action_toggle_screen = tray_menu.addAction("Screen Reading: active")
        self.action_toggle_screen.triggered.connect(self._toggle_screen_reading)

        tray_menu.addSeparator()
        action_quit = tray_menu.addAction("Quit Jarvis")
        action_quit.triggered.connect(self.quit_app)

        self.tray.setContextMenu(tray_menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _connect_signals(self) -> None:
        self.bridge.signals.audio_level.connect(self.waveform.set_level)
        self.bridge.signals.state_changed.connect(self.waveform.set_state)
        self.bridge.signals.status_changed.connect(self.status_dot.set_status)
        self.bridge.signals.turn_completed.connect(self._on_turn_completed)
        self.bridge.signals.notification_pushed.connect(self.notification_panel.push)

    # --------------------------------------------------------------------------
    # User Input & Turn Flow
    # --------------------------------------------------------------------------
    def _on_submit_text(self) -> None:
        text = self.input_field.text().strip()
        if not text:
            return
        self.input_field.clear()
        self.waveform.set_state("thinking")
        self.bridge.submit_message(text)

    def _on_turn_completed(self) -> None:
        self.waveform.set_state("idle")

    # --------------------------------------------------------------------------
    # Tray & Window Controls
    # --------------------------------------------------------------------------
    def _toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide()
            self.action_toggle_window.setText("Show Jarvis")
        else:
            self.show()
            self.raise_()
            self.activateWindow()
            self.action_toggle_window.setText("Hide Jarvis")

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._toggle_visibility()

    def _toggle_screen_reading(self) -> None:
        session = self.bridge.session
        if session and session.screen:
            session.screen.toggle()
            state = "paused" if session.screen.paused else "active"
            self.action_toggle_screen.setText(f"Screen Reading: {state}")

    def closeEvent(self, event) -> None:
        """Closing the window hides it to tray rather than exiting."""
        if not self._quitting:
            event.ignore()
            self.hide()
            self.action_toggle_window.setText("Show Jarvis")
        else:
            event.accept()

    def quit_app(self) -> None:
        """Clean shutdown of background services and exit."""
        self._quitting = True
        self.tray.hide()
        self.bridge.stop()
        QApplication.quit()

    # --------------------------------------------------------------------------
    # Frameless window dragging
    # --------------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self.show_caption:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if (
            event.buttons() == Qt.MouseButton.LeftButton
            and self._drag_pos is not None
            and not self.show_caption
        ):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None


# ==============================================================================
# Background Assistant Bridge (Asyncio Worker Thread)
# ==============================================================================
class JarvisBridge:
    """Runs ChatSession in a dedicated background worker thread."""

    def __init__(self, signals: JarvisSignals):
        self.signals = signals
        self.session = ChatSession()
        self.cfg: Dict[str, Any] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._turn_queue: "asyncio.Queue[str]" = None
        self._check_timer: Optional[threading.Timer] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_main, name="jarvis-bridge", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._check_timer:
            self._check_timer.cancel()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._cleanup_sync)

    def submit_message(self, text: str) -> None:
        """Enqueue message from typed input or voice transcript."""
        if self._loop and self._loop.is_running() and self._turn_queue:
            self._loop.call_soon_threadsafe(self._turn_queue.put_nowait, text)

    def _worker_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_async())
        finally:
            self._loop.close()

    async def _run_async(self) -> None:
        self.cfg = self.session.load_config()
        self._turn_queue = asyncio.Queue()

        # Connect audio level and state callbacks
        def on_audio_level(lvl: float) -> None:
            self.signals.audio_level.emit(lvl)

        def on_audio_state(state: str) -> None:
            self.signals.state_changed.emit(state)

        # 1. Voice output
        speech_cfg = self.cfg.get("speech") or {}
        if speech_cfg.get("enabled", True):
            try:
                from jarvis.speech import SpeechOutput

                self.session.speech = SpeechOutput(
                    self.cfg,
                    on_level=on_audio_level,
                    on_state=on_audio_state,
                    log=lambda msg: _logger.info("Speech: %s", msg),
                )
                if self.session.speech.enabled:
                    self.session.speech.warm()
            except Exception as exc:
                _logger.warning("Voice output unavailable: %s", exc)
                self.session.speech = None

        # 2. Hold-to-talk voice input
        voice_cfg = self.cfg.get("voice") or {}
        if voice_cfg.get("enabled", True):
            try:
                from jarvis.voice_input import HoldToTalk

                self.session.voice = HoldToTalk(
                    self.cfg,
                    on_transcript=self.submit_message,
                    on_record_start=(self.session.speech.stop if self.session.speech else None),
                    on_level=on_audio_level,
                    on_state=on_audio_state,
                    status=lambda msg: _logger.info("Voice: %s", msg),
                )
                self.session.voice.start()
            except Exception as exc:
                _logger.warning("Voice input unavailable: %s", exc)
                self.session.voice = None

        # 3. Screen tools & privacy hotkey
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

        # 4. Reminders & Scheduler
        task_cfg = self.cfg.get("tasks") or {}
        if task_cfg.get("enabled", True):
            try:
                from jarvis import task_tools
                from jarvis.scheduler import ReminderScheduler

                self.session.task_service = task_tools.get_service()
                self.session.scheduler = ReminderScheduler(
                    self.session.task_service,
                    task_cfg,
                    deliver=lambda msg: self.signals.notification_pushed.emit(
                        "Reminder", msg, "info"
                    ),
                )
                self.session.scheduler.start()
            except Exception as exc:
                _logger.warning("Reminders scheduler unavailable: %s", exc)

        # 5. Initialize providers and connectivity detection
        await self._init_providers()

        # Start periodic background connectivity check
        self._schedule_connectivity_check(initial=True)

        # Process message queue turns
        while self._running:
            try:
                user_input = await self._turn_queue.get()
            except asyncio.CancelledError:
                break

            if not user_input:
                continue

            await self._process_turn(user_input)

    async def _init_providers(self) -> None:
        """Select online or local mode based on availability."""
        ready, _reason = self.session.validate_mode("online")
        if ready:
            try:
                self.session.setup_providers("online")
                self.signals.status_changed.emit("online", "online")
                return
            except Exception as exc:
                _logger.warning("Online providers setup failed: %s", exc)

        ready_local, _reason_local = self.session.validate_mode("local")
        if ready_local:
            try:
                self.session.setup_providers("local")
                self.signals.status_changed.emit("local", "local")
                return
            except Exception as exc:
                _logger.warning("Local providers setup failed: %s", exc)

        self.signals.status_changed.emit("error", "error")

    def _schedule_connectivity_check(self, initial: bool = False) -> None:
        """Periodically ping in background thread without blocking the UI."""
        if not self._running:
            return

        def _check():
            is_online = False
            try:
                # Fast check to public DNS / HTTP
                resp = httpx.get("https://1.1.1.1", timeout=3.0)
                is_online = resp.status_code in (200, 301, 302, 400)
            except Exception:
                is_online = False

            local_avail = False
            ollama_url = os.environ.get("OLLAMA_BASE_URL", "")
            if ollama_url:
                local_avail = True

            has_error = self.session.provider_chain is None
            status, label = determine_connectivity(
                is_online_reachable=is_online,
                local_available=local_avail,
                has_error=has_error,
            )
            self.signals.status_changed.emit(status, label)

            if self._running:
                self._check_timer = threading.Timer(25.0, _check)
                self._check_timer.daemon = True
                self._check_timer.start()

        delay = 0.5 if initial else 25.0
        self._check_timer = threading.Timer(delay, _check)
        self._check_timer.daemon = True
        self._check_timer.start()

    async def _process_turn(self, user_input: str) -> None:
        """Execute one conversation turn (identical routing, tools, and persona)."""
        turn_started = time.perf_counter()
        if self.session.speech:
            self.session.speech.start_turn(turn_started)

        self.signals.state_changed.emit("thinking")
        had_speech = False

        try:
            async for chunk in self.session.get_response_stream(user_input):
                self.signals.response_chunk.emit(chunk)
                if self.session.speech:
                    had_speech = True
                    self.session.speech.feed(chunk)

            if self.session.speech and had_speech:
                await self.session.speech.flush()

        except Exception as exc:
            _logger.error("Error processing message: %s", exc)
            self.signals.error_occurred.emit(str(exc))
            self.signals.state_changed.emit("idle")
        finally:
            self.signals.turn_completed.emit()

    def _cleanup_sync(self) -> None:
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


# ==============================================================================
# UI Main Entrypoint
# ==============================================================================
def setup_quiet_logging(project_root: Path) -> None:
    """Send all logs to logs/jarvis.log, keeping terminal quiet."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "jarvis.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # File handler with full detail
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Silence noisy loggers
    for noisy in ("apscheduler", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    """Launch the Jarvis desktop UI application."""
    project_root = Path(__file__).resolve().parents[1]
    setup_quiet_logging(project_root)

    app = QApplication(sys.argv)
    app.setApplicationName("Jarvis")
    app.setQuitOnLastWindowClosed(False)

    signals = JarvisSignals()
    bridge = JarvisBridge(signals)
    bridge.start()

    cfg = bridge.session.load_config()
    window = JarvisWindow(bridge, config=cfg)

    ui_cfg = cfg.get("ui") or {}
    start_hidden = bool(ui_cfg.get("start_hidden", False))

    if not start_hidden:
        window.show()

    # Clean exit on Ctrl+C or app quit
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
