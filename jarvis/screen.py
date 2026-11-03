"""View-only screen reading: capture, OCR and the privacy gates (Phase 5).

Nothing in this module writes an image to disk, moves the mouse or sends keys.
A capture lives in memory, is read once, and is dropped. The only two ways a
capture can leave this laptop are (a) OCR text going to the brain as plain
text and (b) an explicit `describe_screen_image` call, which is off until you
configure a vision provider.

Privacy rules enforced here:
* `screen.access_paused` (Ctrl+Alt+P, the tray icon or `/screen pause`) makes
  every capture refuse with a clear reason.
* A window whose title matches the configured blocklist is refused before the
  pixels are read, so a password manager never even gets captured.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_logger = logging.getLogger("jarvis.screen")

DEFAULT_MAX_WIDTH = 1600
DEFAULT_MAX_CHARS = 4000
DEFAULT_MIN_SCORE = 0.5
DEFAULT_PAUSE_HOTKEY = "<ctrl>+<alt>+p"

# Window titles that must never be read. Matched case-insensitively as
# substrings, because browsers append the page title.
DEFAULT_BLOCKLIST = (
    "1password",
    "bitwarden",
    "keepass",
    "lastpass",
    "dashlane",
    "authenticator",
    "credential manager",
    "windows security",
    "paypal",
    "stripe dashboard",
    "net banking",
    "hdfc bank",
    "icici bank",
    "state bank of india",
    "coinbase",
    "binance",
)


class ScreenError(Exception):
    """A capture was refused or failed. The message is safe to speak aloud."""


@dataclass(frozen=True)
class OcrLine:
    """One recognised piece of text with a rough position in the capture."""

    text: str
    score: float
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Capture:
    """One in-memory screenshot. Never written to disk."""

    image: np.ndarray
    region: str
    title: str
    capture_ms: float


# ----------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------
class _Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def active_window() -> Optional[Tuple[str, Dict[str, int]]]:
    """Return (title, rect) for the focused window, or None when unusable."""
    if not hasattr(ctypes, "windll"):
        return None
    try:
        user32 = ctypes.windll.user32
        handle = user32.GetForegroundWindow()
        if not handle:
            return None

        length = user32.GetWindowTextLengthW(handle)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, buffer, length + 1)
        title = buffer.value.strip()

        rect = _Rect()
        if not user32.GetWindowRect(handle, ctypes.byref(rect)):
            return None

        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width < 200 or height < 120:
            return None

        return title, {
            "left": int(rect.left),
            "top": int(rect.top),
            "width": width,
            "height": height,
        }
    except (AttributeError, OSError):  # not Windows, or no desktop session
        return None


def primary_monitor() -> Dict[str, int]:
    """Bounds of the primary monitor, without importing mss."""
    if hasattr(ctypes, "windll"):
        try:
            user32 = ctypes.windll.user32
            return {
                "left": 0,
                "top": 0,
                "width": int(user32.GetSystemMetrics(0)),
                "height": int(user32.GetSystemMetrics(1)),
            }
        except (AttributeError, OSError):
            pass
    return {"left": 0, "top": 0, "width": 1920, "height": 1080}


def _downscale(image: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or image.shape[1] <= max_width:
        return image
    from PIL import Image

    ratio = max_width / float(image.shape[1])
    size = (max_width, max(1, int(round(image.shape[0] * ratio))))
    return np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR))


def grab(
    rect: Optional[Dict[str, int]] = None,
    title: str = "",
    region: str = "monitor",
    max_width: int = DEFAULT_MAX_WIDTH,
) -> Capture:
    """Capture a region into memory. Returns BGRA-derived BGR pixels."""
    try:
        import mss
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ScreenError("screen capture needs mss; run 'pip install mss'") from exc

    bounds = rect or primary_monitor()
    started = time.perf_counter()
    try:
        with mss.mss() as screen:
            raw = np.asarray(screen.grab(bounds))
    except Exception as exc:  # noqa: BLE001 - locked screen, driver trouble
        raise ScreenError(f"could not capture the screen: {exc}") from exc
    capture_ms = (time.perf_counter() - started) * 1000

    image = np.ascontiguousarray(raw[:, :, :3])  # BGRA -> BGR, matching cv2
    if not image.size:
        raise ScreenError("the capture was empty")
    return Capture(
        image=_downscale(image, max_width),
        region=region,
        title=title,
        capture_ms=capture_ms,
    )


# ----------------------------------------------------------------------
# OCR engines
# ----------------------------------------------------------------------
def _result_lines(result: Any, min_score: float) -> List[OcrLine]:
    """Turn a RapidOCR result into OcrLine records, skipping weak matches."""
    lines: List[OcrLine] = []
    for item in result or []:
        try:
            box, text, score = item[0], item[1], item[2]
        except (TypeError, IndexError):
            continue
        text = str(text or "").strip()
        if not text or float(score) < min_score:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        lines.append(
            OcrLine(
                text=text,
                score=float(score),
                x=int(min(xs)),
                y=int(min(ys)),
                width=int(max(xs) - min(xs)),
                height=int(max(ys) - min(ys)),
            )
        )
    return lines


class RapidOcrEngine:
    """RapidOCR (ONNX runtime). ~560 ms to load, ~1.5 s per window warm."""

    name = "rapidocr"

    def __init__(self, min_score: float = DEFAULT_MIN_SCORE):
        self.min_score = float(min_score)
        self._engine = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("rapidocr_onnxruntime") is not None

    def _load(self):
        with self._lock:
            if self._engine is None:
                from rapidocr_onnxruntime import RapidOCR

                self._engine = RapidOCR()
            return self._engine

    def warm(self) -> None:
        """Pay the first-inference cost away from a real question."""
        engine = self._load()
        blank = np.full((64, 240, 3), 255, dtype=np.uint8)
        try:
            engine(blank)
        except Exception as exc:  # noqa: BLE001 - warm-up is best effort
            _logger.debug("OCR warm-up failed: %s", exc)

    def read(self, image: np.ndarray) -> List[OcrLine]:
        engine = self._load()
        try:
            result, _elapsed = engine(image)
        except Exception as exc:  # noqa: BLE001 - bad frame, ONNX hiccup
            raise ScreenError(f"OCR failed: {exc}") from exc
        return _result_lines(result, self.min_score)


class TesseractOcrEngine:
    """Fallback when RapidOCR is missing. Needs pytesseract + the tesseract binary."""

    name = "tesseract"

    def __init__(self, min_score: float = DEFAULT_MIN_SCORE):
        self.min_score = float(min_score)

    @property
    def available(self) -> bool:
        import importlib.util
        import shutil

        return (
            importlib.util.find_spec("pytesseract") is not None
            and shutil.which("tesseract") is not None
        )

    def warm(self) -> None:
        return None

    def read(self, image: np.ndarray) -> List[OcrLine]:
        import pytesseract
        from PIL import Image

        rgb = Image.fromarray(image[:, :, ::-1])
        data = pytesseract.image_to_data(
            rgb, output_type=pytesseract.Output.DICT
        )
        lines: List[OcrLine] = []
        for index, text in enumerate(data.get("text", [])):
            text = str(text or "").strip()
            confidence = float(data["conf"][index])
            if not text or confidence < self.min_score * 100:
                continue
            lines.append(
                OcrLine(
                    text=text,
                    score=confidence / 100.0,
                    x=int(data["left"][index]),
                    y=int(data["top"][index]),
                    width=int(data["width"][index]),
                    height=int(data["height"][index]),
                )
            )
        return lines


def build_ocr(
    backend: str = "auto", min_score: float = DEFAULT_MIN_SCORE
) -> Any:
    """Pick an OCR engine: RapidOCR, or Tesseract as the fallback."""
    backend = (backend or "auto").lower()
    candidates = (
        [TesseractOcrEngine(min_score)]
        if backend == "tesseract"
        else [RapidOcrEngine(min_score), TesseractOcrEngine(min_score)]
    )
    for engine in candidates:
        if engine.available:
            return engine

    raise ScreenError(
        "no OCR engine is installed; run 'pip install rapidocr-onnxruntime' "
        "(or install pytesseract and the tesseract binary)"
    )


# ----------------------------------------------------------------------
# Reading order
# ----------------------------------------------------------------------
def format_lines(lines: Sequence[OcrLine], max_chars: int = DEFAULT_MAX_CHARS) -> Tuple[str, bool]:
    """Join OCR lines into reading order: rows top to bottom, words left to right."""
    if not lines:
        return "", False

    rows: List[Dict[str, Any]] = []
    for line in sorted(lines, key=lambda item: (item.y, item.x)):
        centre = line.y + line.height / 2.0
        tolerance = max(6.0, line.height * 0.6)
        for row in rows:
            if abs(row["centre"] - centre) <= tolerance:
                row["items"].append(line)
                break
        else:
            rows.append({"centre": centre, "y": line.y, "items": [line]})

    ordered_rows = [
        " ".join(item.text for item in sorted(row["items"], key=lambda i: i.x))
        for row in sorted(rows, key=lambda row: row["y"])
    ]
    text = "\n".join(ordered_text for ordered_text in ordered_rows if ordered_text)
    truncated = len(text) > max_chars
    return (text[:max_chars], truncated)


# ----------------------------------------------------------------------
# Reader with privacy gates
# ----------------------------------------------------------------------
class ScreenReader:
    """Capture + OCR with the pause switch and the title blocklist in front."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        ocr: Any = None,
        grabber: Optional[Callable[..., Capture]] = None,
        window_lookup: Optional[Callable[[], Optional[Tuple[str, Dict[str, int]]]]] = None,
    ):
        full = config or {}
        screen = full.get("screen") if isinstance(full.get("screen"), dict) else full
        self.screen_cfg: Dict[str, Any] = dict(screen or {})

        self.enabled = bool(self.screen_cfg.get("enabled", True))
        self.region = str(self.screen_cfg.get("region", "window")).lower()
        self.max_width = int(self.screen_cfg.get("max_width", DEFAULT_MAX_WIDTH))
        self.max_chars = int(self.screen_cfg.get("max_chars", DEFAULT_MAX_CHARS))
        self.blocklist = [
            str(pattern).lower()
            for pattern in (
                self.screen_cfg.get("blocklist") or DEFAULT_BLOCKLIST
            )
            if str(pattern).strip()
        ]

        self._grab = grabber or grab
        self._window = window_lookup or active_window
        self._ocr = ocr or build_ocr(
            str(self.screen_cfg.get("ocr", "auto")),
            float(self.screen_cfg.get("min_confidence", DEFAULT_MIN_SCORE)),
        )
        self._paused = False
        self._pause_reason = ""
        self._lock = threading.Lock()
        self._warm_thread: Optional[threading.Thread] = None

    # -- state ---------------------------------------------------------
    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def pause_reason(self) -> str:
        return self._pause_reason

    @property
    def ocr_name(self) -> str:
        return str(getattr(self._ocr, "name", "ocr"))

    def pause(self, reason: str = "paused by you") -> None:
        self._paused = True
        self._pause_reason = reason
        _logger.info("Screen access paused (%s)", reason)

    def resume(self) -> None:
        self._paused = False
        self._pause_reason = ""
        _logger.info("Screen access resumed")

    def toggle(self) -> bool:
        """Flip the pause switch and return the new paused state."""
        if self._paused:
            self.resume()
        else:
            self.pause()
        return self._paused

    def blocked_reason(self, title: str) -> Optional[str]:
        lowered = (title or "").lower()
        for pattern in self.blocklist:
            if pattern and pattern in lowered:
                return f"the window title matches the protected pattern {pattern!r}"
        return None

    # -- reading -------------------------------------------------------
    def _target(self, region: str) -> Tuple[str, Dict[str, int], str]:
        """Resolve the capture area and its title before any pixels are read."""
        region = (region or self.region or "window").lower()
        if region == "window":
            found = self._window()
            if found:
                title, rect = found
                return title, rect, "window"
        return "", primary_monitor(), "monitor"

    def look(self, region: Optional[str] = None, max_chars: Optional[int] = None) -> Dict[str, Any]:
        """Capture, OCR and return plain text plus timings. Never writes a file."""
        if not self.enabled:
            raise ScreenError("screen reading is switched off in config.yaml")
        if self._paused:
            raise ScreenError(
                "screen access is paused, so I cannot look at the screen. "
                "Press Ctrl+Alt+P to allow it."
            )

        title, rect, used = self._target(region or self.region)
        blocked = self.blocked_reason(title)
        if blocked is not None:
            raise ScreenError(
                f"I will not read that window: {blocked}. Switch to it only if "
                "you want it read."
            )

        capture = self._grab(
            rect=rect,
            title=title,
            region=used,
            max_width=self.max_width,
        )
        started = time.perf_counter()
        lines = self._ocr.read(capture.image)
        ocr_ms = (time.perf_counter() - started) * 1000
        text, truncated = format_lines(lines, max_chars or self.max_chars)

        total_ms = capture.capture_ms + ocr_ms
        _logger.info(
            "Screen read: %d chars from %d lines (%s) in %.0f ms "
            "[capture %.0f ms, ocr %.0f ms]",
            len(text),
            len(lines),
            capture.title or "the monitor",
            total_ms,
            capture.capture_ms,
            ocr_ms,
        )
        return {
            "window": capture.title or "(monitor)",
            "region": capture.region,
            "text": text,
            "lines": len(lines),
            "truncated": truncated,
            "capture_ms": round(capture.capture_ms),
            "ocr_ms": round(ocr_ms),
            "total_ms": round(total_ms),
            "ocr_engine": self.ocr_name,
        }

    def capture_png(self, region: Optional[str] = None) -> Tuple[bytes, str]:
        """PNG bytes for the vision tool, still in memory only."""
        if not self.enabled:
            raise ScreenError("screen reading is switched off in config.yaml")
        if self._paused:
            raise ScreenError(
                "screen access is paused, so I cannot look at the screen. "
                "Press Ctrl+Alt+P to allow it."
            )

        title, rect, used = self._target(region or self.region)
        blocked = self.blocked_reason(title)
        if blocked is not None:
            raise ScreenError(f"I will not read that window: {blocked}.")

        capture = self._grab(
            rect=rect, title=title, region=used, max_width=self.max_width
        )
        import io as _io

        from PIL import Image

        buffer = _io.BytesIO()
        Image.fromarray(capture.image[:, :, ::-1]).save(buffer, format="PNG")
        return buffer.getvalue(), capture.title

    # -- warm-up -------------------------------------------------------
    def warm(self) -> None:
        """Load the OCR model in the background, away from the first question."""
        if not self.enabled or self._warm_thread is not None:
            return

        def _run() -> None:
            started = time.perf_counter()
            try:
                self._ocr.warm()
                _logger.debug(
                    "OCR warm-up (%s) finished in %.0f ms",
                    self.ocr_name,
                    (time.perf_counter() - started) * 1000,
                )
            except Exception as exc:  # noqa: BLE001 - OCR loads lazily anyway
                _logger.warning("OCR warm-up failed: %s", exc)

        self._warm_thread = threading.Thread(
            target=_run, name="ocr-warmup", daemon=True
        )
        self._warm_thread.start()


# ----------------------------------------------------------------------
# Optional rolling history (off by default, RAM only)
# ----------------------------------------------------------------------
class ScreenHistory:
    """Keeps the last few minutes of OCR text in RAM when it is enabled.

    A cheap downscaled frame comparison decides whether the screen actually
    changed, so an idle desktop costs a few milliseconds per interval.
    """

    def __init__(
        self,
        reader: ScreenReader,
        interval_s: float = 3.0,
        keep_s: float = 180.0,
        threshold: float = 3.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        full = config or {}
        history = full.get("history") if isinstance(full.get("history"), dict) else full
        history = history or {}

        self.reader = reader
        self.enabled = bool(history.get("enabled", False))
        self.interval_s = max(1.0, float(history.get("interval_s", interval_s)))
        self.keep_s = max(self.interval_s, float(history.get("keep_s", keep_s)))
        self.threshold = float(history.get("change_threshold", threshold))

        self._entries: List[Tuple[float, str]] = []
        self._signature: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="screen-history", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def recent(self, seconds: Optional[float] = None) -> List[Tuple[float, str]]:
        cutoff = time.time() - float(seconds or self.keep_s)
        return [(stamp, text) for stamp, text in self._entries if stamp >= cutoff]

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self._capture_once()
            except ScreenError as exc:
                _logger.debug("Screen history skipped a frame: %s", exc)
            except Exception as exc:  # noqa: BLE001 - never kill the thread
                _logger.debug("Screen history error: %s", exc)

    def _signature_of(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image

        small = Image.fromarray(image).convert("L").resize((160, 100), Image.BILINEAR)
        return np.asarray(small, dtype=np.float32)

    def _changed(self, image: np.ndarray) -> bool:
        signature = self._signature_of(image)
        if self._signature is None:
            self._signature = signature
            return True
        difference = float(np.mean(np.abs(signature - self._signature)))
        if difference >= self.threshold:
            self._signature = signature
            return True
        return False

    def _capture_once(self) -> None:
        if self.reader.paused or not self.reader.enabled:
            return
        title, rect, region = self.reader._target(self.reader.region)
        if self.reader.blocked_reason(title):
            return
        capture = self.reader._grab(
            rect=rect, title=title, region=region, max_width=self.reader.max_width
        )
        if not self._changed(capture.image):
            return
        result = self.reader.look()
        if result["text"]:
            self._entries.append((time.time(), result["text"]))
        cutoff = time.time() - self.keep_s
        self._entries = [item for item in self._entries if item[0] >= cutoff]


# ----------------------------------------------------------------------
# Pause hotkey
# ----------------------------------------------------------------------
class PauseHotkey:
    """Ctrl+Alt+P toggles screen access; mirrors the tray icon state."""

    def __init__(
        self,
        reader: ScreenReader,
        hotkey: str = DEFAULT_PAUSE_HOTKEY,
        on_change: Optional[Callable[[bool], None]] = None,
    ):
        self.reader = reader
        self.hotkey = hotkey or DEFAULT_PAUSE_HOTKEY
        self.on_change = on_change
        self._listener = None

    def start(self) -> None:
        if self._listener is not None:
            return
        from pynput import keyboard

        self._listener = keyboard.GlobalHotKeys({self.hotkey: self._toggle})
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _toggle(self) -> None:
        paused = self.reader.toggle()
        state = "paused" if paused else "active"
        print(f"\n[screen] access {state} - press {self.hotkey} to change")
        if self.on_change:
            self.on_change(paused)
