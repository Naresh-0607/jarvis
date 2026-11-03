"""Tests for the view-only screen reader (Phase 5)."""

from __future__ import annotations

import asyncio
import json
import types
from typing import Any, List, Optional

import numpy as np
import pytest

from jarvis import screen_tools
from jarvis.screen import (
    DEFAULT_BLOCKLIST,
    Capture,
    OcrLine,
    PauseHotkey,
    ScreenError,
    ScreenHistory,
    ScreenReader,
    format_lines,
)
from jarvis.tray import ScreenTray


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeOcr:
    name = "fake"

    def __init__(self, lines: Optional[List[OcrLine]] = None):
        self.lines = lines if lines is not None else [
            OcrLine("ValueError: bad input", 0.98, 10, 40, 200, 18),
            OcrLine("File app.py, line 42", 0.95, 10, 10, 220, 18),
        ]
        self.calls = 0
        self.warmed = 0

    def warm(self) -> None:
        self.warmed += 1

    def read(self, image: np.ndarray) -> List[OcrLine]:
        self.calls += 1
        return list(self.lines)


def white_image(height: int = 60, width: int = 160) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def make_reader(
    title: str = "Terminal - python app.py",
    ocr: Optional[FakeOcr] = None,
    grabbed: Optional[List[dict]] = None,
    config: Optional[dict] = None,
) -> ScreenReader:
    ocr = ocr or FakeOcr()

    def fake_grab(rect=None, title="", region="monitor", max_width=1600):
        if grabbed is not None:
            grabbed.append({"rect": rect, "title": title, "region": region})
        return Capture(
            image=white_image(), region=region, title=title, capture_ms=12.0
        )

    return ScreenReader(
        config or {},
        ocr=ocr,
        grabber=fake_grab,
        window_lookup=lambda: (title, {"left": 0, "top": 0, "width": 800, "height": 600}),
    )


# ----------------------------------------------------------------------
# Reading order
# ----------------------------------------------------------------------
def test_format_lines_orders_rows_then_words():
    lines = [
        OcrLine("world", 0.9, 80, 40, 40, 14),
        OcrLine("second", 0.9, 10, 40, 60, 14),
        OcrLine("first", 0.9, 10, 10, 40, 14),
    ]
    text, truncated = format_lines(lines)
    assert text == "first\nsecond world"
    assert truncated is False


def test_format_lines_truncates_long_text():
    line = OcrLine("x" * 50, 0.9, 0, 0, 50, 14)
    text, truncated = format_lines([line, line, line], max_chars=60)
    assert len(text) == 60
    assert truncated is True


def test_format_lines_is_empty_for_no_text():
    assert format_lines([]) == ("", False)


# ----------------------------------------------------------------------
# Privacy gates
# ----------------------------------------------------------------------
def test_blocked_reason_matches_case_insensitive_substring():
    reader = make_reader()
    assert reader.blocked_reason("1Password - Unlock") is not None
    assert reader.blocked_reason("Online Banking - HDFC Bank") is not None
    assert reader.blocked_reason("Terminal - python app.py") is None


def test_look_refuses_before_capturing_a_blocked_window():
    grabbed: List[dict] = []
    reader = make_reader(title="Bitwarden - Vault", grabbed=grabbed)
    with pytest.raises(ScreenError, match="will not read"):
        reader.look()
    assert grabbed == []  # no pixels were read at all


def test_look_refuses_when_paused_and_recovers_after_resume():
    reader = make_reader()
    reader.pause()
    with pytest.raises(ScreenError, match="paused"):
        reader.look()
    assert reader.paused is True

    reader.resume()
    assert reader.paused is False
    assert reader.look()["text"]


def test_toggle_flips_pause_state():
    reader = make_reader()
    assert reader.paused is False
    assert reader.toggle() is True
    assert reader.toggle() is False


def test_look_refuses_when_disabled():
    reader = make_reader(config={"screen": {"enabled": False}})
    with pytest.raises(ScreenError, match="switched off"):
        reader.look()


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
def test_look_returns_text_and_timings():
    reader = make_reader()
    result = reader.look()

    assert "ValueError: bad input" in result["text"]
    assert result["lines"] == 2
    assert result["capture_ms"] == 12
    assert result["ocr_ms"] >= 0
    assert result["total_ms"] >= result["capture_ms"]
    assert result["ocr_engine"] == "fake"
    assert result["region"] == "window"


def test_look_uses_monitor_when_no_window_is_usable():
    reader = make_reader()
    reader._window = lambda: None
    result = reader.look()
    assert result["region"] == "monitor"
    assert result["window"] == "(monitor)"


def test_capture_png_is_png_in_memory():
    reader = make_reader()
    png, title = reader.capture_png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert title == "Terminal - python app.py"


def test_max_chars_caps_the_text():
    lines = [OcrLine("y" * 40, 0.9, 0, index * 20, 40, 14) for index in range(20)]
    reader = make_reader(ocr=FakeOcr(lines), config={"screen": {"max_chars": 50}})
    result = reader.look()
    assert len(result["text"]) == 50
    assert result["truncated"] is True


# ----------------------------------------------------------------------
# Rolling history (off by default)
# ----------------------------------------------------------------------
def test_history_does_nothing_when_disabled():
    reader = make_reader()
    history = ScreenHistory(reader, config={"history": {"enabled": False}})
    history.start()
    assert history._thread is None
    assert history.recent() == []


def test_history_skips_frames_that_did_not_change():
    ocr = FakeOcr()
    reader = make_reader(ocr=ocr)
    history = ScreenHistory(
        reader, config={"history": {"enabled": True, "interval_s": 1}}
    )

    history._capture_once()
    history._capture_once()  # identical pixels, so no second OCR run

    assert ocr.calls == 1
    assert len(history.recent()) == 1


def test_history_respects_pause():
    ocr = FakeOcr()
    reader = make_reader(ocr=ocr)
    reader.pause()
    history = ScreenHistory(reader, config={"history": {"enabled": True}})
    history._capture_once()
    assert ocr.calls == 0
    assert history.recent() == []


# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------
def test_wants_screen_matches_common_phrases():
    assert screen_tools.wants_screen("why am I getting this error?")
    assert screen_tools.wants_screen("What's on my screen right now?")
    assert screen_tools.wants_screen("Fix this code for me")
    assert not screen_tools.wants_screen("what's the weather tomorrow")


def test_look_at_screen_tool_returns_a_refusal_instead_of_raising(monkeypatch):
    reader = make_reader()
    reader.pause()
    monkeypatch.setattr(screen_tools, "get_reader", lambda: reader)

    result = screen_tools.look_at_screen()
    assert "paused" in result["error"]
    assert result["paused"] is True


def test_look_at_screen_tool_returns_text(monkeypatch):
    monkeypatch.setattr(screen_tools, "get_reader", lambda: make_reader())
    result = screen_tools.look_at_screen()
    assert "ValueError" in result["text"]


def test_look_at_screen_says_when_there_is_no_text(monkeypatch):
    monkeypatch.setattr(
        screen_tools, "get_reader", lambda: make_reader(ocr=FakeOcr([]))
    )
    result = screen_tools.look_at_screen()
    assert result["text"] == ""
    assert "No readable text" in result["note"]


def test_vision_is_off_without_a_key_or_model(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert screen_tools.vision_configured({"vision": {"enabled": True, "model": "x"}}) is False
    assert (
        screen_tools.vision_configured({"vision": {"enabled": False, "model": "x"}})
        is False
    )


def test_vision_is_on_with_key_and_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    assert (
        screen_tools.vision_configured(
            {"vision": {"enabled": True, "model": "gemini-flash"}}
        )
        is True
    )


def test_describe_screen_image_explains_it_is_not_configured(monkeypatch):
    monkeypatch.setattr(screen_tools, "get_reader", lambda: make_reader())
    monkeypatch.setattr(screen_tools, "vision_configured", lambda config=None: False)

    result = screen_tools.describe_screen_image("what is this?")
    assert "GEMINI_API_KEY" in result["error"]


def test_describe_image_sends_an_openai_vision_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "A bar chart."}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    monkeypatch.setattr(screen_tools.httpx, "post", fake_post)

    config = {
        "vision": {
            "enabled": True,
            "model": "gemini-flash",
            "api_key_env": "GEMINI_API_KEY",
        }
    }
    answer = screen_tools.describe_image(b"\x89PNGdata", "what is this?", config)

    assert answer == "A bar chart."
    assert captured["url"].endswith("/chat/completions")
    part = captured["payload"]["messages"][0]["content"][1]
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_describe_image_reports_an_http_error(monkeypatch):
    class FakeResponse:
        status_code = 429
        text = "rate limited"

    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    monkeypatch.setattr(
        screen_tools.httpx, "post", lambda *a, **k: FakeResponse()
    )
    with pytest.raises(ScreenError, match="HTTP 429"):
        screen_tools.describe_image(
            b"x", "q", {"vision": {"enabled": True, "model": "m"}}
        )


# ----------------------------------------------------------------------
# Tray and hotkey
# ----------------------------------------------------------------------
def test_tray_colour_changes_when_paused():
    reader = make_reader()
    tray = ScreenTray(reader)
    active = np.asarray(tray._image())
    reader.pause()
    paused = np.asarray(tray._image())
    assert active.shape == paused.shape
    assert not np.array_equal(active, paused)


def test_hotkey_toggle_updates_state_and_notifies(monkeypatch):
    reader = make_reader()
    seen = []
    hotkey = PauseHotkey(reader, on_change=seen.append)

    hotkey._toggle()
    assert reader.paused is True
    hotkey._toggle()
    assert reader.paused is False
    assert seen == [True, False]


# ----------------------------------------------------------------------
# Chat wiring
# ----------------------------------------------------------------------
def test_chat_hides_the_vision_tool_until_it_is_configured(monkeypatch):
    from jarvis.chat import ChatSession

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    session = ChatSession()
    session.cfg = {"vision": {"enabled": False}}
    names = [tool["function"]["name"] for tool in session._tools_payload()]
    assert "look_at_screen" in names
    assert "describe_screen_image" not in names

    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    session.cfg = {"vision": {"enabled": True, "model": "gemini-flash"}}
    names = [tool["function"]["name"] for tool in session._tools_payload()]
    assert "describe_screen_image" in names
