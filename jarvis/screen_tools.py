"""LLM tools for the view-only screen (Phase 5).

Two tools, both read-only and both refusing when screen access is paused or the
window title is on the blocklist:

* ``look_at_screen`` captures the active window, runs OCR and returns text.
* ``describe_screen_image`` sends the capture to a cloud vision model, and is
  only used when you ask about layout, a diagram, a chart or a photo.

The screenshot never touches the disk: it is captured, base64-encoded in
memory, and dropped.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx
import yaml

from jarvis.db import PROJECT_ROOT
from jarvis.screen import ScreenError, ScreenReader
from jarvis.tools import tool

_logger = logging.getLogger("jarvis.screen_tools")

_reader: Optional[ScreenReader] = None
_reader_config: Optional[Dict[str, Any]] = None

# Phrases that force a screen read even if the model forgets to call the tool.
SCREEN_PHRASES = (
    "this error",
    "that error",
    "the error",
    "this traceback",
    "that traceback",
    "my screen",
    "on screen",
    "on my screen",
    "this code",
    "that code",
    "what am i looking at",
    "look at the screen",
    "look at my screen",
    "what does this say",
    "why am i getting",
)

VISION_DEFAULTS = {
    "enabled": False,
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    "api_key_env": "GEMINI_API_KEY",
    "model": "",
    "timeout_s": 30,
}


def configure(config: Optional[Dict[str, Any]] = None) -> ScreenReader:
    """Point the tools at a config (tests call this). Rebuilds the reader."""
    global _reader, _reader_config
    _reader_config = config or {}
    _reader = ScreenReader(_reader_config)
    return _reader


def get_reader() -> ScreenReader:
    global _reader, _reader_config
    if _reader is None:
        config: Dict[str, Any] = _reader_config or {}
        if not config:
            cfg_path = PROJECT_ROOT / "config.yaml"
            if cfg_path.exists():
                with cfg_path.open("rt", encoding="utf8") as handle:
                    config = yaml.safe_load(handle) or {}
        _reader = ScreenReader(config)
        _reader_config = config
    return _reader


def wants_screen(text: str) -> bool:
    """True when a sentence should always trigger a screen read."""
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in SCREEN_PHRASES)


def vision_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    full = config if config is not None else (_reader_config or {})
    block = full.get("vision") if isinstance(full.get("vision"), dict) else {}
    settings = dict(VISION_DEFAULTS)
    settings.update(block or {})
    return settings


def vision_configured(config: Optional[Dict[str, Any]] = None) -> bool:
    settings = vision_settings(config)
    if not settings.get("enabled"):
        return False
    if not str(settings.get("model") or "").strip():
        return False
    if not str(settings.get("base_url") or "").strip():
        return False
    return bool(os.environ.get(str(settings.get("api_key_env") or ""), "").strip())


def describe_image(
    png: bytes, question: str, config: Optional[Dict[str, Any]] = None
) -> str:
    """Send one in-memory PNG to the configured vision model."""
    settings = vision_settings(config)
    key_name = str(settings.get("api_key_env") or "")
    key = os.environ.get(key_name, "").strip()
    if not key:
        raise ScreenError(
            f"vision is not configured: put {key_name} in .env and set "
            "vision.model in config.yaml"
        )

    base_url = str(settings["base_url"]).rstrip("/")
    prompt = (
        (question or "Describe what is on this screen.").strip()
        + "\nAnswer in two or three spoken sentences. Mention layout, diagrams, "
        "charts or photos only where they matter."
    )
    payload = {
        "model": settings["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(png).decode("ascii")
                        },
                    },
                ],
            }
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }

    started = time.perf_counter()
    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=float(settings.get("timeout_s", 30)),
        )
    except httpx.HTTPError as exc:
        raise ScreenError(f"the vision model could not be reached: {exc}") from exc

    if response.status_code != 200:
        raise ScreenError(
            f"the vision model returned HTTP {response.status_code}: "
            f"{response.text[:160]}"
        )

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise ScreenError("the vision model returned an unexpected reply") from exc

    _logger.info(
        "Vision description: %d chars in %.0f ms",
        len(str(content)),
        (time.perf_counter() - started) * 1000,
    )
    return str(content).strip()


LOOK_SCHEMA = {
    "name": "look_at_screen",
    "description": (
        "Read the text of the window that is in front right now (view-only, "
        "OCR). Call it whenever the user mentions an error, a traceback, code "
        "or something on their screen. It also reads the whole monitor when "
        "region is 'monitor'. It cannot click, type or change anything."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "region": {
                "type": "string",
                "enum": ["active", "monitor"],
                "description": "The front window (default) or the whole primary monitor",
                "default": "active",
            }
        },
        "required": [],
    },
}

DESCRIBE_SCHEMA = {
    "name": "describe_screen_image",
    "description": (
        "Send the current screen to a cloud vision model and describe it. Use "
        "only when the user asks about layout, a diagram, a chart or a photo, "
        "not for plain text or errors (look_at_screen already reads those)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "What the user wants to know about the image",
            }
        },
        "required": [],
    },
}


@tool(name="look_at_screen", schema=LOOK_SCHEMA)
def look_at_screen(region: str = "active") -> Dict[str, Any]:
    """Capture the active window (or monitor), OCR it, return plain text."""
    reader = get_reader()
    wanted = "monitor" if str(region).lower() == "monitor" else "window"
    try:
        result = reader.look(region=wanted)
    except ScreenError as exc:
        _logger.info("Screen read refused: %s", exc)
        return {"error": str(exc), "paused": reader.paused}

    if not result["text"].strip():
        return {
            "window": result["window"],
            "text": "",
            "note": (
                "No readable text was found in that window. It may be an image, "
                "a game or mostly empty; ask the user what they see."
            ),
            "total_ms": result["total_ms"],
        }
    return result


@tool(name="describe_screen_image", schema=DESCRIBE_SCHEMA)
def describe_screen_image(question: str = "") -> Dict[str, Any]:
    """Describe the current screen with the configured cloud vision model."""
    reader = get_reader()
    if not vision_configured():
        return {
            "error": (
                "Vision is not switched on. Add GEMINI_API_KEY to .env, set "
                "vision.model in config.yaml and vision.enabled: true. For now "
                "I can only read text with look_at_screen."
            )
        }

    started = time.perf_counter()
    try:
        png, title = reader.capture_png()
        description = describe_image(png, question)
    except ScreenError as exc:
        _logger.info("Vision refused: %s", exc)
        return {"error": str(exc), "paused": reader.paused}

    return {
        "window": title or "(monitor)",
        "description": description,
        "total_ms": round((time.perf_counter() - started) * 1000),
    }


def screen_context(question: str) -> Dict[str, Any]:
    """Run a screen read for chat's rule-based fallback, with a spoken hint."""
    reader = get_reader()
    started = time.perf_counter()
    try:
        result = reader.look()
    except ScreenError as exc:
        return {"ok": False, "reason": str(exc), "total_ms": 0}

    return {
        "ok": True,
        "window": result["window"],
        "text": result["text"],
        "truncated": result["truncated"],
        "total_ms": round((time.perf_counter() - started) * 1000),
    }
