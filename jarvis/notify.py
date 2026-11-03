"""Reminder delivery: console now, voice later, plus ntfy.sh push."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import httpx

from jarvis.db import PROJECT_ROOT

_logger = logging.getLogger("jarvis.notify")

NTFY_BASE = "https://ntfy.sh"


def _env_topic() -> str:
    value = os.environ.get("NTFY_TOPIC", "")
    if not value.strip():
        # The scheduler runs on a background thread and may be the first thing
        # to need this, so do not assume the chat already loaded .env.
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
            value = os.environ.get("NTFY_TOPIC", "")
        except Exception:  # noqa: BLE001 - optional dependency at runtime
            _logger.debug("Could not load .env for NTFY_TOPIC")
    return value


def ntfy_url(topic: Optional[str] = None) -> Optional[str]:
    """Normalise NTFY_TOPIC: accepts a bare topic or a full https URL."""
    value = (topic if topic is not None else _env_topic()).strip()
    if not value:
        return None
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"{NTFY_BASE}/{value.lstrip('/')}"


def send_ntfy(message: str, topic: Optional[str] = None, timeout: float = 8.0) -> bool:
    """POST a push notification. Never raises; returns success."""
    url = ntfy_url(topic)
    if not url:
        _logger.info("NTFY_TOPIC not set; skipping push")
        return False

    try:
        response = httpx.post(
            url,
            content=message.encode("utf-8"),
            headers={"Title": "Jarvis", "Tags": "alarm_clock"},
            timeout=timeout,
        )
        response.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001 - delivery must never break a task
        _logger.warning("ntfy push failed: %s", exc)
        return False


def deliver(message: str, speak: bool = False) -> None:
    """Console message (and, in later phases, speech) plus a phone push."""
    print(f"\n[reminder] {message}")
    if speak:
        # Voice lands in Phase 4; this hook keeps the call site stable.
        _logger.debug("speech not wired yet")
    send_ntfy(message)
