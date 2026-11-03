"""A small tray icon that shows whether screen access is active (Phase 5).

One icon, two colours: green when Jarvis may read the screen, amber when you
have paused it. Clicking the icon toggles the same switch as Ctrl+Alt+P.
Phase 9 grows this into the full menu (status, pause listening, logs, quit).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

_logger = logging.getLogger("jarvis.tray")

ACTIVE_COLOUR = (46, 204, 113)  # green
PAUSED_COLOUR = (241, 196, 15)  # amber


class ScreenTray:
    """Tray icon for screen access. Safe to call even when pystray is missing."""

    def __init__(
        self,
        reader: Any,
        on_quit: Optional[Callable[[], None]] = None,
        on_toggle_window: Optional[Callable[[], None]] = None,
        is_window_visible: Optional[Callable[[], bool]] = None,
    ):
        self.reader = reader
        self.on_quit = on_quit
        self.on_toggle_window = on_toggle_window
        self.is_window_visible = is_window_visible
        self._icon = None
        self._pystray = None
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        if self._icon is not None:
            return
        try:
            import pystray
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "the tray icon needs pystray; run 'pip install pystray Pillow'"
            ) from exc

        self._pystray = pystray
        items = []
        if self.on_toggle_window is not None:
            items.append(pystray.MenuItem(self._window_label, self._toggle_window, default=True))
        items.append(pystray.MenuItem(self._label, self._toggle, default=(self.on_toggle_window is None)))
        if self.on_quit is not None:
            items.append(pystray.MenuItem("Quit Jarvis", self._quit))
        self._icon = pystray.Icon(
            "jarvis-screen", self._image(), "Jarvis", menu=pystray.Menu(*items)
        )
        self._thread = threading.Thread(
            target=self._icon.run, name="tray-icon", daemon=True
        )
        self._thread.start()
        _logger.debug("Screen tray icon started")

    def stop(self) -> None:
        icon = self._icon
        self._icon = None
        if icon is not None:
            try:
                icon.stop()
            except Exception as exc:  # noqa: BLE001 - shutting down anyway
                _logger.debug("Tray icon stop failed: %s", exc)

    # -- menu ----------------------------------------------------------
    def _window_label(self, _item: Any = None) -> str:
        if self.is_window_visible and self.is_window_visible():
            return "Hide Jarvis"
        return "Show Jarvis"

    def _toggle_window(self, _icon: Any = None, _item: Any = None) -> None:
        if self.on_toggle_window is not None:
            self.on_toggle_window()
        self.refresh()

    def _label(self, _item: Any = None) -> str:
        return "Screen access: paused" if self.reader.paused else "Screen access: active"

    def _toggle(self, _icon: Any = None, _item: Any = None) -> None:
        self.reader.toggle()
        self.refresh()

    def _quit(self, _icon: Any = None, _item: Any = None) -> None:
        if self.on_quit is not None:
            self.on_quit()

    def refresh(self) -> None:
        """Re-draw the icon after the pause state changed."""
        icon = self._icon
        if icon is None:
            return
        try:
            icon.icon = self._image()
            icon.update_menu()
        except Exception as exc:  # noqa: BLE001 - a dead tray must not break chat
            _logger.debug("Could not refresh the tray icon: %s", exc)

    # -- drawing -------------------------------------------------------
    def _image(self):
        from PIL import Image, ImageDraw

        colour = PAUSED_COLOUR if self.reader.paused else ACTIVE_COLOUR
        image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle([6, 10, 58, 44], radius=6, fill=colour)
        draw.rectangle([24, 46, 40, 52], fill=colour)
        draw.rectangle([16, 52, 48, 56], fill=colour)
        if self.reader.paused:
            # A simple slash marks the paused state even at 16x16.
            draw.line([8, 8, 56, 56], fill=(60, 60, 60), width=6)
        return image
