"""Line input that also accepts transcripts from the hotkey thread.

On Windows we need a character-by-character reader, because `input()` blocks
and a transcript that arrives while the user is idle would otherwise sit
unheard until they pressed Enter. Non-Windows falls back to `input()`.
"""

from __future__ import annotations

import sys
import time
from typing import Optional, Tuple


def read_line(
    prompt: str = "\n> ",
    transcripts=None,
    poll: float = 0.05,
) -> Tuple[Optional[str], str]:
    """Return (text, source) where source is "type", "voice" or "" for EOF.

    ``text`` is None on end-of-input.
    """
    if transcripts is None or not sys.platform.startswith("win") or not sys.stdin.isatty():
        try:
            return input(prompt), "type"
        except EOFError:
            return None, ""

    return _read_line_windows(prompt, transcripts, poll)


def _read_line_windows(prompt: str, transcripts, poll: float) -> Tuple[Optional[str], str]:
    import msvcrt

    sys.stdout.write(prompt)
    sys.stdout.flush()

    buffer = ""
    while True:
        text = transcripts.get_nowait()
        if text:
            if buffer:
                sys.stdout.write("\n")
                print("  (typed text discarded)")
            print(f"[voice] {text}")
            return text, "voice"

        if msvcrt.kbhit():
            char = msvcrt.getwch()

            if char in ("\r", "\n"):
                print()
                return buffer, "type"

            if char == "\x08":  # backspace
                if buffer:
                    buffer = buffer[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue

            if char == "\x03":  # Ctrl+C
                raise KeyboardInterrupt

            if char in ("\x00", "\xe0"):  # arrow / function keys
                msvcrt.getwch()
                continue

            if char == "\x1b":  # Escape
                continue

            if char.isprintable():
                buffer += char
                sys.stdout.write(char)
                sys.stdout.flush()
            continue

        time.sleep(poll)
