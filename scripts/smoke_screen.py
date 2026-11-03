"""Manual check for Phase 5: capture, OCR, pause and the blocklist.

    python scripts\\smoke_screen.py              # read the front window
    python scripts\\smoke_screen.py --monitor    # read the whole screen
    python scripts\\smoke_screen.py --blocked    # prove the blocklist refuses

Hold a terminal or browser with an error on screen, then run it in another
window only if you want the monitor variant; the default reads whatever is in
front, which is the terminal you launched it from.
"""

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from jarvis.screen import ScreenError, ScreenReader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor", action="store_true", help="read the whole screen")
    parser.add_argument("--blocked", action="store_true", help="test the blocklist")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    reader = ScreenReader()

    if args.blocked:
        title = "1Password - Unlock"
        reason = reader.blocked_reason(title)
        print(f"blocklist for {title!r}: {reason or 'NOT BLOCKED (bad)'}")
        return 0 if reason else 1

    print(f"OCR engine: {reader.ocr_name}")
    print("warming the model ...")
    started = time.perf_counter()
    reader._ocr.warm()
    print(f"warm in {(time.perf_counter() - started) * 1000:.0f} ms")

    region = "monitor" if args.monitor else "window"
    started = time.perf_counter()
    try:
        result = reader.look(region=region)
    except ScreenError as exc:
        print(f"refused: {exc}")
        return 1

    total = (time.perf_counter() - started) * 1000
    print(f"\nwindow: {result['window']}")
    print(
        f"capture {result['capture_ms']} ms | ocr {result['ocr_ms']} ms | "
        f"{result['lines']} lines | {len(result['text'])} chars | {total:.0f} ms total"
    )
    print("-" * 60)
    print(result["text"] or "(no readable text)")
    print("-" * 60)

    reader.pause()
    try:
        reader.look(region=region)
    except ScreenError as exc:
        print(f"\npaused correctly: {exc}")
    else:
        print("\npause FAILED: it still read the screen")
        return 1

    reader.resume()
    print("resumed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
