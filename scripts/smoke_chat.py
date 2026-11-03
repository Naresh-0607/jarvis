"""Manual Phase 1 smoke test.

Runs a few turns through the real provider chain and prints which
provider/model answered each one.

    python scripts/smoke_chat.py
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from jarvis.chat import ChatSession
from jarvis.router import choose_tier

PROMPTS = [
    "hey, how are you?",
    "Why does a rate-limited model need a cooldown before it is retried?",
    "Use the get_demo_number tool to look up asha, then tell me the number.",
]


async def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    session = ChatSession()
    session.setup_providers(mode="online")
    print(f"Chain: {session.provider_chain}\n")

    failures = 0
    for prompt in PROMPTS:
        print(f"> {prompt}")
        print(f"  tier chosen: {session.tier_override or choose_tier(prompt)}")
        print("  Jarvis: ", end="", flush=True)
        try:
            async for chunk in session.get_response_stream(prompt):
                print(chunk, end="", flush=True)
            print()
            print(
                f"  answered by: {session.current_provider_name}"
                f"/{session.current_model_name}\n"
            )
        except Exception as exc:  # noqa: BLE001 - smoke script
            failures += 1
            print(f"\n  FAILED: {exc}\n")

    await session.provider_chain.aclose()
    print("Smoke test finished." if not failures else f"{failures} turn(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
