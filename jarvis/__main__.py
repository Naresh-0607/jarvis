"""Entry point for Jarvis assistant."""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from jarvis.chat import main as chat_main
from jarvis.doctor import main as doctor_main

def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        doctor_main()
    elif "--cli" in sys.argv:
        # Run legacy terminal interface for debugging
        sys.argv.remove("--cli")
        chat_main()
    else:
        # Default to new full-screen terminal TUI
        from jarvis.tui import main as tui_main
        tui_main()

if __name__ == "__main__":
    main()
