"""ASCII User Interface for Jarvis with animated elements."""

import sys
import time
import threading
from typing import Optional


class AsciiUI:
    """Provides ASCII animations and visual feedback for the chat interface."""
    
    def __init__(self):
        self.is_listening = False
        self.is_thinking = False
        self.animation_thread: Optional[threading.Thread] = None
        self.stop_animation = threading.Event()
        self.current_animation = ""
        
    def start_listening_animation(self):
        """Start the listening animation."""
        if not self.is_listening:
            self.is_listening = True
            self.stop_animation.clear()
            self.animation_thread = threading.Thread(target=self._listening_animation, daemon=True)
            self.animation_thread.start()
            
    def stop_listening_animation(self):
        """Stop the listening animation."""
        self.is_listening = False
        self.stop_animation.set()
        if self.animation_thread:
            self.animation_thread.join(timeout=1)
            # Clear line after thread finishes
            sys.stdout.write("\r" + " " * 30 + "\r")
            sys.stdout.flush()
            
    def start_thinking_animation(self):
        """Start the thinking animation."""
        if not self.is_thinking:
            self.is_thinking = True
            self.stop_animation.clear()
            self.animation_thread = threading.Thread(target=self._thinking_animation, daemon=True)
            self.animation_thread.start()
            
    def stop_thinking_animation(self):
        """Stop the thinking animation."""
        self.is_thinking = False
        self.stop_animation.set()
        if self.animation_thread:
            self.animation_thread.join(timeout=1)
            # Clear line after thread finishes
            sys.stdout.write("\r" + " " * 30 + "\r")
            sys.stdout.flush()
            
    def _listening_animation(self):
        """Display listening animation."""
        chars = ["●", "◎", "◉", "◌"]
        idx = 0
        
        while not self.stop_animation.is_set() and self.is_listening:
            sys.stdout.write(f"\rListening... {chars[idx]}")
            sys.stdout.flush()
            idx = (idx + 1) % len(chars)
            time.sleep(0.3)
            
    def _thinking_animation(self):
        """Display thinking animation."""
        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = 0
        
        while not self.stop_animation.is_set() and self.is_thinking:
            sys.stdout.write(f"\rThinking... {chars[idx]}")
            sys.stdout.flush()
            idx = (idx + 1) % len(chars)
            time.sleep(0.2)
    
    def clear_screen(self):
        """Clear the terminal screen."""
        print("\033[2J\033[H", end="")
        
    def draw_jarvis_logo(self):
        """Draw an ASCII logo for Jarvis."""
        logo = r"""
         ╔══════════════════════════════════════════════════╗
         ║     █████╗ ██╗   ██╗██╗███████╗██████╗ ██████╗   ║
         ║    ██╔══██╗██║   ██║██║██╔════╝██╔══██╗██╔══██╗  ║
         ║    ███████║██║   ██║██║███████╗██████╔╝██║  ██║  ║
         ║    ██╔══██╗╚██╗ ██╔╝██║╚════██║██╔═══╝ ██║  ██║  ║
         ║    ██║  ██║ ╚████╔╝ ██║███████║██║     ██████╔╝  ║
         ║    ╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝╚═╝     ╚═════╝   ║
         ║                                                   ║
         ║    Personal AI Assistant                          ║
         ╚══════════════════════════════════════════════════╝
        """
        print(logo)
        
    def draw_user_prompt(self):
        """Draw a stylized user input prompt."""
        print("┌──────────────────────────────────────────────┐")
        print("│  You:                                        │")
        print("└──────────────────────────────────────────────┘")
        
    def draw_jarvis_response(self):
        """Draw a stylized jarvis response box."""
        print("┌──────────────────────────────────────────────┐")
        print("│  Jarvis:                                     │")
        print("└──────────────────────────────────────────────┘")
        
    def animate_text(self, text: str, delay: float = 0.02):
        """Animate the typing of text."""
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        sys.stdout.write("\n")
        sys.stdout.flush()
        
    def print_status_line(self, message: str):
        """Print a status line with a nice format."""
        print(f"[>] {message}")
        
    def print_error(self, message: str):
        """Print an error message with red formatting."""
        print(f"\033[91m[ERROR] {message}\033[0m")
        
    def print_info(self, message: str):
        """Print an info message with blue formatting."""
        print(f"\033[94m[INFO] {message}\033[0m")
        
    def print_success(self, message: str):
        """Print a success message with green formatting."""
        print(f"\033[92m[SUCCESS] {message}\033[0m")