#!/usr/bin/env python3
"""Test script for the ASCII UI module."""

import time
from jarvis.ascii_ui import AsciiUI

def test_ascii_ui():
    """Test the ASCII UI animations."""
    ui = AsciiUI()
    
    print("Testing ASCII UI...")
    
    # Test listening animation
    print("\nStarting listening animation (press Ctrl+C to stop):")
    ui.start_listening_animation()
    time.sleep(3)
    ui.stop_listening_animation()
    
    # Test thinking animation  
    print("\nStarting thinking animation (press Ctrl+C to stop):")
    ui.start_thinking_animation()
    time.sleep(3)
    ui.stop_thinking_animation()
    
    # Test logo
    print("\nDisplaying logo:")
    ui.draw_jarvis_logo()
    
    # Test text animation
    print("\nAnimating text:")
    ui.animate_text("Hello, this is an animated message!", delay=0.05)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_ascii_ui()