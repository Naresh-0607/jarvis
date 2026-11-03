#!/usr/bin/env python3
"""Example usage of the Jarvis assistant."""

import asyncio
import sys
from pathlib import Path

# Add project root to path 
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from jarvis.assistant import assistant

async def main():
    """Simple example of using the assistant."""
    print("Initializing Jarvis assistant...")
    
    try:
        # Initialize the assistant
        assistant.initialize()
        print("✓ Assistant initialized successfully")
        
        # Get provider status
        status = assistant.get_provider_status()
        print("Provider status:", status)
        
        # Example: Process a message
        print("\nAsking 'Hello, what can you do?'")
        response = await assistant.process_message("Hello, what can you do?")
        print("Jarvis response:", response)
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())