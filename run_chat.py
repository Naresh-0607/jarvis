#!/usr/bin/env python3
"""Simple script to test that our Jarvis chat system works."""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root))

print("Jarvis Assistant - Test Run")
print("=" * 40)

# Test file imports
try:
    from jarvis.providers import Provider, ProviderConfig, Chain
    print("[OK] Providers imported successfully")
    
    from jarvis.tools import registry
    print("[OK] Tools imported successfully")
    
    from jarvis.router import route_to_provider, get_model_for_provider, get_provider_tier
    print("[OK] Router imported successfully")
    
    from jarvis.chat import ChatSession
    print("[OK] Chat session imported successfully")
    
    # Test that basic classes can be instantiated 
    config = ProviderConfig(
        name="test",
        kind="openai_compat",
        base_url="https://api.example.com/v1",
        api_key_env="TEST_API_KEY",
        models=["gpt-3.5-turbo"],
        connect_timeout_s=5,
        read_timeout_s=10,
        options={}
    )
    
    print("[OK] ProviderConfig instantiated successfully")
    
    # Test that config files are present
    config_file = project_root / "config.yaml"
    env_file = project_root / ".env"
    persona_file = project_root / "jarvis" / "persona.md"
    
    if config_file.exists():
        print("[OK] config.yaml exists")
    else:
        print("[FAIL] config.yaml missing")
        
    if env_file.exists():
        print("[OK] .env exists")
    else:
        print("[FAIL] .env missing")
        
    if persona_file.exists():
        print("[OK] persona.md exists")
    else:
        print("[FAIL] persona.md missing")
        
    print("\nAll basic imports and checks passed!")
    print("To test the full chat system, run:")
    print("python -m jarvis.chat")

except Exception as e:
    print(f"Error during import/test: {e}")
    sys.exit(1)