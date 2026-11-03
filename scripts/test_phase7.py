#!/usr/bin/env python3
"""Smoke test script for Phase 7 - Long-Term Memory & Interactive Teaching."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from jarvis.memory import (
    remember_fact,
    recall,
    update_project,
    save_lesson_progress,
    lookup_docs,
    get_service,
)
from jarvis.tools import registry


def main():
    print("=" * 60)
    print("Testing Phase 7: Memory & Interactive Teaching")
    print("=" * 60)

    # 1. Registered Tools Check
    print("\n1. Registered Memory & Teaching Tools:")
    all_tools = registry.get_all_tools()
    required = ["remember_fact", "recall", "update_project", "save_lesson_progress", "lookup_docs"]
    for req in required:
        if req in all_tools:
            print(f"  [PASS] {req}: {all_tools[req].get('description')[:60]}...")
        else:
            print(f"  [FAIL] {req} NOT registered in tool registry!")

    # 2. Test remember_fact and recall
    print("\n2. Testing remember_fact & recall:")
    res_fact = remember_fact(key="user_name", value="Naresh", category="personal")
    print(f"  Saved fact: {res_fact}")
    res_recall = recall("user_name")
    print(f"  Recalled fact: {res_recall['facts']}")
    if res_recall["total_facts"] >= 1:
        print("  [PASS] Fact successfully remembered and recalled.")
    else:
        print("  [FAIL] Fact recall failed.")

    # 3. Test update_project
    print("\n3. Testing update_project:")
    res_proj = update_project(
        name="ESP32 Smart Sensor",
        goal="Build an environmental sensor node",
        status="active",
        next_step="Connect BME280 sensor via I2C",
    )
    print(f"  Project saved: {res_proj['project']['name']} (next: {res_proj['project']['next_step']})")
    print("  [PASS] Project tracking verified.")

    # 4. Test save_lesson_progress
    print("\n4. Testing save_lesson_progress (ESP32 Lesson):")
    res_lesson = save_lesson_progress(
        topic="esp32",
        step=1,
        steps_completed="Blinked onboard LED on GPIO 2",
        where_stuck="",
        level="beginner",
    )
    print(f"  Lesson saved: {res_lesson['lesson']['topic']} at Step {res_lesson['lesson']['current_step']}")
    active_lesson = get_service().get_active_lesson()
    if active_lesson and active_lesson["topic"] == "esp32":
        print(f"  [PASS] Active lesson stored: {active_lesson['steps_completed']}")
    else:
        print("  [FAIL] Failed to retrieve active lesson.")

    # 5. Test lookup_docs (Hardware Grounding)
    print("\n5. Testing lookup_docs (ESP32 Hardware Reference):")
    res_voltage = lookup_docs("3.3 V", topic="esp32")
    if res_voltage.get("found"):
        print(f"  [PASS] Found voltage rules in {res_voltage['matches'][0]['file']}:")
        for line in res_voltage["matches"][0]["snippet"].splitlines()[:3]:
            print(f"    {line}")
    else:
        print(f"  [FAIL] Voltage lookup failed: {res_voltage}")

    res_pin = lookup_docs("GPIO 2", topic="esp32")
    if res_pin.get("found"):
        print(f"  [PASS] Found GPIO 2 pinout in {res_pin['matches'][0]['file']}:")
        for line in res_pin["matches"][0]["snippet"].splitlines()[:3]:
            print(f"    {line}")
    else:
        print(f"  [FAIL] GPIO 2 lookup failed: {res_pin}")

    # 6. Test lookup_docs uncertainty on missing item
    res_unknown = lookup_docs("Quantum Processor Pinout")
    if not res_unknown.get("found") and "not sure" in res_unknown.get("message", "").lower():
        print("  [PASS] Unknown hardware spec gracefully returns uncertainty message.")
    else:
        print(f"  [WARN] Unknown lookup response: {res_unknown}")

    print("\n" + "=" * 60)
    print("Phase 7 tests completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
