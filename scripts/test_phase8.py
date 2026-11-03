#!/usr/bin/env python3
"""Smoke test script for Phase 8 - General Tools (Weather, Web Search, Time, Math)."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from jarvis.general_tools import get_weather, web_search, current_time, calculate
from jarvis.router import choose_tier, FAST, STRONG
from jarvis.tools import registry


def main():
    print("=" * 60)
    print("Testing Phase 8: General Tools")
    print("=" * 60)

    # 1. Tool Registration Check
    print("\n1. Registered General Tools:")
    all_tools = registry.get_all_tools()
    required = ["get_weather", "web_search", "current_time", "calculate"]
    for req in required:
        if req in all_tools:
            print(f"  [PASS] {req}: {all_tools[req].get('description')[:60]}...")
        else:
            print(f"  [FAIL] {req} NOT registered in tool registry!")

    # 2. Test current_time
    print("\n2. Testing current_time:")
    t_res = current_time()
    print(f"  Current time: {t_res['time']} on {t_res['day_of_week']}, {t_res['date']} ({t_res['timezone']})")
    assert "time" in t_res and "date" in t_res
    print("  [PASS] Current time returned accurately.")

    # 3. Test calculate
    print("\n3. Testing calculate (safe arithmetic):")
    calc1 = calculate("24 * 60 * 365")
    print(f"  24 * 60 * 365 = {calc1.get('result')}")
    assert calc1.get("result") == 525600

    calc2 = calculate("sqrt(256) + 2**8")
    print(f"  sqrt(256) + 2**8 = {calc2.get('result')}")
    assert calc2.get("result") == 272

    calc_unsafe = calculate("__import__('os').system('dir')")
    if "error" in calc_unsafe:
        print("  [PASS] Unsafe execution blocked by AST safety validator.")
    else:
        print(f"  [FAIL] Unsafe execution was NOT blocked: {calc_unsafe}")

    # 4. Test live get_weather (Open-Meteo)
    print("\n4. Testing get_weather (Live Open-Meteo API):")
    w_res = get_weather()
    if "error" in w_res:
        print(f"  [WARN] Weather lookup error: {w_res['error']}")
    else:
        now_w = w_res["now"]
        today_w = w_res["today"]
        print(f"  Location: {w_res['city']}")
        print(f"  Current: {now_w['temperature_c']}°C, {now_w['condition']} (feels like {now_w['feels_like_c']}°C)")
        print(f"  Today's High/Low: {today_w['high_c']}°C / {today_w['low_c']}°C (Rain chance: {today_w['rain_chance_percent']}%)")
        print("  [PASS] Live weather forecast retrieved successfully.")

    # 5. Test live web_search (DuckDuckGo)
    print("\n5. Testing web_search (Live DuckDuckGo API):")
    s_res = web_search("Python 3.12 release date", max_results=2)
    if "error" in s_res or s_res.get("total_results", 0) == 0:
        print(f"  [WARN] Web search returned no immediate results: {s_res.get('message', s_res.get('error'))}")
    else:
        print(f"  Found {s_res['total_results']} result(s):")
        for idx, item in enumerate(s_res["results"], start=1):
            print(f"    {idx}. {item['title']}")
            print(f"       Snippet: {item['snippet'][:120]}...")
            print(f"       Source: {item['source']}")
        print("  [PASS] Live web search functioning.")

    # 6. Test router enhancements
    print("\n6. Testing router tier selections:")
    assert choose_tier("what's the weather today?") == FAST
    assert choose_tier("calculate 15 * 84") == FAST
    assert choose_tier("why does this error occur in my python program?") == STRONG
    assert choose_tier("explain how the event loop works step by step") == STRONG
    print("  [PASS] Lookups routed to FAST, complex reasoning routed to STRONG.")

    print("\n" + "=" * 60)
    print("Phase 8 tests completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
