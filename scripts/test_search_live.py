#!/usr/bin/env python3
"""Live smoke test for Jarvis web search engine (Phase 8+).

Runs at most 6 real queries (including "jk south indian youtuber" and follow-up with hint),
measuring latency, provider selection, cache hits, and credits used.
"""

from __future__ import annotations

import os
from pathlib import Path
import statistics
import sys
import time

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from jarvis.web_search import get_monthly_credit_usage, perform_web_search


def main():
    print("=" * 70)
    print("Jarvis Live Web Search Smoke Test")
    print("=" * 70)

    # Check key presence
    tavily_key = os.environ.get("TAVILY_API_KEY")
    exa_key = os.environ.get("EXA_API_KEY")
    print(f"Tavily API key present: {'YES' if tavily_key else 'NO'}")
    print(f"Exa API key present:    {'YES' if exa_key else 'NO'}")

    initial_usage = get_monthly_credit_usage()
    tavily_start_credits = initial_usage.get("tavily", 0)
    exa_start_credits = initial_usage.get("exa", 0)
    print(f"Initial monthly credits: Tavily={tavily_start_credits}, Exa={exa_start_credits}")
    print("-" * 70)

    # Define test suite (at most 6 queries)
    test_queries = [
        {"name": "Initial query", "query": "jk south indian youtuber", "recent": False, "bypass_cache": True},
        {"name": "Follow-up with hint (combined)", "query": "JK Tamil comedy YouTuber", "recent": False, "bypass_cache": True},
        {"name": "General factual query", "query": "Python 3.12 release highlights", "recent": False, "bypass_cache": True},
        {"name": "Recent / News query", "query": "James Webb space telescope latest discovery", "recent": True, "bypass_cache": True},
        {"name": "Cache verification repeat", "query": "jk south indian youtuber", "recent": False, "bypass_cache": False},
    ]

    latencies = []

    for idx, item in enumerate(test_queries, start=1):
        name = item["name"]
        q = item["query"]
        recent = item["recent"]
        bypass = item["bypass_cache"]

        print(f"\n[{idx}/5] {name}: \"{q}\"")
        t0 = time.perf_counter()
        res = perform_web_search(query=q, max_results=5, recent=recent, bypass_cache=bypass)
        elapsed_s = time.perf_counter() - t0

        if not res.get("cached"):
            latencies.append(elapsed_s)

        status = res.get("status", "unknown")
        provider = res.get("provider", "none")
        cached = res.get("cached", False)
        count = res.get("total_results", 0)
        answer = res.get("answer")

        print(f"    Status:   {status} | Provider: {provider} | Cached: {cached}")
        print(f"    Latency:  {elapsed_s * 1000:0.1f} ms ({elapsed_s:0.2f} s)")
        print(f"    Results:  {count} found")
        if answer:
            print(f"    Answer:   {answer[:120]}...")

        # Display top 2 results
        for r_idx, r in enumerate(res.get("results", [])[:2], start=1):
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("snippet", "")[:100]
            print(f"      {r_idx}. {title} ({url})")
            print(f"         Snippet: {snippet}...")

    # Calculate ending credits
    ending_usage = get_monthly_credit_usage()
    tavily_end_credits = ending_usage.get("tavily", 0)
    exa_end_credits = ending_usage.get("exa", 0)
    tavily_used = tavily_end_credits - tavily_start_credits
    exa_used = exa_end_credits - exa_start_credits

    print("\n" + "=" * 70)
    print("LIVE SEARCH TEST SUMMARY")
    print("=" * 70)
    if latencies:
        median_latency = statistics.median(latencies)
        print(f"Median Search Latency (live non-cached): {median_latency * 1000:0.1f} ms ({median_latency:0.2f} s)")
    print(f"Total Tavily Credits Used:              {tavily_used}")
    print(f"Total Exa Credits Used:                 {exa_used}")
    print(f"Current Monthly Totals:                 Tavily={tavily_end_credits}/1000, Exa={exa_end_credits}/1000")
    print("=" * 70)


if __name__ == "__main__":
    main()

