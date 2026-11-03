#!/usr/bin/env python3
"""Smoke test script for Phase 6 - Project Folder Reader & Coding Mentor."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from jarvis.project_reader import (
    list_files,
    read_file,
    search_text,
    extract_stack_trace_locations,
    extract_code_context_from_ocr,
)
from jarvis.tools import registry


def main():
    print("=" * 60)
    print("Testing Phase 6: Project Folder Reader & Coding Mentor")
    print("=" * 60)

    # 1. Check Tool Registration
    print("\n1. Registered Project Tools:")
    all_tools = registry.get_all_tools()
    required = ["list_files", "read_file", "search_text"]
    for req in required:
        if req in all_tools:
            print(f"  [PASS] {req}: {all_tools[req].get('description')[:60]}...")
        else:
            print(f"  [FAIL] {req} NOT registered in tool registry!")

    # 2. Test list_files
    print("\n2. Testing list_files in project root:")
    res_list = list_files(directory="", recursive=False, max_files=10)
    if "error" in res_list:
        print(f"  [FAIL] list_files error: {res_list['error']}")
    else:
        print(f"  [PASS] Found {res_list['total_items']} items in root (showing first 5):")
        for item in res_list["items"][:5]:
            print(f"    - {item['name']} ({item['type']})")

    # 3. Test read_file with line ranges
    test_target = "jarvis/README.md"
    print(f"\n3. Testing read_file (lines 1 to 5 of {test_target}):")
    res_read = read_file(test_target, start_line=1, end_line=5)
    if "error" in res_read:
        # Fallback to direct README.md if running with jarvis as root
        res_read = read_file("README.md", start_line=1, end_line=5)
        if "error" not in res_read:
            test_target = "README.md"
    if "error" in res_read:
        print(f"  [FAIL] read_file error: {res_read['error']}")
    else:
        print(f"  [PASS] Read {res_read['start_line']} to {res_read['end_line']} of {res_read['path']}:")
        for line in res_read["content"].splitlines()[:5]:
            print(f"    {line}")

    # 4. Test search_text
    print("\n4. Testing search_text ('Phase 6' across project):")
    res_search = search_text("Phase 6", file_pattern="*.py", max_matches=5)
    if "error" in res_search:
        print(f"  [FAIL] search_text error: {res_search['error']}")
    else:
        print(f"  [PASS] Found {res_search['total_matches']} matches:")
        for m in res_search["matches"][:3]:
            print(f"    {m['path']}:{m['line']} -> {m['snippet']}")

    # 5. Test Path Safety (Escape attempts & sensitive files)
    print("\n5. Testing Path Safety & Jailbreak Protection:")

    # Escape attempt: parent traversal
    res_escape = read_file("../../windows/system32/cmd.exe")
    if "error" in res_escape and "Access denied" in res_escape["error"]:
        print("  [PASS] Parent path traversal blocked correctly.")
    else:
        print(f"  [FAIL] Parent path traversal was NOT blocked: {res_escape}")

    # Protected file: .env
    res_env = read_file(".env")
    if "error" in res_env and ("protected" in res_env["error"] or "Access denied" in res_env["error"]):
        print("  [PASS] Reading .env rejected as protected.")
    else:
        print(f"  [FAIL] Reading .env was NOT protected: {res_env}")

    # Protected file: *.key
    res_key = read_file("secret.key")
    if "error" in res_key and ("protected" in res_key["error"] or "Access denied" in res_key["error"]):
        print("  [PASS] Reading secret.key rejected as protected.")
    else:
        print(f"  [FAIL] Reading secret.key was NOT protected: {res_key}")

    # 6. Test Traceback & OCR Linking
    print("\n6. Testing OCR Traceback Extraction & Code Linking:")
    sample_traceback = """
Traceback (most recent call last):
  File "jarvis/doctor.py", line 65, in check_python_version
    if sys.version_info < (3, 11):
ValueError: Python 3.11 required.
    """
    locs = extract_stack_trace_locations(sample_traceback)
    print(f"  Extracted locations: {locs}")
    if locs:
        context = extract_code_context_from_ocr(sample_traceback)
        if context:
            print(f"  [PASS] Auto-linked code context found for traceback:")
            for line in context.splitlines()[:6]:
                print(f"    {line}")
        else:
            print("  [WARN] Traceback location parsed but code context could not be extracted.")
    else:
        print("  [FAIL] Failed to extract traceback location.")

    print("\n" + "=" * 60)
    print("Phase 6 tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()