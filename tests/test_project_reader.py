"""Unit tests for Phase 6 - Project Folder Reader & Coding Mentor."""

from pathlib import Path
import pytest

from jarvis.project_reader import (
    configure,
    get_allowed_roots,
    is_protected_filename,
    resolve_safe_path,
    list_files,
    read_file,
    search_text,
    extract_stack_trace_locations,
    get_code_context_for_error,
    extract_code_context_from_ocr,
)
from jarvis.chat import ChatSession


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary sandbox project structure for testing."""
    root = tmp_path / "sandbox_project"
    root.mkdir()

    # Normal files
    (root / "hello.py").write_text("print('hello world')\nline2\nline3\nline4\nline5\n", encoding="utf-8")
    sub = root / "src"
    sub.mkdir()
    (sub / "app.py").write_text("def run():\n    raise ValueError('crash')\n", encoding="utf-8")

    # Sensitive/protected files
    (root / ".env").write_text("SECRET_KEY=12345", encoding="utf-8")
    (root / ".env.local").write_text("LOCAL=true", encoding="utf-8")
    (root / "server.key").write_text("PRIVATE KEY", encoding="utf-8")
    (root / "id_rsa").write_text("SSH KEY", encoding="utf-8")

    # Excluded directories
    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config", encoding="utf-8")

    venv_dir = root / "venv"
    venv_dir.mkdir()
    (venv_dir / "pyvenv.cfg").write_text("venv config", encoding="utf-8")

    # Configure project reader to use this sandbox root
    configure(roots=[root], max_file_size=1024 * 1024)

    yield root

    # Reset configuration
    configure(roots=None)


def test_list_files_basic(temp_project):
    res = list_files(directory="", recursive=False)
    assert "error" not in res
    assert res["total_items"] > 0
    names = [item["name"] for item in res["items"]]
    assert "hello.py" in names
    assert "src" in names


def test_list_files_skips_protected_and_excluded(temp_project):
    res = list_files(directory="", recursive=True)
    assert "error" not in res
    names = [item["name"] for item in res["items"]]

    # Normal files should be present
    assert "hello.py" in names
    assert "app.py" in names

    # Protected files and excluded directories must NOT appear
    assert ".env" not in names
    assert ".env.local" not in names
    assert "server.key" not in names
    assert "id_rsa" not in names
    assert ".git" not in names
    assert "config" not in names
    assert "venv" not in names


def test_read_file_full(temp_project):
    res = read_file("hello.py")
    assert "error" not in res
    assert res["total_lines"] == 5
    assert "1: print('hello world')" in res["content"]
    assert "5: line5" in res["content"]


def test_read_file_line_range(temp_project):
    res = read_file("hello.py", start_line=2, end_line=4)
    assert "error" not in res
    assert res["start_line"] == 2
    assert res["end_line"] == 4
    assert "2: line2" in res["content"]
    assert "4: line4" in res["content"]
    assert "1: print" not in res["content"]
    assert "5: line5" not in res["content"]


def test_read_file_not_found(temp_project):
    res = read_file("does_not_exist.py")
    assert "error" in res
    assert "not found" in res["error"].lower()


def test_read_file_too_large(temp_project):
    configure(roots=[temp_project], max_file_size=10)
    res = read_file("hello.py")
    assert "error" in res
    assert "too large" in res["error"].lower()


def test_path_safety_parent_traversal(temp_project):
    res = read_file("../../../secret_file.txt")
    assert "error" in res
    assert "Access denied" in res["error"]


def test_path_safety_protected_files(temp_project):
    for protected in [".env", ".env.local", "server.key", "id_rsa"]:
        res = read_file(protected)
        assert "error" in res, f"Expected {protected} to be blocked"
        assert "Access denied" in res["error"] or "protected" in res["error"]


def test_path_safety_excluded_dirs(temp_project):
    res = read_file(".git/config")
    assert "error" in res
    assert "Access denied" in res["error"]


def test_search_text(temp_project):
    res = search_text("hello", directory="")
    assert "error" not in res
    assert res["total_matches"] == 1
    match = res["matches"][0]
    assert "hello.py" in match["path"]
    assert match["line"] == 1
    assert "print('hello world')" in match["snippet"]


def test_search_text_pattern_filter(temp_project):
    res = search_text("run", file_pattern="*.py")
    assert "error" not in res
    assert res["total_matches"] == 1
    assert "app.py" in res["matches"][0]["path"]


def test_extract_stack_trace_locations():
    traceback_sample = """
Traceback (most recent call last):
  File "src/app.py", line 42, in run
    do_something()
  File "utils/calc.py", line 18, in add
    return x + y
    """
    locs = extract_stack_trace_locations(traceback_sample)
    assert len(locs) == 2
    assert locs[0] == ("src/app.py", 42)
    assert locs[1] == ("utils/calc.py", 18)


def test_get_code_context_for_error(temp_project):
    ctx = get_code_context_for_error("src/app.py", line_no=2, context_lines=2)
    assert ctx is not None
    assert ctx["file"] == "app.py"
    assert "raise ValueError" in ctx["snippet"]


def test_extract_code_context_from_ocr(temp_project):
    ocr_sample = """
Console Error Output:
Traceback (most recent call last):
  File "src/app.py", line 2, in run
    raise ValueError('crash')
ValueError: crash
    """
    context = extract_code_context_from_ocr(ocr_sample)
    assert "app.py" in context
    assert "raise ValueError" in context


def test_chat_repeated_error_awareness():
    session = ChatSession()
    error_text = "ValueError: invalid literal for int() with base 10: 'abc'"

    # First encounter
    first_check = session._check_repeated_error(error_text)
    assert first_check is None
    assert len(session.recent_errors) == 1

    # Second encounter of same error
    second_check = session._check_repeated_error(error_text)
    assert second_check is not None
    assert "Repeated Error Note" in second_check
    assert "ValueError: invalid literal for int() with base 10: 'abc'" in second_check
