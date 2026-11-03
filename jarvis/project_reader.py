"""Project folder reader and coding mentor tools (Phase 6).

Read-only tools for inspecting project files safely:
* ``list_files``: list files and directories within allowed project roots.
* ``read_file``: read file contents with 1-indexed line numbers and line ranges.
* ``search_text``: search for text/regex across code files.

Strict path safety:
* Rejects any path escaping allowed roots (no ``..`` traversal, no foreign absolute paths).
* Rejects protected files (``.env``, secrets, private keys).
* Rejects system/build directories (``.git``, ``node_modules``, ``venv``).
* Enforces read-only operations (no write, edit, delete, or shell commands).
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from jarvis.db import PROJECT_ROOT
from jarvis.tools import tool

_logger = logging.getLogger("jarvis.project_reader")

# Directories always excluded from listing and reading
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
}

# Protected files that must never be read by the LLM
DEFAULT_PROTECTED_FILES = {
    ".env",
    ".env.example",
    ".env.local",
    ".env.production",
}

DEFAULT_PROTECTED_PATTERNS = [
    "*.pem",
    "*.key",
    "*id_rsa*",
    "*.pfx",
    "*.p12",
    "*secret*",
    "*token*",
    ".env*",
]

DEFAULT_MAX_FILE_SIZE = 1024 * 1024  # 1 MB

_configured_roots: Optional[List[Path]] = None
_max_file_size: int = DEFAULT_MAX_FILE_SIZE


# ---------------------------------------------------------------------------
# Configuration and Roots
# ---------------------------------------------------------------------------

def configure(roots: Optional[List[Path | str]] = None, max_file_size: Optional[int] = None) -> None:
    """Configure allowed roots and limits (used by app startup and tests)."""
    global _configured_roots, _max_file_size
    if roots is not None:
        _configured_roots = [Path(r).resolve() for r in roots]
    else:
        _configured_roots = None

    if max_file_size is not None:
        _max_file_size = max_file_size
    else:
        _max_file_size = DEFAULT_MAX_FILE_SIZE


def get_allowed_roots() -> List[Path]:
    """Return the list of allowed project roots."""
    global _configured_roots
    if _configured_roots:
        return _configured_roots

    roots: List[Path] = []
    cfg_path = PROJECT_ROOT / "config.yaml"
    if cfg_path.exists():
        try:
            with cfg_path.open("rt", encoding="utf8") as handle:
                cfg = yaml.safe_load(handle) or {}

            # Check projects.roots list
            projects_cfg = cfg.get("projects") or {}
            if isinstance(projects_cfg.get("roots"), list):
                for r in projects_cfg["roots"]:
                    p = Path(r).resolve()
                    if p.exists():
                        roots.append(p)

            # Check projects.directory
            if not roots and projects_cfg.get("directory"):
                p = Path(projects_cfg["directory"]).resolve()
                if p.exists():
                    roots.append(p)

            # Check top-level project_root
            if not roots and cfg.get("project_root"):
                p = Path(cfg["project_root"]).resolve()
                if p.exists():
                    roots.append(p)
        except Exception as exc:
            _logger.warning("Could not load roots from config.yaml: %s", exc)

    # Fallback to current repository root if no root found or valid
    if not roots:
        roots.append(PROJECT_ROOT.resolve())

    return roots


# ---------------------------------------------------------------------------
# Path Safety
# ---------------------------------------------------------------------------

def is_protected_filename(name: str) -> bool:
    """Check if a file name matches protected sensitive file patterns."""
    lower_name = name.lower()
    if lower_name in DEFAULT_PROTECTED_FILES:
        return True
    for pattern in DEFAULT_PROTECTED_PATTERNS:
        if fnmatch.fnmatch(lower_name, pattern):
            return True
    return False


def is_in_excluded_dir(path: Path, root: Path) -> bool:
    """Check if any path component relative to root is in excluded directories."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False

    for part in rel.parts:
        if part in DEFAULT_EXCLUDED_DIRS:
            return True
    return False


def resolve_safe_path(target_path: Union[str, Path]) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve a path safely within the allowed roots.

    Returns:
        (resolved_path, None) on success.
        (None, error_message) if access is denied or path is invalid.
    """
    raw_str = str(target_path).strip()
    if not raw_str:
        return None, "Path cannot be empty"

    allowed_roots = get_allowed_roots()
    target = Path(raw_str)

    matched_root: Optional[Path] = None
    candidate: Optional[Path] = None

    if target.is_absolute():
        resolved = target.resolve()
        for root in allowed_roots:
            try:
                resolved.relative_to(root)
                matched_root = root
                candidate = resolved
                break
            except ValueError:
                continue
        if candidate is None:
            return None, f"Access denied: '{raw_str}' is outside allowed project roots."
    else:
        # Relative path: attempt resolution against each root
        for root in allowed_roots:
            resolved = (root / target).resolve()
            try:
                resolved.relative_to(root)
                matched_root = root
                candidate = resolved
                break
            except ValueError:
                continue
        if candidate is None or matched_root is None:
            return None, f"Access denied: path traversal outside allowed project roots."

    # Verify no excluded directories in relative path
    if is_in_excluded_dir(candidate, matched_root):
        return None, f"Access denied: '{raw_str}' is inside a protected system directory."

    # Verify not a protected file
    if is_protected_filename(candidate.name):
        return None, f"Access denied: '{candidate.name}' is protected and cannot be read."

    return candidate, None


# ---------------------------------------------------------------------------
# LLM Tools
# ---------------------------------------------------------------------------

LIST_FILES_SCHEMA = {
    "name": "list_files",
    "description": (
        "List files and directories within the project. Path is relative to the "
        "project root. Automatically skips protected files (.env, keys) and system "
        "folders (.git, node_modules, venv)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "Subdirectory relative to project root (empty string for project root)",
                "default": "",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to list subdirectories recursively (default false)",
                "default": False,
            },
            "max_files": {
                "type": "integer",
                "description": "Maximum number of files to return (default 50)",
                "default": 50,
            },
        },
    },
}


@tool(name="list_files", schema=LIST_FILES_SCHEMA)
def list_files(
    directory: str = "",
    recursive: bool = False,
    max_files: int = 50,
) -> Dict[str, Any]:
    """List files in the project."""
    allowed_roots = get_allowed_roots()
    if not directory:
        target_dir = allowed_roots[0]
    else:
        resolved, error = resolve_safe_path(directory)
        if error or resolved is None:
            return {"error": error}
        target_dir = resolved

    if not target_dir.exists():
        return {"error": f"Directory not found: '{directory}'"}
    if not target_dir.is_dir():
        return {"error": f"Path is not a directory: '{directory}'"}

    # Find the matching root for computing relative paths
    root_for_rel = allowed_roots[0]
    for r in allowed_roots:
        try:
            target_dir.relative_to(r)
            root_for_rel = r
            break
        except ValueError:
            continue

    results: List[Dict[str, Any]] = []
    iterator = target_dir.rglob("*") if recursive else target_dir.iterdir()

    for item in iterator:
        if len(results) >= max_files:
            break

        if is_in_excluded_dir(item, root_for_rel):
            continue

        if item.is_file() and is_protected_filename(item.name):
            continue

        try:
            rel_path = str(item.relative_to(root_for_rel)).replace("\\", "/")
            results.append(
                {
                    "name": item.name,
                    "path": rel_path,
                    "type": "directory" if item.is_dir() else "file",
                    "size_bytes": item.stat().st_size if item.is_file() else None,
                }
            )
        except (OSError, ValueError):
            continue

    return {
        "root": str(root_for_rel),
        "directory": str(target_dir.relative_to(root_for_rel) if target_dir != root_for_rel else "."),
        "total_items": len(results),
        "items": results,
    }


READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": (
        "Read contents of a file in the project with 1-indexed line numbers. "
        "Optionally specify a line range (start_line to end_line). Rejects protected "
        "files (.env, keys) and files outside project roots."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file (relative to project root)",
            },
            "start_line": {
                "type": "integer",
                "description": "First line to read (1-indexed, default 1)",
                "default": 1,
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to read (inclusive, optional)",
            },
        },
        "required": ["path"],
    },
}


@tool(name="read_file", schema=READ_FILE_SCHEMA)
def read_file(
    path: str,
    start_line: int = 1,
    end_line: Optional[int] = None,
) -> Dict[str, Any]:
    """Read file content with line numbers."""
    resolved, error = resolve_safe_path(path)
    if error or resolved is None:
        return {"error": error}

    if not resolved.exists():
        return {"error": f"File not found: '{path}'"}
    if not resolved.is_file():
        return {"error": f"Path is a directory, not a file: '{path}'"}

    size = resolved.stat().st_size
    if size > _max_file_size:
        return {
            "error": f"File '{path}' is too large ({size} bytes, limit {_max_file_size} bytes)."
        }

    try:
        with resolved.open("r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except Exception as exc:
        return {"error": f"Could not read '{path}': {exc}"}

    total_lines = len(all_lines)
    s_line = max(1, int(start_line or 1))
    e_line = min(total_lines, int(end_line)) if end_line is not None else total_lines

    if s_line > total_lines:
        return {
            "path": path,
            "total_lines": total_lines,
            "content": "",
            "note": f"start_line {s_line} is beyond total lines ({total_lines})",
        }

    selected_lines = all_lines[s_line - 1 : e_line]
    formatted_content = "".join(
        f"{s_line + idx}: {line}" for idx, line in enumerate(selected_lines)
    )

    return {
        "path": path,
        "start_line": s_line,
        "end_line": e_line,
        "total_lines": total_lines,
        "content": formatted_content,
    }


SEARCH_TEXT_SCHEMA = {
    "name": "search_text",
    "description": (
        "Search for a text string or pattern across code files in the project. "
        "Returns matching files, line numbers, and line snippets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text string or regex pattern to search for",
            },
            "directory": {
                "type": "string",
                "description": "Subdirectory to restrict search to (default project root)",
                "default": "",
            },
            "file_pattern": {
                "type": "string",
                "description": "File glob filter (e.g. '*.py', '*.ts', default '*')",
                "default": "*",
            },
            "max_matches": {
                "type": "integer",
                "description": "Maximum number of matching lines to return (default 25)",
                "default": 25,
            },
        },
        "required": ["query"],
    },
}


@tool(name="search_text", schema=SEARCH_TEXT_SCHEMA)
def search_text(
    query: str,
    directory: str = "",
    file_pattern: str = "*",
    max_matches: int = 25,
) -> Dict[str, Any]:
    """Search for query across project files."""
    if not query:
        return {"error": "Query cannot be empty"}

    allowed_roots = get_allowed_roots()
    if not directory:
        target_dir = allowed_roots[0]
    else:
        resolved, error = resolve_safe_path(directory)
        if error or resolved is None:
            return {"error": error}
        target_dir = resolved

    if not target_dir.is_dir():
        return {"error": f"Search directory not found: '{directory}'"}

    root_for_rel = allowed_roots[0]
    for r in allowed_roots:
        try:
            target_dir.relative_to(r)
            root_for_rel = r
            break
        except ValueError:
            continue

    # Compile regex or literal
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        pattern = re.compile(re.escape(query), re.IGNORECASE)

    matches: List[Dict[str, Any]] = []

    for file_path in target_dir.rglob(file_pattern or "*"):
        if len(matches) >= max_matches:
            break
        if not file_path.is_file():
            continue
        if is_in_excluded_dir(file_path, root_for_rel):
            continue
        if is_protected_filename(file_path.name):
            continue
        if file_path.stat().st_size > _max_file_size:
            continue

        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, start=1):
                    if pattern.search(line):
                        rel = str(file_path.relative_to(root_for_rel)).replace("\\", "/")
                        matches.append(
                            {
                                "path": rel,
                                "line": line_no,
                                "snippet": line.strip()[:200],
                            }
                        )
                        if len(matches) >= max_matches:
                            break
        except Exception:
            continue

    return {
        "query": query,
        "total_matches": len(matches),
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# ---------------------------------------------------------------------------

def read_project_folder(directory: str = "", recursive: bool = True, max_files: int = 50, **kwargs) -> Dict[str, Any]:
    """Backward compatibility wrapper for existing tests/scripts."""
    listing = list_files(directory=directory, recursive=recursive, max_files=max_files)
    if "error" in listing:
        return listing

    files_info = []
    for item in listing.get("items", []):
        if item.get("type") == "file":
            files_info.append({
                "path": item["path"],
                "name": item["name"],
                "size": item.get("size_bytes", 0),
            })
    return {
        "directory": listing.get("directory", ""),
        "total_files": len(files_info),
        "files": files_info,
    }


def list_project_files(directory: str = "", recursive: bool = True) -> Dict[str, Any]:
    """Backward compatibility alias for list_files."""
    return list_files(directory=directory, recursive=recursive)


# ---------------------------------------------------------------------------
# Screen OCR Error Linking
# ---------------------------------------------------------------------------

# Patterns matching Python tracebacks, compiler errors, or file:line markers
TRACEBACK_PATTERNS = [
    # Python: File "foo.py", line 42, in bar
    re.compile(r'File\s+["\']([^"\']+)["\'],\s+line\s+(\d+)', re.IGNORECASE),
    # Common format: path/to/file.ext:42 or path/to/file.ext(42)
    re.compile(r'([a-zA-Z0-9_\-\.\/\\]+\.(?:py|js|ts|jsx|tsx|html|css|json|yaml|yml|c|cpp|h))(?::|\()(\d+)', re.IGNORECASE),
]


def extract_stack_trace_locations(text: str) -> List[Tuple[str, int]]:
    """Extract (file_path, line_number) pairs from error or traceback text."""
    locations: List[Tuple[str, int]] = []
    seen = set()

    for pat in TRACEBACK_PATTERNS:
        for match in pat.finditer(text):
            fpath, lnum = match.group(1), int(match.group(2))
            clean_path = fpath.strip().replace("\\", "/")
            key = (clean_path, lnum)
            if key not in seen:
                seen.add(key)
                locations.append(key)

    return locations


def get_code_context_for_error(
    raw_path: str, line_no: int, context_lines: int = 5
) -> Optional[Dict[str, Any]]:
    """Fetch surrounding source code for an error found in screen OCR.

    Looks up raw_path in allowed roots. If found, returns lines surrounding line_no.
    """
    allowed_roots = get_allowed_roots()
    target_file: Optional[Path] = None

    # Try resolving directly
    res, err = resolve_safe_path(raw_path)
    if res and res.is_file():
        target_file = res
    else:
        # Search by file name across roots if path was partial or relative
        filename = Path(raw_path).name
        for root in allowed_roots:
            candidates = list(root.rglob(filename))
            for cand in candidates:
                res_cand, err_cand = resolve_safe_path(cand)
                if res_cand and res_cand.is_file():
                    target_file = res_cand
                    break
            if target_file:
                break

    if not target_file or not target_file.is_file():
        return None

    # Read slice
    start = max(1, line_no - context_lines)
    end = line_no + context_lines
    res = read_file(str(target_file), start_line=start, end_line=end)
    if "error" in res:
        return None

    return {
        "file": str(target_file.name),
        "full_path": str(target_file),
        "target_line": line_no,
        "start_line": start,
        "end_line": end,
        "snippet": res.get("content", ""),
    }


def extract_code_context_from_ocr(text: str) -> str:
    """Format surrounding source code for any error tracebacks found in OCR text."""
    locs = extract_stack_trace_locations(text)
    if not locs:
        return ""

    snippets: List[str] = []
    for fpath, lnum in locs[:2]:  # Limit to first 2 error locations
        ctx = get_code_context_for_error(fpath, lnum)
        if ctx:
            snippets.append(
                f"File: {ctx['file']} (around line {ctx['target_line']}):\n{ctx['snippet']}"
            )

    return "\n\n".join(snippets)