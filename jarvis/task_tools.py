"""LLM tools for tasks and reminders.

The model only turns words into structured fields. Every read comes from the
database, never from conversation memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from jarvis.db import PROJECT_ROOT
from jarvis.tasks import TaskService, human_deadline, parse_deadline
from jarvis.tools import tool

_logger = logging.getLogger("jarvis.task_tools")

_service: Optional[TaskService] = None
_db_path: Optional[str] = None


def configure(db_path: Optional[str] = None) -> None:
    """Point the tools at a specific database (tests use this)."""
    global _service, _db_path
    _db_path = db_path
    if _service is not None:
        _service.close()
    _service = None


def get_service() -> TaskService:
    global _service
    if _service is None:
        path: Optional[str] = _db_path
        if not path:
            cfg_path = PROJECT_ROOT / "config.yaml"
            if cfg_path.exists():
                with cfg_path.open("rt", encoding="utf8") as handle:
                    cfg = yaml.safe_load(handle) or {}
                path = (cfg.get("tasks") or {}).get("db_path")
        if path and not Path(path).is_absolute():
            path = str(PROJECT_ROOT / path)
        _service = TaskService(path)
    return _service


TASK_SCHEMA = {
    "name": "add_task",
    "description": (
        "Save a task with an optional deadline. Deadlines must be ISO 8601 in "
        "Asia/Kolkata time, for example 2026-10-30T17:00:00+05:30. Convert the "
        "user's words into that format yourself. If the time they said was vague, "
        "call this first with confirm=false to get a read-back sentence, ask the "
        "user, then call again with confirm=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short task title"},
            "deadline": {
                "type": "string",
                "description": "ISO 8601 datetime with offset, or omit if none",
            },
            "notes": {"type": "string", "description": "Extra detail, optional"},
            "important": {
                "type": "boolean",
                "description": "True for things that really matter; adds an extra reminder",
            },
            "confirm": {
                "type": "boolean",
                "description": "Set true once the user agrees with the read-back",
                "default": False,
            },
        },
        "required": ["title"],
    },
}

LIST_SCHEMA = {
    "name": "list_tasks",
    "description": (
        "Read tasks from the database. Use this for 'what's pending?', "
        "'anything today?' or 'what did I miss?'. Never answer from memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "range": {
                "type": "string",
                "enum": ["today", "week", "overdue", "all"],
                "description": "Which slice of tasks to read",
                "default": "all",
            }
        },
        "required": [],
    },
}

COMPLETE_SCHEMA = {
    "name": "complete_task",
    "description": "Mark a task as done.",
    "parameters": {
        "type": "object",
        "properties": {"task_id": {"type": "integer"}},
        "required": ["task_id"],
    },
}

SNOOZE_SCHEMA = {
    "name": "snooze_task",
    "description": "Push a task's deadline later by a number of minutes.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "integer"},
            "minutes": {"type": "integer", "description": "How long to snooze"},
        },
        "required": ["task_id", "minutes"],
    },
}

DELETE_SCHEMA = {
    "name": "delete_task",
    "description": (
        "Delete a task. Always ask the user out loud first; only then call "
        "again with confirmed=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "integer"},
            "confirmed": {"type": "boolean", "default": False},
        },
        "required": ["task_id"],
    },
}


@tool(name="add_task", schema=TASK_SCHEMA)
def add_task(
    title: str,
    deadline: Optional[str] = None,
    notes: str = "",
    important: bool = False,
    confirm: bool = False,
) -> Dict[str, Any]:
    """Add a task, reading a vague deadline back before saving."""
    service = get_service()
    due = parse_deadline(deadline)

    if due is not None and not confirm:
        return {
            "status": "needs_confirmation",
            "read_back": f"{human_deadline(due)}, right?",
            "parsed_deadline": due.isoformat(timespec="seconds"),
            "next_step": (
                "Ask the user to confirm. If they agree, call add_task again "
                "with confirm=true; if not, call it with the corrected deadline."
            ),
        }

    task = service.add_task(
        title=title, deadline=due, notes=notes, important=bool(important)
    )
    _logger.info("Task %s saved: %s", task["id"], task["title"])
    return {
        "status": "saved",
        "task": task,
        "read_back": f"Saved: {task['title']}, {task['deadline_human']}.",
    }


@tool(name="list_tasks", schema=LIST_SCHEMA)
def list_tasks(range: str = "all") -> Dict[str, Any]:  # noqa: A002 - schema name
    """List tasks from the database."""
    service = get_service()
    tasks = service.list_tasks(range)
    return {"range": range, "count": len(tasks), "tasks": tasks}


@tool(name="complete_task", schema=COMPLETE_SCHEMA)
def complete_task(task_id: int) -> Dict[str, Any]:
    """Mark a task done."""
    service = get_service()
    task = service.complete_task(int(task_id))
    if not task:
        return {"status": "not_found", "task_id": task_id}
    return {"status": "done", "task": task}


@tool(name="snooze_task", schema=SNOOZE_SCHEMA)
def snooze_task(task_id: int, minutes: int) -> Dict[str, Any]:
    """Delay a task's deadline."""
    service = get_service()
    task = service.snooze_task(int(task_id), int(minutes))
    if not task:
        return {"status": "not_found", "task_id": task_id}
    return {
        "status": "snoozed",
        "task": task,
        "read_back": f"{task['title']} moved to {task['deadline_human']}.",
    }


@tool(name="delete_task", schema=DELETE_SCHEMA)
def delete_task(task_id: int, confirmed: bool = False) -> Dict[str, Any]:
    """Delete a task after an explicit confirmation."""
    service = get_service()
    task = service.get_task(int(task_id))
    if not task:
        return {"status": "not_found", "task_id": task_id}

    if not confirmed:
        return {
            "status": "needs_confirmation",
            "read_back": f"Delete '{task['title']}'?",
            "next_step": "Ask the user, then call again with confirmed=true.",
        }

    service.delete_task(int(task_id))
    _logger.info("Task %s deleted", task_id)
    return {"status": "deleted", "task_id": int(task_id)}
