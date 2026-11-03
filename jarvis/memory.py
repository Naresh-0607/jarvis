"""Long-term memory, hardware documentation lookup, and teaching tools (Phase 7).

Manages:
* Facts about the user and preferences (facts table)
* Projects, goals, and next steps (projects table)
* Lesson progress, topics, and where stuck (lessons table)
* Local hardware documentation search (lookup_docs tool)
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from jarvis.db import PROJECT_ROOT, connect
from jarvis.tools import tool

_logger = logging.getLogger("jarvis.memory")
TZ = ZoneInfo("Asia/Kolkata")

_service: Optional[MemoryService] = None
_db_path: Optional[str] = None


def now() -> str:
    """Current time formatted in ISO 8601 with project timezone."""
    return datetime.now(TZ).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Memory Service
# ---------------------------------------------------------------------------

class MemoryService:
    """SQLite-backed memory service for user facts, projects, and lessons."""

    def __init__(self, db_path: Optional[str | Path] = None, docs_dir: Optional[str | Path] = None):
        self.conn = connect(db_path)
        self.docs_dir = Path(docs_dir).resolve() if docs_dir else (PROJECT_ROOT / "docs").resolve()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # -- Facts ---------------------------------------------------------
    def remember_fact(self, key: str, value: str, category: str = "general") -> Dict[str, Any]:
        """Save or update a fact about the user or preferences."""
        key = (key or "").strip().lower()
        value = (value or "").strip()
        category = (category or "general").strip().lower()

        if not key or not value:
            return {"error": "Both key and value are required"}

        updated_at = now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO facts (key, value, category, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    updated_at = excluded.updated_at
                """,
                (key, value, category, updated_at),
            )
        return {"status": "saved", "key": key, "value": value, "category": category}

    def recall(self, query: str = "") -> Dict[str, Any]:
        """Recall saved facts matching query, or return all facts if empty."""
        query = (query or "").strip().lower()

        if query:
            cursor = self.conn.execute(
                """
                SELECT key, value, category, updated_at FROM facts
                WHERE key LIKE ? OR value LIKE ? OR category LIKE ?
                ORDER BY updated_at DESC
                """,
                (f"%{query}%", f"%{query}%", f"%{query}%"),
            )
        else:
            cursor = self.conn.execute(
                "SELECT key, value, category, updated_at FROM facts ORDER BY updated_at DESC"
            )

        facts = [
            {"key": r["key"], "value": r["value"], "category": r["category"], "updated_at": r["updated_at"]}
            for r in cursor.fetchall()
        ]

        # Also fetch active projects and ongoing lessons for context
        projects_cur = self.conn.execute(
            "SELECT name, goal, status, next_step FROM projects WHERE status = 'active'"
        )
        active_projects = [dict(r) for r in projects_cur.fetchall()]

        lessons_cur = self.conn.execute(
            "SELECT topic, level, current_step, steps_completed, where_stuck, last_session FROM lessons ORDER BY last_session DESC LIMIT 1"
        )
        active_lesson = lessons_cur.fetchone()

        return {
            "facts": facts,
            "total_facts": len(facts),
            "active_projects": active_projects,
            "active_lesson": dict(active_lesson) if active_lesson else None,
        }

    # -- Projects ------------------------------------------------------
    def update_project(
        self,
        name: str,
        goal: str = "",
        status: str = "active",
        next_step: str = "",
        notes: str = "",
    ) -> Dict[str, Any]:
        """Create or update a tracked user project."""
        name = (name or "").strip()
        if not name:
            return {"error": "Project name is required"}

        updated_at = now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO projects (name, goal, status, notes, next_step, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    goal = CASE WHEN excluded.goal != '' THEN excluded.goal ELSE projects.goal END,
                    status = CASE WHEN excluded.status != '' THEN excluded.status ELSE projects.status END,
                    notes = CASE WHEN excluded.notes != '' THEN excluded.notes ELSE projects.notes END,
                    next_step = CASE WHEN excluded.next_step != '' THEN excluded.next_step ELSE projects.next_step END,
                    updated_at = excluded.updated_at
                """,
                (name, goal, status, notes, next_step, updated_at),
            )

        row = self.conn.execute("SELECT * FROM projects WHERE name = ?", (name,)).fetchone()
        return {"status": "saved", "project": dict(row) if row else {}}

    def list_projects(self, status: str = "all") -> List[Dict[str, Any]]:
        """List tracked projects."""
        if status.lower() == "all":
            cursor = self.conn.execute("SELECT * FROM projects ORDER BY updated_at DESC")
        else:
            cursor = self.conn.execute(
                "SELECT * FROM projects WHERE status = ? ORDER BY updated_at DESC", (status.lower(),)
            )
        return [dict(r) for r in cursor.fetchall()]

    # -- Lessons / Teaching --------------------------------------------
    def save_lesson_progress(
        self,
        topic: str,
        step: int,
        steps_completed: str = "",
        where_stuck: str = "",
        level: str = "beginner",
    ) -> Dict[str, Any]:
        """Record teaching lesson progress so teaching can resume accurately."""
        topic = (topic or "").strip().lower()
        if not topic:
            return {"error": "Lesson topic is required"}

        updated_at = now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO lessons (topic, level, current_step, steps_completed, where_stuck, last_session)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(topic) DO UPDATE SET
                    level = excluded.level,
                    current_step = excluded.current_step,
                    steps_completed = CASE WHEN excluded.steps_completed != '' THEN excluded.steps_completed ELSE lessons.steps_completed END,
                    where_stuck = excluded.where_stuck,
                    last_session = excluded.last_session
                """,
                (topic, level, step, steps_completed, where_stuck, updated_at),
            )

        row = self.conn.execute("SELECT * FROM lessons WHERE topic = ?", (topic,)).fetchone()
        return {"status": "saved", "lesson": dict(row) if row else {}}

    def get_lesson(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get lesson progress for a specific topic."""
        topic = (topic or "").strip().lower()
        row = self.conn.execute("SELECT * FROM lessons WHERE topic = ?", (topic,)).fetchone()
        return dict(row) if row else None

    def get_active_lesson(self) -> Optional[Dict[str, Any]]:
        """Get the most recently updated lesson."""
        row = self.conn.execute(
            "SELECT * FROM lessons ORDER BY last_session DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    # -- Documentation Lookup ------------------------------------------
    def lookup_docs(self, query: str, topic: str = "") -> Dict[str, Any]:
        """Search local hardware documentation for pin numbers, voltages, and guides."""
        query = (query or "").strip()
        if not query:
            return {"error": "Query cannot be empty"}

        if not self.docs_dir.exists():
            return {
                "found": False,
                "message": f"Documentation folder '{self.docs_dir.name}' not found. Say 'I am not sure' rather than guessing.",
            }

        # Select target doc files
        doc_files = list(self.docs_dir.glob("*.md")) + list(self.docs_dir.glob("*.txt"))
        if topic:
            filtered = [f for f in doc_files if topic.lower() in f.stem.lower()]
            if filtered:
                doc_files = filtered

        if not doc_files:
            return {
                "found": False,
                "message": f"No documentation found for topic '{topic}'. Admit uncertainty rather than guessing pin numbers or voltages.",
            }

        matches: List[Dict[str, Any]] = []
        try:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
        except re.error:
            pattern = re.compile(query, re.IGNORECASE)

        for doc_file in doc_files:
            try:
                content = doc_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            lines = content.splitlines()
            for idx, line in enumerate(lines):
                if pattern.search(line):
                    # Extract surrounding context lines
                    start_idx = max(0, idx - 3)
                    end_idx = min(len(lines), idx + 4)
                    context_snippet = "\n".join(lines[start_idx:end_idx])
                    matches.append(
                        {
                            "file": doc_file.name,
                            "line": idx + 1,
                            "snippet": context_snippet,
                        }
                    )
                    if len(matches) >= 3:
                        break

        if not matches:
            return {
                "found": False,
                "query": query,
                "message": (
                    f"No mentions of '{query}' found in local docs. Say 'I am not sure' "
                    "instead of guessing hardware specifications."
                ),
            }

        return {
            "found": True,
            "query": query,
            "total_matches": len(matches),
            "matches": matches,
        }

    # -- Context builder for chat startup ------------------------------
    def get_startup_context(self) -> str:
        """Format known user facts and active lesson for session injection."""
        parts: List[str] = []

        # Facts
        facts_cur = self.conn.execute("SELECT key, value FROM facts ORDER BY updated_at DESC LIMIT 5")
        facts = facts_cur.fetchall()
        if facts:
            fact_lines = [f"- {r['key']}: {r['value']}" for r in facts]
            parts.append("Known facts about the user:\n" + "\n".join(fact_lines))

        # Active projects
        projects_cur = self.conn.execute(
            "SELECT name, goal, next_step FROM projects WHERE status = 'active' LIMIT 3"
        )
        projects = projects_cur.fetchall()
        if projects:
            proj_lines = [
                f"- {r['name']}: {r['goal']}" + (f" (next step: {r['next_step']})" if r['next_step'] else "")
                for r in projects
            ]
            parts.append("Active user projects:\n" + "\n".join(proj_lines))

        # Active lesson
        lesson = self.get_active_lesson()
        if lesson:
            parts.append(
                f"Active ongoing lesson: {lesson['topic']} (Step {lesson['current_step']}: "
                f"{lesson['steps_completed'] or 'in progress'}). If the user says 'continue' "
                f"or 'where were we', resume here."
            )

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Global Service Accessor
# ---------------------------------------------------------------------------

def configure(db_path: Optional[str | Path] = None, docs_dir: Optional[str | Path] = None) -> None:
    """Configure or reset memory service (used in tests)."""
    global _service, _db_path
    if _service:
        _service.close()
    _db_path = str(db_path) if db_path else None
    _service = MemoryService(db_path=db_path, docs_dir=docs_dir)


def get_service() -> MemoryService:
    global _service
    if _service is None:
        _service = MemoryService(db_path=_db_path)
    return _service


# ---------------------------------------------------------------------------
# LLM Tools
# ---------------------------------------------------------------------------

REMEMBER_FACT_SCHEMA = {
    "name": "remember_fact",
    "description": (
        "Remember a permanent fact about the user, their background, preferences, "
        "or hardware setup (e.g. key='board', value='ESP32 DevKit V1')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Identifier or subject of the fact (e.g. 'coding_experience', 'preferred_board')",
            },
            "value": {
                "type": "string",
                "description": "The fact or preference to remember",
            },
            "category": {
                "type": "string",
                "description": "Category (e.g. 'preference', 'hardware', 'background')",
                "default": "general",
            },
        },
        "required": ["key", "value"],
    },
}


@tool(name="remember_fact", schema=REMEMBER_FACT_SCHEMA)
def remember_fact(key: str, value: str, category: str = "general") -> Dict[str, Any]:
    """Remember a fact."""
    return get_service().remember_fact(key, value, category)


RECALL_SCHEMA = {
    "name": "recall",
    "description": (
        "Recall remembered facts about the user, their active projects, and lesson progress. "
        "Leave query empty to review all known context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional search term to filter facts",
                "default": "",
            },
        },
    },
}


@tool(name="recall", schema=RECALL_SCHEMA)
def recall(query: str = "") -> Dict[str, Any]:
    """Recall facts from memory."""
    return get_service().recall(query)


UPDATE_PROJECT_SCHEMA = {
    "name": "update_project",
    "description": (
        "Track or update a project the user is working on, including goal, status, and next step."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Project name (e.g. 'ESP32 Weather Station')",
            },
            "goal": {
                "type": "string",
                "description": "Primary goal or outcome of the project",
                "default": "",
            },
            "status": {
                "type": "string",
                "description": "Current status ('active', 'completed', 'paused')",
                "default": "active",
            },
            "next_step": {
                "type": "string",
                "description": "The immediate next action or task to try",
                "default": "",
            },
            "notes": {
                "type": "string",
                "description": "Additional notes, components, or details",
                "default": "",
            },
        },
        "required": ["name"],
    },
}


@tool(name="update_project", schema=UPDATE_PROJECT_SCHEMA)
def update_project(
    name: str,
    goal: str = "",
    status: str = "active",
    next_step: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    """Track or update a user project."""
    return get_service().update_project(name, goal, status, next_step, notes)


SAVE_LESSON_PROGRESS_SCHEMA = {
    "name": "save_lesson_progress",
    "description": (
        "Save step-by-step progress during an interactive teaching session "
        "(e.g. topic='esp32', step=1, steps_completed='Blinked onboard LED')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Topic of the lesson (e.g. 'esp32')",
            },
            "step": {
                "type": "integer",
                "description": "Current step number reached",
            },
            "steps_completed": {
                "type": "string",
                "description": "Summary of steps successfully finished so far",
                "default": "",
            },
            "where_stuck": {
                "type": "string",
                "description": "Details on confusion, error, or current obstacle (if any)",
                "default": "",
            },
            "level": {
                "type": "string",
                "description": "User comfort level ('beginner', 'intermediate')",
                "default": "beginner",
            },
        },
        "required": ["topic", "step"],
    },
}


@tool(name="save_lesson_progress", schema=SAVE_LESSON_PROGRESS_SCHEMA)
def save_lesson_progress(
    topic: str,
    step: int,
    steps_completed: str = "",
    where_stuck: str = "",
    level: str = "beginner",
) -> Dict[str, Any]:
    """Save progress for a learning topic."""
    return get_service().save_lesson_progress(topic, step, steps_completed, where_stuck, level)


LOOKUP_DOCS_SCHEMA = {
    "name": "lookup_docs",
    "description": (
        "Search local hardware documentation and datasheets in docs/ for pin numbers, "
        "voltage limits, GPIO rules, and wiring specifications. Always call this BEFORE "
        "advising on pins or voltages."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Pin number, peripheral, or specification (e.g. 'GPIO 2', 'operating voltage', 'ADC1', 'I2C')",
            },
            "topic": {
                "type": "string",
                "description": "Documentation topic or filename keyword (e.g. 'esp32')",
                "default": "",
            },
        },
        "required": ["query"],
    },
}


@tool(name="lookup_docs", schema=LOOKUP_DOCS_SCHEMA)
def lookup_docs(query: str, topic: str = "") -> Dict[str, Any]:
    """Search local documentation."""
    return get_service().lookup_docs(query, topic)
