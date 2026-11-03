"""SQLite storage for tasks and sent reminders."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "jarvis.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT    NOT NULL,
    notes        TEXT    NOT NULL DEFAULT '',
    deadline     TEXT,                          -- ISO 8601, Asia/Kolkata
    importance   TEXT    NOT NULL DEFAULT 'normal',   -- normal | important
    status       TEXT    NOT NULL DEFAULT 'open',     -- open | done
    created_at   TEXT    NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS reminders_sent (
    task_id INTEGER NOT NULL,
    fire_at TEXT    NOT NULL,                   -- the slot that fired
    sent_at TEXT    NOT NULL,
    PRIMARY KEY (task_id, fire_at),
    FOREIGN KEY (task_id) REFERENCES tasks (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);
CREATE INDEX IF NOT EXISTS idx_tasks_deadline ON tasks (deadline);

CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT    NOT NULL UNIQUE,
    value      TEXT    NOT NULL,
    category   TEXT    NOT NULL DEFAULT 'general',
    updated_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL UNIQUE,
    goal       TEXT    NOT NULL DEFAULT '',
    status     TEXT    NOT NULL DEFAULT 'active',
    notes      TEXT    NOT NULL DEFAULT '',
    next_step  TEXT    NOT NULL DEFAULT '',
    updated_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS lessons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic           TEXT    NOT NULL UNIQUE,
    level           TEXT    NOT NULL DEFAULT 'beginner',
    current_step    INTEGER NOT NULL DEFAULT 0,
    steps_completed TEXT    NOT NULL DEFAULT '',
    where_stuck     TEXT    NOT NULL DEFAULT '',
    last_session    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_key ON facts (key);
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects (status);
CREATE INDEX IF NOT EXISTS idx_lessons_topic ON lessons (topic);

CREATE TABLE IF NOT EXISTS search_cache (
    query_norm  TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_usage (
    month    TEXT NOT NULL,
    provider TEXT NOT NULL,
    credits  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (month, provider)
);

CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def connect(db_path: Optional[Path | str] = None) -> sqlite3.Connection:
    """Open (and if needed create) the tasks database."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def get_setting(
    key: str, default: Optional[str] = None, db_path: Optional[Path | str] = None
) -> Optional[str]:
    """Retrieve a persisted setting value by key."""
    conn = connect(db_path)
    try:
        cur = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
        return str(row["value"]) if row else default
    finally:
        conn.close()


def set_setting(key: str, value: str, db_path: Optional[Path | str] = None) -> None:
    """Store or update a setting value by key."""
    from datetime import datetime, timezone

    now_iso = datetime.now(timezone.utc).isoformat()
    conn = connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, str(value), now_iso),
        )
        conn.commit()
    finally:
        conn.close()
