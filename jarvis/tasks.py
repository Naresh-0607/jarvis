"""Task and reminder logic.

All times are handled in Asia/Kolkata. Reminder slots are pure arithmetic -
no model, no network - so they keep working offline.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

from jarvis.db import connect

TZ = ZoneInfo("Asia/Kolkata")

# Offsets before the deadline that always get a reminder.
BASE_OFFSETS = (
    timedelta(days=1),
    timedelta(hours=1),
    timedelta(minutes=10),
    timedelta(0),
)
# Extra nudge for important tasks.
IMPORTANT_OFFSET = timedelta(hours=3)
IMPORTANT_LEVELS = {"important", "high", "urgent"}


def now(reference: Optional[datetime] = None) -> datetime:
    """Current time in the project timezone."""
    if reference is not None:
        return as_aware(reference)
    return datetime.now(TZ)


def as_aware(value: datetime) -> datetime:
    """Attach the project timezone to naive datetimes."""
    if value.tzinfo is None:
        return value.replace(tzinfo=TZ)
    return value.astimezone(TZ)


def parse_deadline(value: Any) -> Optional[datetime]:
    """Parse an ISO string, a datetime, or None into an aware datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return as_aware(value)
    try:
        return as_aware(datetime.fromisoformat(str(value)))
    except ValueError:
        return None


def reminder_slots(
    deadline: datetime,
    important: bool = False,
    repeat_minutes: int = 30,
    until: Optional[datetime] = None,
) -> List[datetime]:
    """Every reminder slot for a task, including overdue repeats.

    ``until`` extends the list with the repeating nudges that follow the
    deadline, so the scheduler can tell what *should* have fired while the
    laptop was off.
    """
    due = as_aware(deadline)
    offsets = list(BASE_OFFSETS)
    if important:
        offsets.append(IMPORTANT_OFFSET)

    slots = {due - offset for offset in offsets}
    if until is not None:
        moment = due + timedelta(minutes=max(1, repeat_minutes))
        limit = as_aware(until)
        while moment <= limit:
            slots.add(moment)
            moment += timedelta(minutes=max(1, repeat_minutes))

    return sorted(slots)


def reminder_times(
    deadline: datetime,
    important: bool = False,
    reference: Optional[datetime] = None,
    repeat_minutes: int = 30,
) -> List[datetime]:
    """Reminder slots for a task, skipping any that are already past.

    One day, one hour and ten minutes before, then at the deadline itself.
    Important tasks also get a nudge three hours before. The overdue repeat
    is handled by the scheduler's periodic scan, not here.
    """
    due = as_aware(deadline)
    start = now(reference)

    offsets = list(BASE_OFFSETS)
    if important:
        offsets.append(IMPORTANT_OFFSET)

    slots = sorted({due - offset for offset in offsets})
    return [slot for slot in slots if slot > start]


def human_deadline(deadline: Optional[datetime]) -> str:
    """Plain-language deadline, for reading back to the user."""
    if deadline is None:
        return "no deadline"
    due = as_aware(deadline)
    today = now().date()
    if due.date() == today:
        day = "today"
    elif due.date() == today + timedelta(days=1):
        day = "tomorrow"
    else:
        day = due.strftime("%A")
    time_part = due.strftime("%I:%M %p").lstrip("0")
    return f"{day} at {time_part}"


class TaskService:
    """Small facade over the SQLite task tables."""

    def __init__(self, db_path: Optional[Path | str] = None, conn: Optional[sqlite3.Connection] = None):
        self.conn = conn or connect(db_path)

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------
    def add_task(
        self,
        title: str,
        deadline: Any = None,
        notes: str = "",
        important: bool = False,
    ) -> Dict[str, Any]:
        title = (title or "").strip()
        if not title:
            raise ValueError("title is required")

        due = parse_deadline(deadline)
        created = now().isoformat(timespec="seconds")
        cursor = self.conn.execute(
            """
            INSERT INTO tasks (title, notes, deadline, importance, status, created_at)
            VALUES (?, ?, ?, ?, 'open', ?)
            """,
            (
                title,
                notes or "",
                due.isoformat(timespec="seconds") if due else None,
                "important" if important else "normal",
                created,
            ),
        )
        self.conn.commit()
        task = self.get_task(cursor.lastrowid)
        self._mark_past_slots(task["id"], due, important)
        return task

    def _mark_past_slots(
        self, task_id: int, due: Optional[datetime], important: bool
    ) -> None:
        """Silence reminder slots that were already past when the task was made."""
        if due is None:
            return
        created = now()
        for slot in reminder_slots(due, important=important):
            if slot <= created:
                self.mark_sent(task_id, slot)

    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return self._row_to_task(row) if row else None

    def _row_to_task(self, row: sqlite3.Row) -> Dict[str, Any]:
        due = parse_deadline(row["deadline"])
        return {
            "id": row["id"],
            "title": row["title"],
            "notes": row["notes"],
            "deadline": due.isoformat(timespec="seconds") if due else None,
            "deadline_human": human_deadline(due),
            "importance": row["importance"],
            "status": row["status"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
        }

    def list_tasks(
        self,
        range: str = "all",  # noqa: A002 - matches the tool schema
        reference: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        current = now(reference)
        range = (range or "all").lower()

        if range == "today":
            start = current.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
            rows = self.conn.execute(
                """
                SELECT * FROM tasks
                WHERE status = 'open' AND deadline >= ? AND deadline < ?
                ORDER BY deadline
                """,
                (start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")),
            ).fetchall()

        elif range == "week":
            end = current + timedelta(days=7)
            rows = self.conn.execute(
                """
                SELECT * FROM tasks
                WHERE status = 'open' AND deadline IS NOT NULL
                  AND deadline >= ? AND deadline < ?
                ORDER BY deadline
                """,
                (current.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")),
            ).fetchall()

        elif range == "overdue":
            rows = self.conn.execute(
                """
                SELECT * FROM tasks
                WHERE status = 'open' AND deadline IS NOT NULL AND deadline < ?
                ORDER BY deadline
                """,
                (current.isoformat(timespec="seconds"),),
            ).fetchall()

        elif range == "all":
            rows = self.conn.execute(
                """
                SELECT * FROM tasks
                WHERE status = 'open'
                ORDER BY (deadline IS NULL), deadline
                """
            ).fetchall()

        else:
            raise ValueError(
                "range must be one of: today, week, overdue, all"
            )

        return [self._row_to_task(row) for row in rows]

    def complete_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            """
            UPDATE tasks SET status = 'done', completed_at = ?
            WHERE id = ? AND status = 'open'
            """,
            (now().isoformat(timespec="seconds"), task_id),
        )
        self.conn.commit()
        if cursor.rowcount == 0:
            return None
        return self.get_task(task_id)

    def snooze_task(self, task_id: int, minutes: int) -> Optional[Dict[str, Any]]:
        task = self.get_task(task_id)
        if not task or not task["deadline"]:
            return None

        current = now()
        due = parse_deadline(task["deadline"])
        base = due if due > current else current
        new_due = base + timedelta(minutes=max(1, int(minutes)))

        self.conn.execute(
            "UPDATE tasks SET deadline = ? WHERE id = ?",
            (new_due.isoformat(timespec="seconds"), task_id),
        )
        # Clear fired slots so the new deadline can remind again.
        self.conn.execute("DELETE FROM reminders_sent WHERE task_id = ?", (task_id,))
        self.conn.commit()
        self._mark_past_slots(
            task_id, new_due, task["importance"] in IMPORTANT_LEVELS
        )
        return self.get_task(task_id)

    def delete_task(self, task_id: int) -> bool:
        cursor = self.conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Reminder bookkeeping
    # ------------------------------------------------------------------
    def was_sent(self, task_id: int, fire_at: datetime) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM reminders_sent WHERE task_id = ? AND fire_at = ?",
            (task_id, fire_at.isoformat(timespec="seconds")),
        ).fetchone()
        return row is not None

    def mark_sent(self, task_id: int, fire_at: datetime) -> bool:
        """Record a fired reminder. Returns False if it had already fired."""
        try:
            self.conn.execute(
                """
                INSERT INTO reminders_sent (task_id, fire_at, sent_at)
                VALUES (?, ?, ?)
                """,
                (
                    task_id,
                    fire_at.isoformat(timespec="seconds"),
                    now().isoformat(timespec="seconds"),
                ),
            )
        except sqlite3.IntegrityError:
            return False
        self.conn.commit()
        return True
