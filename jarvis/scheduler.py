"""Reminder scheduler.

APScheduler runs two jobs:

* a short tick that looks for reminder slots that are due and announces them
  (one grouped message per tick, never twice - the database remembers what
  has already fired);
* a daily morning briefing.

Jobs live in a SQLite job store so they survive a restart, and the "already
sent" bookkeeping lives in the tasks database, which is what actually
prevents duplicates.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from jarvis import notify
from jarvis.tasks import (
    IMPORTANT_LEVELS,
    TZ,
    TaskService,
    human_deadline,
    now,
    parse_deadline,
    reminder_slots,
)

_logger = logging.getLogger("jarvis.scheduler")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBSTORE = PROJECT_ROOT / "data" / "scheduler.db"

# Module-level entry points: APScheduler's SQLite job store needs picklable
# references, and bound methods of a live object are not picklable.
_active: Optional["ReminderScheduler"] = None


def _tick_job() -> None:
    if _active is not None:
        _active.tick()


def _briefing_job() -> None:
    if _active is not None:
        _active.briefing()


class ReminderScheduler:
    """Owns the clock. Plain code, no model, no network dependency."""

    def __init__(
        self,
        service: TaskService,
        config: Optional[Dict] = None,
        deliver: Optional[Callable[[str], None]] = None,
        jobstore_path: Optional[Path | str] = DEFAULT_JOBSTORE,
    ):
        self.service = service
        self.config = config or {}
        self.repeat_minutes = int(self.config.get("repeat_overdue_minutes", 30))
        self.tick_seconds = int(self.config.get("tick_seconds", 20))
        self.briefing_time = str(self.config.get("morning_briefing") or "").strip()
        self.deliver = deliver or (lambda message: notify.deliver(message))
        self.started = False

        jobstores = {}
        if jobstore_path:
            path = Path(jobstore_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            jobstores["default"] = SQLAlchemyJobStore(url=f"sqlite:///{path.as_posix()}")

        scheduler_kwargs = {"timezone": TZ}
        if jobstores:
            scheduler_kwargs["jobstores"] = jobstores
        self.scheduler = BackgroundScheduler(**scheduler_kwargs)

        self.scheduler.add_job(
            _tick_job,
            "interval",
            seconds=self.tick_seconds,
            id="reminder_tick",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

        if self.briefing_time:
            hour, minute = self._parse_briefing_time(self.briefing_time)
            self.scheduler.add_job(
                _briefing_job,
                CronTrigger(hour=hour, minute=minute, timezone=TZ),
                id="morning_briefing",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )

    @staticmethod
    def _parse_briefing_time(value: str) -> tuple:
        try:
            hour, minute = value.split(":")
            hour_i, minute_i = int(hour), int(minute)
            if 0 <= hour_i < 24 and 0 <= minute_i < 60:
                return hour_i, minute_i
        except (ValueError, TypeError):
            pass
        _logger.warning("Invalid morning_briefing %r; using 08:00", value)
        return 8, 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        global _active
        _active = self
        if not self.started:
            self.scheduler.start()
            self.started = True
        # Announce anything that came due while the laptop was off.
        self.tick()

    def shutdown(self) -> None:
        global _active
        if self.started:
            self.scheduler.shutdown(wait=False)
            self.started = False
        if _active is self:
            _active = None

    # ------------------------------------------------------------------
    # Work
    # ------------------------------------------------------------------
    def _phrase(self, task: Dict, due: datetime, slot: datetime, current: datetime) -> str:
        title = task["title"]
        if due < current:
            minutes = int((current - due).total_seconds() // 60)
            if minutes >= 90:
                late = f"{minutes // 60} hours ago"
            else:
                late = f"{minutes} minutes ago"
            return f"Overdue: {title} (was due {human_deadline(due)}, {late})."
        if slot == due:
            return f"Due now: {title}."
        return f"Coming up: {title}, due {human_deadline(due)}."

    def due_lines(self, reference: Optional[datetime] = None) -> List[str]:
        """Collect and mark every reminder slot that is due right now."""
        current = reference or now()
        lines: List[str] = []

        for task in self.service.list_tasks("all"):
            due = parse_deadline(task["deadline"])
            if not due:
                continue

            important = task["importance"] in IMPORTANT_LEVELS
            slots = reminder_slots(
                due,
                important=important,
                repeat_minutes=self.repeat_minutes,
                until=current,
            )
            unsent = [
                slot
                for slot in slots
                if slot <= current and not self.service.was_sent(task["id"], slot)
            ]
            if not unsent:
                continue

            # If several slots piled up (laptop was off), send one line and
            # mark the stale ones so nothing repeats.
            latest = unsent[-1]
            for stale in unsent[:-1]:
                self.service.mark_sent(task["id"], stale)
            if not self.service.mark_sent(task["id"], latest):
                continue

            lines.append(self._phrase(task, due, latest, current))

        return lines

    def tick(self, reference: Optional[datetime] = None) -> List[str]:
        """Announce due reminders as one grouped message."""
        lines = self.due_lines(reference)
        if lines:
            message = "\n".join(lines)
            _logger.info("Reminder: %s", message.replace("\n", " | "))
            self.deliver(message)
        return lines

    def briefing(self) -> str:
        """Summarise today's and overdue tasks (plain code, no model)."""
        today = self.service.list_tasks("today")
        overdue = self.service.list_tasks("overdue")

        if not today and not overdue:
            message = "Morning briefing: nothing due today and nothing overdue."
            self.deliver(message)
            return message

        parts = ["Morning briefing."]
        if overdue:
            names = "; ".join(
                f"{t['title']} ({t['deadline_human']})" for t in overdue[:5]
            )
            parts.append(f"Overdue: {names}.")
        if today:
            names = "; ".join(
                f"{t['title']} at {t['deadline_human'].split(' at ')[-1]}" for t in today[:8]
            )
            parts.append(f"Today: {names}.")

        message = " ".join(parts)
        self.deliver(message)
        return message
