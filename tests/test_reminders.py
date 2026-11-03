"""Tests for reminder-time calculation and the task database."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from jarvis.scheduler import ReminderScheduler
from jarvis.tasks import (
    TaskService,
    parse_deadline,
    reminder_slots,
    reminder_times,
)

TZ = ZoneInfo("Asia/Kolkata")
REF = datetime(2026, 10, 29, 14, 0, tzinfo=TZ)


@pytest.fixture
def service(tmp_path):
    svc = TaskService(tmp_path / "tasks.db")
    yield svc
    svc.close()


@pytest.fixture
def delivered():
    messages = []

    def _deliver(message: str) -> None:
        messages.append(message)

    return messages, _deliver


# ---------------------------------------------------------------------------
# Reminder-time maths
# ---------------------------------------------------------------------------
def test_slots_for_task_due_in_three_hours():
    due = REF + timedelta(hours=3)
    slots = reminder_times(due, reference=REF)
    assert slots == [due - timedelta(hours=1), due - timedelta(minutes=10), due]


def test_task_due_in_under_an_hour_skips_the_early_slot():
    due = REF + timedelta(minutes=20)
    slots = reminder_times(due, reference=REF)
    assert slots == [due - timedelta(minutes=10), due]


def test_overdue_task_has_no_future_slots():
    due = REF - timedelta(hours=2)
    assert reminder_times(due, reference=REF) == []


def test_important_task_gets_an_extra_three_hour_nudge():
    due = REF + timedelta(hours=6)
    normal = reminder_times(due, reference=REF)
    important = reminder_times(due, important=True, reference=REF)
    assert due - timedelta(hours=3) in important
    assert due - timedelta(hours=3) not in normal


def test_past_slots_are_skipped_when_task_is_created_late():
    due = REF + timedelta(hours=2)
    slots = reminder_times(due, reference=REF)
    assert all(slot > REF for slot in slots)
    assert due - timedelta(days=1) not in slots


def test_reminder_slots_include_overdue_repeats():
    due = REF
    slots = reminder_slots(due, repeat_minutes=30, until=due + timedelta(hours=2))
    repeats = [slot for slot in slots if slot > due]
    assert repeats == [
        due + timedelta(minutes=30),
        due + timedelta(minutes=60),
        due + timedelta(minutes=90),
        due + timedelta(minutes=120),
    ]


def test_naive_deadline_is_treated_as_kolkata_time():
    parsed = parse_deadline("2026-10-29T17:00:00")
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(hours=5, minutes=30)
    assert parsed.hour == 17


# ---------------------------------------------------------------------------
# Database behaviour
# ---------------------------------------------------------------------------
def test_add_and_list_tasks(service):
    due = datetime.now(TZ) + timedelta(minutes=5)
    task = service.add_task("Buy milk", deadline=due, notes="2 litres")
    assert task["id"] > 0
    assert task["status"] == "open"
    assert task["deadline_human"].startswith("today")

    listed = service.list_tasks("all")
    assert [t["title"] for t in listed] == ["Buy milk"]


def test_list_ranges(service):
    now = datetime.now(TZ)
    service.add_task("later today", deadline=now + timedelta(hours=1))
    service.add_task("next week", deadline=now + timedelta(days=10))
    service.add_task("no deadline")
    service.add_task("overdue", deadline=now - timedelta(hours=1))

    today = [t["title"] for t in service.list_tasks("today")]
    overdue = [t["title"] for t in service.list_tasks("overdue")]
    week = [t["title"] for t in service.list_tasks("week")]

    assert "later today" in today
    assert "next week" not in today
    assert overdue == ["overdue"]
    assert "next week" not in week  # ten days out


def test_complete_and_snooze(service):
    now = datetime.now(TZ)
    task = service.add_task("write report", deadline=now + timedelta(minutes=10))

    snoozed = service.snooze_task(task["id"], 60)
    assert parse_deadline(snoozed["deadline"]) >= now + timedelta(minutes=60)

    done = service.complete_task(task["id"])
    assert done["status"] == "done"
    assert done["completed_at"] is not None
    assert service.list_tasks("all") == []
    assert service.complete_task(task["id"]) is None


def test_snooze_clears_sent_reminders(service):
    task = service.add_task("call mom", deadline=datetime.now(TZ) + timedelta(hours=1))
    slot = parse_deadline(task["deadline"]) - timedelta(minutes=10)
    assert service.mark_sent(task["id"], slot) is True

    service.snooze_task(task["id"], 30)
    assert service.was_sent(task["id"], slot) is False


def test_reminder_is_only_sent_once(service):
    task = service.add_task("stretch", deadline=datetime.now(TZ) + timedelta(minutes=5))
    slot = datetime.now(TZ)

    assert service.mark_sent(task["id"], slot) is True
    assert service.mark_sent(task["id"], slot) is False
    assert service.was_sent(task["id"], slot) is True


def test_delete_task(service):
    task = service.add_task("delete me")
    assert service.delete_task(task["id"]) is True
    assert service.get_task(task["id"]) is None
    assert service.delete_task(9999) is False


# ---------------------------------------------------------------------------
# Scheduler behaviour
# ---------------------------------------------------------------------------
def make_scheduler(service, delivered, cfg=None):
    messages, deliver = delivered
    return ReminderScheduler(
        service,
        config={"repeat_overdue_minutes": 30, **(cfg or {})},
        deliver=deliver,
        jobstore_path=None,
    )


def test_tick_announces_upcoming_reminder(service, delivered):
    from unittest.mock import patch
    with patch("jarvis.tasks.now", return_value=REF - timedelta(seconds=1)):
        due = REF + timedelta(minutes=10)
        service.add_task("standup", deadline=due)

        scheduler = make_scheduler(service, delivered)
        lines = scheduler.tick(reference=REF)

    assert len(lines) == 1
    assert "standup" in lines[0]
    messages, _ = delivered
    assert messages == [lines[0]]


def test_task_created_late_does_not_fire_already_past_slots(service, delivered):
    # Due in three hours, so the "one day before" slot is already gone.
    service.add_task("later today", deadline=datetime.now(TZ) + timedelta(hours=3))

    scheduler = make_scheduler(service, delivered)
    assert scheduler.tick() == []


def test_tick_groups_missed_reminders_into_one_message(service, delivered):
    service.add_task("task one", deadline=REF + timedelta(minutes=5))
    service.add_task("task two", deadline=REF + timedelta(minutes=15))

    scheduler = make_scheduler(service, delivered)
    # Three hours later: both were missed while the laptop was off.
    lines = scheduler.tick(reference=REF + timedelta(hours=3))

    messages, _ = delivered
    assert len(messages) == 1
    assert "task one" in messages[0] and "task two" in messages[0]
    assert len(lines) == 2


def test_tick_does_not_repeat_or_flood(service, delivered):
    service.add_task("overdue thing", deadline=REF + timedelta(minutes=5))
    scheduler = make_scheduler(service, delivered)

    first = scheduler.tick(reference=REF + timedelta(hours=3))
    second = scheduler.tick(reference=REF + timedelta(hours=3))

    assert len(first) == 1  # one line, not one per 30-minute slot
    assert second == []  # the newest slot is already recorded


def test_overdue_repeat_fires_after_the_interval(service, delivered):
    service.add_task("keep nudging", deadline=REF)
    scheduler = make_scheduler(service, delivered)

    assert len(scheduler.tick(reference=REF + timedelta(minutes=31))) == 1
    assert len(scheduler.tick(reference=REF + timedelta(minutes=32))) == 0
    assert len(scheduler.tick(reference=REF + timedelta(minutes=62))) == 1


def test_completed_task_is_never_announced(service, delivered):
    task = service.add_task("already done", deadline=REF + timedelta(minutes=5))
    service.complete_task(task["id"])

    scheduler = make_scheduler(service, delivered)
    assert scheduler.tick(reference=REF + timedelta(hours=2)) == []


def test_briefing_mentions_today_and_overdue(service, delivered):
    service.add_task("today thing", deadline=datetime.now(TZ) + timedelta(minutes=5))
    service.add_task("late thing", deadline=datetime.now(TZ) - timedelta(days=1))

    scheduler = make_scheduler(service, delivered)
    message = scheduler.briefing()

    assert "today thing" in message
    assert "late thing" in message
