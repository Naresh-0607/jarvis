"""Manual Phase 2 smoke test.

Uses a throwaway database so your real task list is untouched, then proves
that rows survive a "restart" (a fresh connection to the same file).

    python scripts/smoke_tasks.py            # no phone push
    python scripts/smoke_tasks.py --notify   # also sends one test push
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from jarvis import notify, task_tools
from jarvis.scheduler import ReminderScheduler
from jarvis.tasks import TZ, TaskService, parse_deadline, reminder_times

DB = project_root / "data" / "smoke_tasks.db"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    DB.unlink(missing_ok=True)
    task_tools.configure(str(DB))

    due = (datetime.now(TZ) + timedelta(hours=3)).replace(second=0, microsecond=0)
    print(f"Task due: {due.isoformat(timespec='seconds')}")

    print("\n1) add_task without confirmation (should NOT save):")
    preview = task_tools.add_task("Call the dentist", deadline=due.isoformat())
    print("   ", preview["status"], "->", preview["read_back"])

    print("\n2) add_task after confirming:")
    saved = task_tools.add_task("Call the dentist", deadline=due.isoformat(), confirm=True)
    print("   ", saved["status"], "->", saved["task"]["title"], saved["task"]["deadline_human"])
    task_id = saved["task"]["id"]

    print("\n3) Reminder slots that will fire:")
    for slot in reminder_times(due):
        print("   ", slot.strftime("%a %d %b %H:%M"))

    print("\n4) Scheduler tick right now (nothing due yet):")
    service = TaskService(DB)
    messages = []
    scheduler = ReminderScheduler(
        service,
        config={"repeat_overdue_minutes": 30},
        deliver=messages.append,
        jobstore_path=None,
    )
    print("   ", scheduler.tick() or "no reminders due")

    print("\n5) Simulating the laptop being off until the deadline:")
    lines = scheduler.tick(reference=due + timedelta(minutes=5))
    print("   ", lines)

    print("\n6) list_tasks('today'):")
    print("   ", task_tools.list_tasks("today"))

    service.close()

    print("\n7) Reopening the database (simulates a restart):")
    reopened = TaskService(DB)
    tasks = reopened.list_tasks("all")
    print("    persisted tasks:", [t["title"] for t in tasks])
    task = tasks[0]

    print("\n8) delete_task needs confirmation:")
    print("   ", task_tools.delete_task(task["id"]))
    print("   ", task_tools.delete_task(task["id"], confirmed=True))
    reopened.close()

    if "--notify" in sys.argv:
        print("\n9) Sending a test ntfy push (check your phone)...")
        ok = notify.send_ntfy("Jarvis test reminder - phase 2 delivery works.")
        print("    push sent:", ok)
    else:
        print("\n9) Skipped the ntfy push (pass --notify to send one).")

    print("\nSmoke test finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
