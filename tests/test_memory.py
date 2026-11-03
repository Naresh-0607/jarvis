"""Unit tests for Phase 7 - Long-Term Memory, Hardware Docs & Interactive Teaching."""

from pathlib import Path
import pytest

from jarvis.memory import MemoryService, configure, get_service
from jarvis.chat import ChatSession


@pytest.fixture
def memory_sandbox(tmp_path):
    """Create an isolated SQLite database and test docs folder for testing."""
    db_file = tmp_path / "test_memory.db"
    docs_folder = tmp_path / "docs"
    docs_folder.mkdir()

    # Create a sample ESP32 doc
    esp32_doc = docs_folder / "esp32_reference.md"
    esp32_doc.write_text(
        "# ESP32 Reference\n"
        "- Logic Voltage: 3.3V (NOT 5V tolerant!)\n"
        "- Onboard LED: GPIO 2\n"
        "- ADC1: GPIO 32 to 39 (safe with Wi-Fi)\n"
        "- ADC2: cannot be used with Wi-Fi\n"
        "- I2C: SDA=GPIO 21, SCL=GPIO 22\n",
        encoding="utf-8",
    )

    configure(db_path=db_file, docs_dir=docs_folder)
    yield get_service()
    configure(db_path=None, docs_dir=None)


def test_remember_and_recall_facts(memory_sandbox):
    # Save a fact
    res1 = memory_sandbox.remember_fact("board", "ESP32 DevKit V1", category="hardware")
    assert res1["status"] == "saved"

    # Save second fact
    memory_sandbox.remember_fact("favorite_editor", "VS Code", category="tools")

    # Recall all
    all_facts = memory_sandbox.recall()
    assert all_facts["total_facts"] == 2
    keys = [f["key"] for f in all_facts["facts"]]
    assert "board" in keys
    assert "favorite_editor" in keys

    # Query filter
    filtered = memory_sandbox.recall("hardware")
    assert filtered["total_facts"] == 1
    assert filtered["facts"][0]["key"] == "board"

    # UPSERT update
    memory_sandbox.remember_fact("board", "ESP32-S3", category="hardware")
    updated = memory_sandbox.recall("board")
    assert updated["total_facts"] == 1
    assert updated["facts"][0]["value"] == "ESP32-S3"


def test_update_and_list_projects(memory_sandbox):
    # Create project
    res = memory_sandbox.update_project(
        name="Weather Node",
        goal="Read temperature and post to ntfy",
        status="active",
        next_step="Wire DHT22 sensor to GPIO 4",
    )
    assert res["status"] == "saved"
    assert res["project"]["name"] == "Weather Node"
    assert res["project"]["next_step"] == "Wire DHT22 sensor to GPIO 4"

    # List active
    active = memory_sandbox.list_projects("active")
    assert len(active) == 1
    assert active[0]["name"] == "Weather Node"

    # Update next step
    memory_sandbox.update_project(name="Weather Node", next_step="Test Wi-Fi connectivity")
    updated_proj = memory_sandbox.list_projects("active")[0]
    assert updated_proj["next_step"] == "Test Wi-Fi connectivity"


def test_save_lesson_progress_and_resume(memory_sandbox):
    # Step 1
    res1 = memory_sandbox.save_lesson_progress(
        topic="esp32",
        step=1,
        steps_completed="Blinked onboard LED on GPIO 2",
        where_stuck="",
    )
    assert res1["status"] == "saved"
    assert res1["lesson"]["current_step"] == 1

    # Check active lesson
    active = memory_sandbox.get_active_lesson()
    assert active is not None
    assert active["topic"] == "esp32"
    assert active["current_step"] == 1

    # Advance to Step 2
    memory_sandbox.save_lesson_progress(
        topic="esp32",
        step=2,
        steps_completed="Read push button on GPIO 4 with pull-down",
    )
    active2 = memory_sandbox.get_active_lesson()
    assert active2["current_step"] == 2
    assert "push button" in active2["steps_completed"]


def test_lookup_docs_success(memory_sandbox):
    res_voltage = memory_sandbox.lookup_docs("3.3V")
    assert res_voltage["found"] is True
    assert res_voltage["total_matches"] >= 1
    assert "3.3V" in res_voltage["matches"][0]["snippet"]

    res_led = memory_sandbox.lookup_docs("GPIO 2")
    assert res_led["found"] is True
    assert "GPIO 2" in res_led["matches"][0]["snippet"]


def test_lookup_docs_not_found(memory_sandbox):
    res = memory_sandbox.lookup_docs("Raspberry Pi Pico")
    assert res["found"] is False
    assert "not sure" in res["message"].lower()


def test_startup_context_builder(memory_sandbox):
    memory_sandbox.remember_fact("experience", "Intermediate Python")
    memory_sandbox.update_project("Robot Car", goal="Line follower", next_step="Solder motor driver")
    memory_sandbox.save_lesson_progress("esp32", step=1, steps_completed="LED Blink")

    ctx = memory_sandbox.get_startup_context()
    assert "experience: Intermediate Python" in ctx
    assert "Robot Car: Line follower" in ctx
    assert "Active ongoing lesson: esp32" in ctx


def test_chat_lesson_resume_trigger(memory_sandbox):
    session = ChatSession()
    memory_sandbox.save_lesson_progress("esp32", step=2, steps_completed="Button input")

    # Non-trigger phrase
    assert session._check_lesson_resume("hello what is your name?") is None

    # Trigger phrases
    res1 = session._check_lesson_resume("where were we?")
    assert res1 is not None
    assert "esp32" in res1
    assert "Button input" in res1

    res2 = session._check_lesson_resume("let's continue")
    assert res2 is not None
    assert "esp32" in res2
