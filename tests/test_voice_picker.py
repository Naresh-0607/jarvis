"""Unit and integration tests for Voice Picker, Settings persistence, and Mid-Session switching."""

import pytest
from textual.app import App, ComposeResult

from jarvis.db import get_setting, set_setting
from jarvis.chat import ChatSession
from jarvis.tui import JarvisTUIApp, StartupSelectorModal, StatusIndicator


def test_sqlite_settings_persistence(tmp_path):
    """Test get_setting and set_setting in SQLite data store."""
    db_file = tmp_path / "test_settings.db"

    # Initially missing key returns default
    assert get_setting("mode", default="online", db_path=db_file) == "online"
    assert get_setting("voice", db_path=db_file) is None

    # Save values
    set_setting("mode", "local", db_path=db_file)
    set_setting("voice", "shadow", db_path=db_file)

    assert get_setting("mode", db_path=db_file) == "local"
    assert get_setting("voice", db_path=db_file) == "shadow"

    # Update existing value (upsert)
    set_setting("voice", "calm", db_path=db_file)
    assert get_setting("voice", db_path=db_file) == "calm"


def test_chat_session_persona_swapping_preserves_history(tmp_path):
    """Switching voice preset mid-session updates system persona without clearing chat history."""
    alt_persona = tmp_path / "persona_alt.md"
    alt_persona.write_text("# Alt Persona\nYou speak concisely.", encoding="utf-8")

    session = ChatSession()
    session.cfg = {
        "voice": {
            "default_preset": "calm",
            "presets": {
                "calm": {
                    "name": "calm",
                    "persona": "jarvis/persona.md",
                },
                "alt": {
                    "name": "alt",
                    "persona": str(alt_persona),
                },
            },
        }
    }
    # Manually initialize messages and persona
    persona_calm = session._build_persona("calm")
    session.messages.append({"role": "system", "content": persona_calm})

    # Add conversation history
    session.messages.append({"role": "user", "content": "Hello Jarvis"})
    session.messages.append({"role": "assistant", "content": "Greetings."})

    assert len(session.messages) == 3
    assert session.messages[0]["role"] == "system"
    assert "calm, warm" in session.messages[0]["content"]

    # Swap voice to alt
    session.set_voice_preset("alt")

    # Conversation history must remain intact!
    assert len(session.messages) == 3
    assert session.messages[1] == {"role": "user", "content": "Hello Jarvis"}
    assert session.messages[2] == {"role": "assistant", "content": "Greetings."}

    # System message must now contain alt persona
    assert session.messages[0]["role"] == "system"
    assert "Alt Persona" in session.messages[0]["content"]


@pytest.mark.asyncio
async def test_status_indicator_shows_voice():
    """StatusIndicator displays both mode and voice preset inside app."""
    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield StatusIndicator(id="status")

    app = TestApp()
    async with app.run_test() as pilot:
        indicator = app.query_one("#status", StatusIndicator)
        indicator.set_status("online", "online", voice="calm")
        rendered = str(indicator.render())
        assert "(o) online / calm" in rendered

        indicator.set_status("local", "local", voice="shadow")
        rendered = str(indicator.render())
        assert "(o) local / shadow" in rendered

        indicator.set_status("error", "error", reason="timeout")
        rendered = str(indicator.render())
        assert "(o) error: timeout" in rendered


@pytest.mark.asyncio
async def test_startup_selector_modal_navigation_and_confirm():
    """StartupSelectorModal allows toggling mode and cycling voice."""
    cfg = {
        "voice": {
            "default_preset": "calm",
            "presets": {
                "calm": {
                    "name": "calm",
                    "piper_model": "models/piper/alba-medium.onnx",
                },
                "shadow": {
                    "name": "shadow",
                    "piper_model": "models/piper/alba-medium.onnx",
                },
                "missing": {
                    "name": "missing",
                    "piper_model": "models/piper/nonexistent-model.onnx",
                },
            },
        }
    }
    modal = StartupSelectorModal(default_mode="online", default_voice="calm", cfg=cfg)

    class TestApp(App):
        def on_mount(self):
            self.push_screen(modal)

    app = TestApp()
    async with app.run_test() as pilot:
        # Initial state
        assert modal.active_row == 0
        assert modal.current_mode == "online"
        assert modal.current_voice == "calm"

        # Toggle mode option (Left/Right)
        modal.action_next_opt()
        assert modal.current_mode == "local"
        modal.action_prev_opt()
        assert modal.current_mode == "online"

        # Switch active row to VOICE (Down)
        modal.action_next_row()
        assert modal.active_row == 1

        # Cycle voice options
        modal.action_next_opt()
        assert modal.current_voice == "shadow"

        # Missing model is unselectable
        modal.action_next_opt()
        assert "missing" in modal.voice_presets[modal.voice_index][0]
        # Attempt to confirm on missing model fails
        modal.action_confirm()
        assert modal.is_current  # not dismissed!

        # Move back to selectable shadow
        modal.action_prev_opt()
        assert modal.current_voice == "shadow"
        modal.action_confirm()


@pytest.mark.asyncio
async def test_tui_app_voice_toggle_ctrl_v(tmp_path):
    """TUI app cycles voice with action_toggle_voice (Ctrl+V) and updates status."""
    app = JarvisTUIApp(skip_prompt=True, init_services=False)
    async with app.run_test() as pilot:
        app.session.validate_mode = lambda mode: (True, "")
        app.session.setup_providers = lambda mode: None
        app.cfg = {
            "ui": {"skip_mode_prompt": True, "default_mode": "online"},
            "voice": {
                "default_preset": "calm",
                "presets": {
                    "calm": {"name": "calm", "piper_model": "models/piper/alba-medium.onnx"},
                    "shadow": {"name": "shadow", "piper_model": "models/piper/alba-medium.onnx"},
                },
            },
        }
        await app.apply_mode("online")
        app.apply_voice("calm")
        assert app.selected_voice == "calm"

        status_text = str(app.status_indicator.render())
        assert "online / calm" in status_text

        # Toggle voice (Ctrl+V)
        app.action_toggle_voice()
        assert app.selected_voice == "shadow"

        status_text = str(app.status_indicator.render())
        assert "online / shadow" in status_text
        assert app.session.voice_preset == "shadow"
