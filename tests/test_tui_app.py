"""Integration tests for JarvisTUIApp with Textual's async test harness."""

import pytest
from jarvis.tui import JarvisTUIApp, StartupModeModal, OrbWidget, StatusIndicator, NotificationPanel


@pytest.mark.asyncio
async def test_tui_app_mount_and_widgets():
    app = JarvisTUIApp(skip_prompt=True, init_services=False)
    async with app.run_test() as pilot:
        # Check all core widgets exist
        assert app.query_one("#app-title") is not None
        assert app.query_one("#notifications-slot", NotificationPanel) is not None
        assert app.query_one("#orb", OrbWidget) is not None
        assert app.query_one("#prompt-symbol") is not None
        assert app.query_one("#user-input") is not None
        assert app.query_one("#status-indicator", StatusIndicator) is not None

        # Check notification panel is hidden by default
        notif = app.query_one("#notifications-slot", NotificationPanel)
        assert not notif.display

        # Test pushing a notification
        notif.push("Reminder", "Test reminder body")
        assert notif.display
        assert "Test reminder body" in str(notif.render())


@pytest.mark.asyncio
async def test_tui_app_mode_toggle_ctrl_m():
    app = JarvisTUIApp(skip_prompt=True, init_services=False)
    async with app.run_test() as pilot:
        app.session.validate_mode = lambda mode: (True, "")
        app.session.setup_providers = lambda mode: None
        # Set skip_mode_prompt so modal does not block
        app.cfg = {"ui": {"skip_mode_prompt": True, "default_mode": "online"}}
        await app.apply_mode("online")
        assert app.selected_mode == "online"

        # Toggle mode via Ctrl+M action
        await app.action_toggle_mode()
        assert app.selected_mode == "local"

        # Toggle back to online
        await app.action_toggle_mode()
        assert app.selected_mode == "online"
        status_text = str(app.status_indicator.render())
        assert "online" in status_text


@pytest.mark.asyncio
async def test_orb_widget_state_and_levels():
    app = JarvisTUIApp(skip_prompt=True, init_services=False)
    async with app.run_test() as pilot:
        orb = app.query_one("#orb", OrbWidget)
        assert orb.state == "idle"

        # Change state to listening
        orb.set_state("listening")
        assert orb.state == "listening"

        # Set mic level
        orb.set_mic_level(0.7)
        assert orb._raw_mic_level == 0.7

        # Return to idle
        orb.set_state("idle")
        assert orb.state == "idle"


@pytest.mark.asyncio
async def test_tui_app_response_display():
    app = JarvisTUIApp(skip_prompt=True, init_services=False)
    async with app.run_test() as pilot:
        response_text = app.query_one("#response-text")
        assert "Jarvis online" in str(response_text.render())

        # Test command handling in turn worker
        app.submit_user_message("/provider")
        await pilot.pause(0.2)
        assert "executed" in str(response_text.render()).lower()


@pytest.mark.asyncio
async def test_tui_app_full_turn_text_and_voice():
    """Verify that user messages produce both visible text and speech output."""
    app = JarvisTUIApp(skip_prompt=True, init_services=False)
    async with app.run_test() as pilot:
        response_text = app.query_one("#response-text")

        # Mock session.get_response_stream to yield known chunks
        async def mock_stream(msg: str):
            yield "Hello Naresh! "
            yield "I am ready."

        app.session.get_response_stream = mock_stream

        # Track speech feeds
        fed_chunks = []
        class MockSpeech:
            enabled = True
            def start_turn(self, ts): pass
            def feed(self, chunk): fed_chunks.append(chunk)
            async def flush(self): pass
            def stop(self): pass

        app.session.speech = MockSpeech()

        # Submit user query
        app.submit_user_message("how are you?")
        await pilot.pause(0.3)

        rendered = str(response_text.render())
        assert "You: " in rendered
        assert "how are you?" in rendered
        assert "Jarvis: " in rendered
        assert "Hello Naresh! I am ready." in rendered
        assert fed_chunks == ["Hello Naresh! ", "I am ready."]
