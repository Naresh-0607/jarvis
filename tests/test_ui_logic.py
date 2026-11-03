"""Unit tests for TUI pure logic: state machine, level smoothing, ASCII orb rendering, ripples, connectivity."""

import pytest
from jarvis.tui import (
    UIStateMachine,
    LevelSmoother,
    RippleManager,
    determine_connectivity,
    generate_orb_grid,
    render_ascii_waveform,
    probe_internet,
)


def test_state_machine_initial_state():
    sm = UIStateMachine()
    assert sm.state == "idle"
    assert sm.is_idle


def test_state_machine_valid_transitions():
    sm = UIStateMachine()
    assert sm.transition("listening")
    assert sm.state == "listening"
    assert sm.is_listening

    assert sm.transition("thinking")
    assert sm.state == "thinking"
    assert sm.is_thinking

    assert sm.transition("speaking")
    assert sm.state == "speaking"
    assert sm.is_speaking

    assert sm.transition("idle")
    assert sm.state == "idle"
    assert sm.is_idle


def test_state_machine_barge_in_transition():
    sm = UIStateMachine()
    sm.transition("speaking")
    # Barge in directly transitions from speaking to listening
    assert sm.transition("listening")
    assert sm.state == "listening"


def test_state_machine_invalid_transition():
    sm = UIStateMachine()
    assert not sm.transition("unknown_state")
    assert sm.state == "idle"


def test_level_smoother_attack_and_decay():
    smoother = LevelSmoother(attack=0.5, decay=0.2)
    assert smoother.current == 0.0

    # Rising level (attack phase)
    val1 = smoother.update(1.0)
    assert val1 == 0.5

    val2 = smoother.update(1.0)
    assert val2 == 0.75

    # Falling level (decay phase)
    val3 = smoother.update(0.0)
    assert val3 == 0.6

    # Reset
    smoother.reset()
    assert smoother.current == 0.0


def test_determine_connectivity_online():
    status, label = determine_connectivity(is_online_reachable=True, local_available=True)
    assert status == "online"
    assert label == "online"


def test_determine_connectivity_local():
    status, label = determine_connectivity(is_online_reachable=False, local_available=True)
    assert status == "local"
    assert label == "local"


def test_determine_connectivity_error_when_none_available():
    status, label = determine_connectivity(is_online_reachable=False, local_available=False)
    assert status == "error"
    assert label == "error"


def test_determine_connectivity_explicit_error():
    status, label = determine_connectivity(
        is_online_reachable=True, local_available=True, has_error=True
    )
    assert status == "error"
    assert label == "error"


def test_generate_orb_grid_pure_ascii_guarantee():
    """Verify that every character in the rendered orb grid is strictly ASCII 32-126."""
    lines = generate_orb_grid(
        width=50,
        height=19,
        radius=5.0,
        level=0.5,
        gain=5.0,
        state="speaking",
        phase=1.2,
        ripples=[{"radius": 7.0, "age": 0.2}],
    )
    assert len(lines) == 19
    for line in lines:
        assert len(line) == 50
        for ch in line:
            assert 32 <= ord(ch) <= 126, (
                f"Character {ch!r} (ord {ord(ch)}) is not in ASCII range 32-126!"
            )


def test_generate_orb_grid_core_density():
    """Verify that the center of the orb has higher density characters than the perimeter."""
    lines = generate_orb_grid(width=41, height=15, radius=5.0, state="idle", phase=0.0)
    cx = 41 // 2
    cy = 15 // 2
    center_char = lines[cy][cx]
    perimeter_char = lines[cy][cx + 8]  # near edge
    outside_char = lines[cy][0]  # far outside

    ramp = " .:-=+*#%@"
    assert ramp.index(center_char) > ramp.index(perimeter_char)
    assert outside_char == " "


def test_generate_orb_grid_dynamic_growth_with_level():
    """Verify that higher audio amplitude results in a larger orb with more non-space characters."""
    quiet = generate_orb_grid(width=40, height=16, radius=4.0, level=0.0, state="listening")
    loud = generate_orb_grid(width=40, height=16, radius=4.0, level=1.0, state="listening")

    quiet_count = sum(ch != " " for line in quiet for ch in line)
    loud_count = sum(ch != " " for line in loud for ch in line)
    assert loud_count > quiet_count


def test_generate_orb_grid_empty():
    assert generate_orb_grid(width=0, height=0, radius=5.0) == []
    assert generate_orb_grid(width=-5, height=10, radius=5.0) == []


def test_ripple_manager_trigger_and_throttle():
    rm = RippleManager(max_ripples=3, throttle_s=0.2)
    assert rm.count == 0

    # First trigger should succeed
    assert rm.trigger(base_radius=5.0, now=1.0)
    assert rm.count == 1

    # Immediate second trigger should be throttled
    assert not rm.trigger(base_radius=5.0, now=1.1)
    assert rm.count == 1

    # Trigger after throttle interval should succeed
    assert rm.trigger(base_radius=5.0, now=1.3)
    assert rm.count == 2


def test_ripple_manager_update_and_expiration():
    rm = RippleManager(max_ripples=3, throttle_s=0.1)
    rm.trigger(base_radius=4.0, now=1.0)
    assert rm.count == 1

    # Advancing age
    rm.update(0.3)
    assert rm.count == 1
    assert rm.ripples[0]["age"] == pytest.approx(0.3)

    # Exceeding 1.0 second expires the ripple
    rm.update(0.8)
    assert rm.count == 0


def test_render_ascii_waveform_pure_ascii_guarantee():
    """Verify that every single character rendered in the waveform is within ASCII 32-126."""
    buffer = [0.0, 0.1, 0.3, 0.7, 1.0, 0.8, 0.4, 0.2, 0.0]
    lines = render_ascii_waveform(buffer, width=9, height=9)
    assert len(lines) == 9
    for line in lines:
        assert len(line) == 9
        for ch in line:
            assert 32 <= ord(ch) <= 126, (
                f"Character {ch!r} (ord {ord(ch)}) is not in ASCII range 32-126!"
            )


def test_render_ascii_waveform_center_line():
    """Verify that the center line is a continuous row of center_char."""
    buffer = [0.5] * 20
    lines = render_ascii_waveform(buffer, width=20, height=7, center_char="-")
    cy = 7 // 2
    assert lines[cy] == "-" * 20


def test_render_ascii_waveform_mirrored():
    """Verify that rows above and below the center line are mirror images."""
    buffer = [0.2, 0.8, 0.4, 0.9, 0.1]
    lines = render_ascii_waveform(buffer, width=5, height=9)
    cy = 9 // 2
    for dy in range(1, cy + 1):
        assert lines[cy - dy] == lines[cy + dy], f"Row {cy - dy} does not mirror row {cy + dy}"


def test_render_ascii_waveform_empty():
    assert render_ascii_waveform([], width=0, height=0) == []


def test_main_cli_dispatch():
    import sys
    from unittest.mock import patch

    with patch("jarvis.__main__.chat_main") as mock_chat:
        from jarvis.__main__ import main

        sys.argv = ["jarvis", "--cli"]
        main()
        assert mock_chat.called


def test_main_tui_default_dispatch():
    import sys
    from unittest.mock import patch

    with patch("jarvis.tui.main") as mock_tui:
        from jarvis.__main__ import main

        sys.argv = ["jarvis"]
        main()
        assert mock_tui.called
