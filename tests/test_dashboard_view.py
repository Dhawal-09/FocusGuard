"""Tests for the DashboardView contract and formatting helpers
(FOCUSGUARD_PRD.md sections 22-24).

Fully deterministic, pure-function tests - no pygame, no webcam, no YOLO,
no MediaPipe model. This module intentionally never imports pygame.
"""

from __future__ import annotations

from src.core.types import VisionQuality
from src.events.event_manager import Event, EventType, Severity
from src.face.eye_metrics import EyeState
from src.face.head_pose import HeadOrientation
from src.state.state_manager import FocusState
from src.ui.dashboard_view import (
    DashboardView,
    DebugInfo,
    UIAction,
    format_confidence,
    format_duration,
    format_event_timestamp,
    format_event_type,
    format_eye_state,
    format_head_orientation,
    format_presence,
    format_status,
    format_timer,
    format_vision_quality,
    recent_events,
)


def make_event(event_type: EventType, timestamp: float) -> Event:
    return Event(event_type=event_type, timestamp=timestamp, severity=Severity.INFO)


# --- format_duration -----------------------------------------------------------


def test_format_duration_zero() -> None:
    assert format_duration(0.0) == "00:00:00"


def test_format_duration_seconds_only() -> None:
    assert format_duration(45.0) == "00:00:45"


def test_format_duration_minutes_and_seconds() -> None:
    assert format_duration(125.0) == "00:02:05"


def test_format_duration_hours_minutes_seconds() -> None:
    assert format_duration(3661.0) == "01:01:01"


def test_format_duration_truncates_fractional_seconds() -> None:
    assert format_duration(65.9) == "00:01:05"


def test_format_duration_negative_clamps_to_zero() -> None:
    assert format_duration(-5.0) == "00:00:00"


# --- format_presence -------------------------------------------------------------


def test_format_presence_true() -> None:
    assert format_presence(True) == "Detected"


def test_format_presence_false() -> None:
    assert format_presence(False) == "Not Detected"


# --- format_eye_state --------------------------------------------------------------


def test_format_eye_state_open() -> None:
    assert format_eye_state(EyeState.OPEN) == "Open"


def test_format_eye_state_closed() -> None:
    assert format_eye_state(EyeState.CLOSED) == "Closed"


def test_format_eye_state_unknown() -> None:
    assert format_eye_state(EyeState.UNKNOWN) == "Unknown"


# --- format_head_orientation --------------------------------------------------------


def test_format_head_orientation_all_values() -> None:
    expected = {
        HeadOrientation.CENTER: "Center",
        HeadOrientation.LEFT: "Left",
        HeadOrientation.RIGHT: "Right",
        HeadOrientation.UP: "Up",
        HeadOrientation.DOWN: "Down",
        HeadOrientation.UNKNOWN: "Unknown",
    }
    for orientation, label in expected.items():
        assert format_head_orientation(orientation) == label


# --- format_status -----------------------------------------------------------------


def test_format_status_single_word() -> None:
    assert format_status(FocusState.FOCUSED) == "FOCUSED"
    assert format_status(FocusState.IDLE) == "IDLE"
    assert format_status(FocusState.AWAY) == "AWAY"
    assert format_status(FocusState.UNKNOWN) == "UNKNOWN"


def test_format_status_multi_word_replaces_underscore_with_space() -> None:
    assert format_status(FocusState.PHONE_DISTRACTION) == "PHONE DISTRACTION"
    assert format_status(FocusState.DROWSINESS_SIGNAL) == "DROWSINESS SIGNAL"
    assert format_status(FocusState.ATTENTION_DIVERTED) == "ATTENTION DIVERTED"


# --- format_vision_quality ----------------------------------------------------------


def test_format_vision_quality_all_values() -> None:
    assert format_vision_quality(VisionQuality.GOOD) == "Good"
    assert format_vision_quality(VisionQuality.DEGRADED) == "Degraded"
    assert format_vision_quality(VisionQuality.NO_PERSON) == "No Person"


# --- format_confidence --------------------------------------------------------------


def test_format_confidence_rounds_to_two_decimals() -> None:
    assert format_confidence(0.8734) == "0.87"


def test_format_confidence_exact_value() -> None:
    assert format_confidence(0.5) == "0.50"


# --- format_timer --------------------------------------------------------------------


def test_format_timer_none_is_placeholder() -> None:
    assert format_timer(None) == "--"


def test_format_timer_formats_seconds() -> None:
    assert format_timer(0.2) == "0.20s"


def test_format_timer_zero_is_not_confused_with_none() -> None:
    assert format_timer(0.0) == "0.00s"


# --- format_event_type ----------------------------------------------------------------


def test_format_event_type_single_word() -> None:
    event = make_event(EventType.SESSION_STARTED, 0.0)
    assert format_event_type(event) == "Session Started"


def test_format_event_type_multi_word() -> None:
    event = make_event(EventType.PHONE_DETECTED, 0.0)
    assert format_event_type(event) == "Phone Detected"


def test_format_event_type_drowsiness_cleared() -> None:
    event = make_event(EventType.DROWSINESS_CLEARED, 0.0)
    assert format_event_type(event) == "Drowsiness Cleared"


# --- format_event_timestamp: elapsed/session-relative, never wall-clock --------------


def test_format_event_timestamp_no_session_start_is_placeholder() -> None:
    assert format_event_timestamp(100.0, None) == "--:--"


def test_format_event_timestamp_elapsed_since_session_start() -> None:
    assert format_event_timestamp(191.0, 100.0) == "01:31"


def test_format_event_timestamp_at_session_start_is_zero() -> None:
    assert format_event_timestamp(100.0, 100.0) == "00:00"


def test_format_event_timestamp_never_negative() -> None:
    """An event timestamp somehow before session_start (should not
    normally happen) must not render a negative/garbage duration."""
    assert format_event_timestamp(50.0, 100.0) == "00:00"


def test_format_event_timestamp_does_not_resemble_wall_clock_hours() -> None:
    """Regression guard: this must never attempt HH:MM:SS wall-clock
    formatting from a monotonic timestamp - only elapsed MM:SS."""
    result = format_event_timestamp(4000.0, 100.0)  # ~65 minutes elapsed
    assert result.count(":") == 1


# --- recent_events -------------------------------------------------------------------


def test_recent_events_fewer_than_limit_returns_all() -> None:
    events = [make_event(EventType.SESSION_STARTED, float(i)) for i in range(3)]

    result = recent_events(events, limit=8)

    assert result == tuple(events)


def test_recent_events_more_than_limit_returns_latest() -> None:
    events = [make_event(EventType.SESSION_STARTED, float(i)) for i in range(20)]

    result = recent_events(events, limit=8)

    assert len(result) == 8
    assert [e.timestamp for e in result] == list(range(12, 20))


def test_recent_events_default_limit_is_eight() -> None:
    events = [make_event(EventType.SESSION_STARTED, float(i)) for i in range(20)]

    result = recent_events(events)

    assert len(result) == 8


def test_recent_events_preserves_oldest_first_order() -> None:
    events = [make_event(EventType.SESSION_STARTED, float(i)) for i in range(5)]

    result = recent_events(events, limit=3)

    assert [e.timestamp for e in result] == [2.0, 3.0, 4.0]


def test_recent_events_empty_input() -> None:
    assert recent_events([], limit=8) == ()


def test_recent_events_zero_limit_returns_empty() -> None:
    events = [make_event(EventType.SESSION_STARTED, 0.0)]

    assert recent_events(events, limit=0) == ()


def test_recent_events_negative_limit_returns_empty() -> None:
    events = [make_event(EventType.SESSION_STARTED, 0.0)]

    assert recent_events(events, limit=-1) == ()


def test_recent_events_accepts_tuple_input() -> None:
    events = tuple(make_event(EventType.SESSION_STARTED, float(i)) for i in range(3))

    result = recent_events(events, limit=8)

    assert result == events


# --- DashboardView / DebugInfo construction -------------------------------------------


def test_dashboard_view_defaults() -> None:
    view = DashboardView(
        status=FocusState.IDLE,
        person_present=False,
        phone_detected=False,
        eyes_state=EyeState.UNKNOWN,
        head_orientation=HeadOrientation.UNKNOWN,
        session_elapsed_seconds=0.0,
        focus_score=100,
        fps=0.0,
        inference_latency_ms=0.0,
    )

    assert view.recent_events == ()
    assert view.session_start_timestamp is None
    assert view.debug is False
    assert view.debug_info is None


def test_debug_info_defaults() -> None:
    debug_info = DebugInfo()

    assert debug_info.detections == ()
    assert debug_info.landmarks is None
    assert debug_info.eye_metric is None
    assert debug_info.phone_timer_seconds is None


# --- UIAction ---------------------------------------------------------------------------


def test_ui_action_has_exactly_the_prd_controls() -> None:
    names = {action.name for action in UIAction}
    assert names == {"START_PAUSE_RESUME", "EXIT", "TOGGLE_MUTE", "TOGGLE_DEBUG", "RESET"}
