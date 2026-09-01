"""Tests for EventManager (FOCUSGUARD_PRD.md sections 20, 28, 36).

Fully deterministic: direct calls to EventManager's public methods with
synthetic timestamps. No webcam, GPU, YOLO, or MediaPipe model.
"""

from __future__ import annotations

import pytest

from src.core.config_manager import PhoneConfig, SessionConfig
from src.events.event_manager import EventManager, EventType, Severity

COOLDOWN = 10.0


def make_session_config(max_event_log_entries: int = 100) -> SessionConfig:
    return SessionConfig(max_event_log_entries=max_event_log_entries)


def make_phone_config(warning_cooldown_seconds: float = COOLDOWN) -> PhoneConfig:
    return PhoneConfig(
        confirm_duration_seconds=0.35,
        clear_duration_seconds=0.60,
        warning_cooldown_seconds=warning_cooldown_seconds,
    )


def make_manager(max_event_log_entries: int = 100, warning_cooldown_seconds: float = COOLDOWN) -> EventManager:
    return EventManager(
        make_session_config(max_event_log_entries),
        make_phone_config(warning_cooldown_seconds),
    )


# --- Construction validation ---------------------------------------------------


def test_zero_max_entries_raises_value_error() -> None:
    with pytest.raises(ValueError):
        make_manager(max_event_log_entries=0)


# --- Empty log initially ---------------------------------------------------------


def test_log_starts_empty() -> None:
    manager = make_manager()

    assert manager.events == []


# --- Session lifecycle events ----------------------------------------------------


def test_session_started_emits_info_event() -> None:
    manager = make_manager()

    event = manager.session_started(1.0)

    assert event.event_type == EventType.SESSION_STARTED
    assert event.severity == Severity.INFO
    assert event.timestamp == 1.0
    assert manager.events == [event]


def test_session_ended_emits_info_event() -> None:
    manager = make_manager()

    event = manager.session_ended(1.0)

    assert event.event_type == EventType.SESSION_ENDED
    assert event.severity == Severity.INFO


# --- Signal-level events: phone --------------------------------------------------


def test_phone_confirmed_emits_warning_event() -> None:
    manager = make_manager()

    event = manager.phone_confirmed(1.0)

    assert event is not None
    assert event.event_type == EventType.PHONE_DETECTED
    assert event.severity == Severity.WARNING


def test_phone_cleared_emits_info_event() -> None:
    manager = make_manager()

    event = manager.phone_cleared(1.0)

    assert event.event_type == EventType.PHONE_CLEARED
    assert event.severity == Severity.INFO


# --- Signal-level events: drowsiness ----------------------------------------------


def test_drowsiness_confirmed_emits_warning_event() -> None:
    manager = make_manager()

    event = manager.drowsiness_confirmed(1.0)

    assert event.event_type == EventType.DROWSINESS_SIGNAL
    assert event.severity == Severity.WARNING


def test_drowsiness_cleared_emits_info_event() -> None:
    manager = make_manager()

    event = manager.drowsiness_cleared(1.0)

    assert event.event_type == EventType.DROWSINESS_CLEARED
    assert event.severity == Severity.INFO


# --- Signal-level events: attention -----------------------------------------------


def test_attention_diverted_emits_warning_event() -> None:
    manager = make_manager()

    event = manager.attention_diverted(1.0)

    assert event.event_type == EventType.ATTENTION_DIVERTED
    assert event.severity == Severity.WARNING


def test_attention_restored_emits_info_event() -> None:
    manager = make_manager()

    event = manager.attention_restored(1.0)

    assert event.event_type == EventType.ATTENTION_RESTORED
    assert event.severity == Severity.INFO


# --- Signal-level events: person away ---------------------------------------------


def test_person_left_emits_warning_event() -> None:
    manager = make_manager()

    event = manager.person_left(1.0)

    assert event.event_type == EventType.PERSON_LEFT
    assert event.severity == Severity.WARNING


def test_person_returned_emits_info_event() -> None:
    manager = make_manager()

    event = manager.person_returned(1.0)

    assert event.event_type == EventType.PERSON_RETURNED
    assert event.severity == Severity.INFO


# --- State-level event: focus restored --------------------------------------------


def test_focus_restored_emits_info_event() -> None:
    manager = make_manager()

    event = manager.focus_restored(1.0)

    assert event.event_type == EventType.FOCUS_RESTORED
    assert event.severity == Severity.INFO


# --- Error events ------------------------------------------------------------------


def test_camera_error_emits_error_event_with_message_metadata() -> None:
    manager = make_manager()

    event = manager.camera_error(1.0, "camera disconnected")

    assert event.event_type == EventType.CAMERA_ERROR
    assert event.severity == Severity.ERROR
    assert event.metadata == {"message": "camera disconnected"}


def test_model_error_emits_error_event() -> None:
    manager = make_manager()

    event = manager.model_error(1.0, "failed to load model")

    assert event.event_type == EventType.MODEL_ERROR
    assert event.severity == Severity.ERROR


def test_vision_error_emits_error_event() -> None:
    manager = make_manager()

    event = manager.vision_error(1.0, "inference failed")

    assert event.event_type == EventType.VISION_ERROR
    assert event.severity == Severity.ERROR


# --- Bounded log (PRD section 28) --------------------------------------------------


def test_log_respects_max_entries_and_keeps_most_recent() -> None:
    manager = make_manager(max_event_log_entries=3)

    for i in range(5):
        manager.session_started(float(i))

    assert len(manager.events) == 3
    assert [e.timestamp for e in manager.events] == [2.0, 3.0, 4.0]


def test_log_at_exactly_max_entries_keeps_all() -> None:
    manager = make_manager(max_event_log_entries=3)

    for i in range(3):
        manager.session_started(float(i))

    assert len(manager.events) == 3
    assert [e.timestamp for e in manager.events] == [0.0, 1.0, 2.0]


def test_events_property_returns_a_copy_not_a_live_reference() -> None:
    manager = make_manager()
    manager.session_started(1.0)

    snapshot = manager.events
    manager.session_started(2.0)

    assert len(snapshot) == 1
    assert len(manager.events) == 2


def test_log_is_timestamp_ordered_oldest_first() -> None:
    manager = make_manager()
    manager.session_started(1.0)
    manager.phone_confirmed(2.0)
    manager.attention_diverted(3.0)

    timestamps = [e.timestamp for e in manager.events]

    assert timestamps == [1.0, 2.0, 3.0]


# --- Cooldown (PRD section 21, 36): phone-only ---------------------------------------


def test_second_phone_confirmed_within_cooldown_is_suppressed() -> None:
    manager = make_manager()
    first = manager.phone_confirmed(1.0)

    second = manager.phone_confirmed(1.0 + COOLDOWN - 1.0)

    assert first is not None
    assert second is None
    assert len(manager.events) == 1


def test_phone_confirmed_exactly_at_cooldown_boundary_fires() -> None:
    manager = make_manager()
    manager.phone_confirmed(0.0)

    event = manager.phone_confirmed(COOLDOWN)

    assert event is not None
    assert len(manager.events) == 2


def test_phone_confirmed_after_cooldown_elapses_fires() -> None:
    manager = make_manager()
    manager.phone_confirmed(0.0)

    event = manager.phone_confirmed(COOLDOWN + 1.0)

    assert event is not None
    assert len(manager.events) == 2


def test_suppressed_phone_confirmed_does_not_consume_a_log_slot() -> None:
    manager = make_manager(max_event_log_entries=100)
    manager.phone_confirmed(0.0)

    manager.phone_confirmed(1.0)  # suppressed by cooldown
    manager.phone_confirmed(2.0)  # suppressed by cooldown

    assert len(manager.events) == 1


def test_phone_cleared_is_never_subject_to_cooldown() -> None:
    """Only PHONE_DETECTED has a PRD-defined cooldown value
    (warning_cooldown_seconds); PHONE_CLEARED always logs."""
    manager = make_manager()

    first = manager.phone_cleared(0.0)
    second = manager.phone_cleared(0.1)

    assert first is not None
    assert second is not None
    assert len(manager.events) == 2


def test_other_event_types_are_not_subject_to_phone_cooldown() -> None:
    manager = make_manager()

    manager.drowsiness_confirmed(0.0)
    manager.drowsiness_confirmed(0.1)
    manager.attention_diverted(0.2)

    assert len(manager.events) == 3


def test_zero_cooldown_never_suppresses() -> None:
    manager = make_manager(warning_cooldown_seconds=0.0)

    first = manager.phone_confirmed(0.0)
    second = manager.phone_confirmed(0.0)

    assert first is not None
    assert second is not None
