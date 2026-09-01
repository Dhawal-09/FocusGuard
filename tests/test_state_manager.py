"""Tests for StateManager (FOCUSGUARD_PRD.md sections 17-19, 36, 37).

Fully deterministic: synthetic filtered-signal inputs only. No webcam,
GPU, YOLO, or MediaPipe model - and no dependency on any specific
temporal filter's internals, since StateManager only consumes already
decided booleans/VisionQuality.
"""

from __future__ import annotations

from src.core.types import VisionQuality
from src.state.state_manager import FocusState, StateManager

GOOD = VisionQuality.GOOD
DEGRADED = VisionQuality.DEGRADED
NO_PERSON = VisionQuality.NO_PERSON


def evaluate(
    manager: StateManager,
    *,
    is_away: bool = False,
    is_phone_distraction: bool = False,
    is_drowsy: bool = False,
    is_diverted: bool = False,
    vision_quality: VisionQuality = GOOD,
    timestamp: float = 0.0,
):
    return manager.evaluate(
        is_away=is_away,
        is_phone_distraction=is_phone_distraction,
        is_drowsy=is_drowsy,
        is_diverted=is_diverted,
        vision_quality=vision_quality,
        timestamp=timestamp,
    )


# --- Initial state / IDLE ------------------------------------------------------


def test_initial_state_is_idle() -> None:
    manager = StateManager()

    assert manager.state == FocusState.IDLE


def test_evaluate_while_idle_is_a_no_op() -> None:
    manager = StateManager()

    result = evaluate(manager, vision_quality=GOOD)

    assert result.state == FocusState.IDLE
    assert result.changed is False


# --- PRD section 19: IDLE -> UNKNOWN, then UNKNOWN -> appropriate state -----


def test_start_session_transitions_idle_to_unknown() -> None:
    manager = StateManager()

    result = manager.start_session(timestamp=1.0)

    assert result.previous_state == FocusState.IDLE
    assert result.state == FocusState.UNKNOWN
    assert result.changed is True
    assert manager.state == FocusState.UNKNOWN


def test_start_session_is_idempotent_while_already_active() -> None:
    manager = StateManager()
    manager.start_session(timestamp=1.0)
    evaluate(manager, vision_quality=GOOD, timestamp=1.5)  # now FOCUSED

    result = manager.start_session(timestamp=2.0)

    assert result.state == FocusState.FOCUSED
    assert result.changed is False


def test_first_evaluate_after_start_with_good_vision_reaches_focused() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)

    result = evaluate(manager, vision_quality=GOOD, timestamp=1.0)

    assert result.previous_state == FocusState.UNKNOWN
    assert result.state == FocusState.FOCUSED
    assert result.changed is True


# --- PRD section 19 minimum transitions ----------------------------------------


def test_idle_to_focused_via_start_then_evaluate() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)

    result = evaluate(manager, vision_quality=GOOD, timestamp=1.0)

    assert result.state == FocusState.FOCUSED


def test_focused_to_phone_distraction() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)
    evaluate(manager, vision_quality=GOOD, timestamp=1.0)

    result = evaluate(manager, is_phone_distraction=True, vision_quality=GOOD, timestamp=2.0)

    assert result.previous_state == FocusState.FOCUSED
    assert result.state == FocusState.PHONE_DISTRACTION
    assert result.changed is True


def test_phone_distraction_to_focused() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)
    evaluate(manager, is_phone_distraction=True, vision_quality=GOOD, timestamp=1.0)

    result = evaluate(manager, is_phone_distraction=False, vision_quality=GOOD, timestamp=2.0)

    assert result.previous_state == FocusState.PHONE_DISTRACTION
    assert result.state == FocusState.FOCUSED


def test_focused_to_drowsiness_signal() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)
    evaluate(manager, vision_quality=GOOD, timestamp=1.0)

    result = evaluate(manager, is_drowsy=True, vision_quality=GOOD, timestamp=2.0)

    assert result.state == FocusState.DROWSINESS_SIGNAL


def test_drowsiness_signal_to_focused() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)
    evaluate(manager, is_drowsy=True, vision_quality=GOOD, timestamp=1.0)

    result = evaluate(manager, is_drowsy=False, vision_quality=GOOD, timestamp=2.0)

    assert result.state == FocusState.FOCUSED


def test_focused_to_attention_diverted() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)
    evaluate(manager, vision_quality=GOOD, timestamp=1.0)

    result = evaluate(manager, is_diverted=True, vision_quality=GOOD, timestamp=2.0)

    assert result.state == FocusState.ATTENTION_DIVERTED


def test_attention_diverted_to_focused() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)
    evaluate(manager, is_diverted=True, vision_quality=GOOD, timestamp=1.0)

    result = evaluate(manager, is_diverted=False, vision_quality=GOOD, timestamp=2.0)

    assert result.state == FocusState.FOCUSED


def test_focused_to_away() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)
    evaluate(manager, vision_quality=GOOD, timestamp=1.0)

    result = evaluate(manager, is_away=True, vision_quality=NO_PERSON, timestamp=2.0)

    assert result.state == FocusState.AWAY


def test_away_to_focused() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)
    evaluate(manager, is_away=True, vision_quality=NO_PERSON, timestamp=1.0)

    result = evaluate(manager, is_away=False, vision_quality=GOOD, timestamp=2.0)

    assert result.state == FocusState.FOCUSED


def test_any_active_state_to_unknown() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)
    evaluate(manager, is_phone_distraction=True, vision_quality=GOOD, timestamp=1.0)

    # Phone clears but face tracking is simultaneously lost.
    result = evaluate(manager, is_phone_distraction=False, vision_quality=DEGRADED, timestamp=2.0)

    assert result.state == FocusState.UNKNOWN


def test_unknown_to_appropriate_valid_state() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)  # lands on UNKNOWN
    assert manager.state == FocusState.UNKNOWN

    result = evaluate(manager, vision_quality=GOOD, timestamp=1.0)

    assert result.state == FocusState.FOCUSED


# --- PRD section 18 / 36: priority under simultaneous signals ----------------


def test_priority_phone_and_drowsiness_yields_phone_distraction() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)

    result = evaluate(manager, is_phone_distraction=True, is_drowsy=True, vision_quality=GOOD, timestamp=1.0)

    assert result.state == FocusState.PHONE_DISTRACTION


def test_priority_away_and_phone_yields_away() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)

    result = evaluate(
        manager, is_away=True, is_phone_distraction=True, vision_quality=NO_PERSON, timestamp=1.0
    )

    assert result.state == FocusState.AWAY


def test_priority_away_beats_everything() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)

    result = evaluate(
        manager,
        is_away=True,
        is_phone_distraction=True,
        is_drowsy=True,
        is_diverted=True,
        vision_quality=NO_PERSON,
        timestamp=1.0,
    )

    assert result.state == FocusState.AWAY


def test_priority_phone_beats_drowsiness_and_attention() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)

    result = evaluate(
        manager,
        is_phone_distraction=True,
        is_drowsy=True,
        is_diverted=True,
        vision_quality=GOOD,
        timestamp=1.0,
    )

    assert result.state == FocusState.PHONE_DISTRACTION


def test_priority_drowsiness_beats_attention() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)

    result = evaluate(manager, is_drowsy=True, is_diverted=True, vision_quality=GOOD, timestamp=1.0)

    assert result.state == FocusState.DROWSINESS_SIGNAL


def test_priority_attention_beats_degraded_unknown() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)

    result = evaluate(manager, is_diverted=True, vision_quality=DEGRADED, timestamp=1.0)

    assert result.state == FocusState.ATTENTION_DIVERTED


# --- Approved Phase 7 decision: UNKNOWN only when person present but --------
# --- face-derived perception unavailable; missing person means AWAY --------


def test_degraded_vision_with_person_present_yields_unknown() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)

    result = evaluate(manager, vision_quality=DEGRADED, timestamp=1.0)

    assert result.state == FocusState.UNKNOWN


def test_no_person_before_away_confirms_holds_previous_state_not_unknown() -> None:
    """The away-duration grace period: person not currently visible, but
    PersonAwayFilter has not confirmed AWAY yet - PRD section 14 says no
    event/no visible change should happen yet, so the state machine must
    hold the last confirmed state rather than flapping into UNKNOWN."""
    manager = StateManager()
    manager.start_session(timestamp=0.0)
    evaluate(manager, vision_quality=GOOD, timestamp=1.0)  # FOCUSED
    assert manager.state == FocusState.FOCUSED

    result = evaluate(manager, is_away=False, vision_quality=NO_PERSON, timestamp=1.5)

    assert result.state == FocusState.FOCUSED
    assert result.changed is False


def test_no_person_from_the_very_first_evaluate_falls_back_to_unknown() -> None:
    """If the very first evaluate() after start_session() already has no
    person, there is no 'previous confirmed state' to hold beyond the
    UNKNOWN start_session() already landed on."""
    manager = StateManager()
    manager.start_session(timestamp=0.0)  # UNKNOWN

    result = evaluate(manager, is_away=False, vision_quality=NO_PERSON, timestamp=0.5)

    assert result.state == FocusState.UNKNOWN


def test_no_person_holds_previous_distraction_state_too() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)
    evaluate(manager, is_phone_distraction=True, vision_quality=GOOD, timestamp=1.0)
    assert manager.state == FocusState.PHONE_DISTRACTION

    result = evaluate(manager, is_phone_distraction=False, is_away=False, vision_quality=NO_PERSON, timestamp=1.5)

    assert result.state == FocusState.PHONE_DISTRACTION


# --- PRD section 19 / 37: no frame-by-frame flapping for an unchanged state -


def test_no_repeated_change_flag_for_unchanged_state() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)
    evaluate(manager, vision_quality=GOOD, timestamp=1.0)  # FOCUSED, changed=True

    for t in (2.0, 3.0, 4.0):
        result = evaluate(manager, vision_quality=GOOD, timestamp=t)
        assert result.state == FocusState.FOCUSED
        assert result.changed is False


# --- end_session ---------------------------------------------------------------


def test_end_session_returns_to_idle() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)
    evaluate(manager, vision_quality=GOOD, timestamp=1.0)

    result = manager.end_session(timestamp=2.0)

    assert result.previous_state == FocusState.FOCUSED
    assert result.state == FocusState.IDLE
    assert result.changed is True
    assert manager.state == FocusState.IDLE


def test_evaluate_after_end_session_is_a_no_op_again() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)
    evaluate(manager, vision_quality=GOOD, timestamp=1.0)
    manager.end_session(timestamp=2.0)

    result = evaluate(manager, is_phone_distraction=True, vision_quality=GOOD, timestamp=3.0)

    assert result.state == FocusState.IDLE
    assert result.changed is False


def test_new_session_after_end_session_starts_clean() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)
    evaluate(manager, is_phone_distraction=True, vision_quality=GOOD, timestamp=1.0)
    manager.end_session(timestamp=2.0)

    manager.start_session(timestamp=3.0)
    result = evaluate(manager, vision_quality=GOOD, timestamp=4.0)

    assert result.state == FocusState.FOCUSED


# --- Timestamp propagation ------------------------------------------------------


def test_timestamp_propagates_into_result() -> None:
    manager = StateManager()
    manager.start_session(timestamp=0.0)

    result = evaluate(manager, vision_quality=GOOD, timestamp=42.5)

    assert result.timestamp == 42.5
