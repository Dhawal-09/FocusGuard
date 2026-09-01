"""Tests for SessionManager (FOCUSGUARD_PRD.md sections 25-27, 33, 37).

Fully deterministic: synthetic StateTransition/Event/timestamp sequences
only. No webcam, GPU, YOLO, MediaPipe, Pygame display, or audio device -
this is a pure data/logic phase, the same category as StateManager/
EventManager. JSON persistence tests always write to a pytest tmp_path,
never the repository's real logs/ directory.
"""

from __future__ import annotations

import json
import math

import pytest

from src.core.config_manager import ScoreConfig
from src.events.event_manager import Event, EventType, Severity
from src.session.session_manager import SessionError, SessionManager, SessionSummary
from src.state.state_manager import FocusState, StateTransition

STARTING_SCORE = 100
PHONE_PENALTY = 10
DROWSY_PENALTY = 5
ATTENTION_PENALTY = 3
AWAY_PENALTY = 5


def ts(*parts: float) -> float:
    """Precise timestamp summation - avoids binary float drift at exact
    boundaries (established convention from every prior phase's tests)."""
    return round(math.fsum(parts), 9)


def make_score_config(**overrides) -> ScoreConfig:
    defaults = dict(
        starting_score=STARTING_SCORE,
        phone_event_penalty=PHONE_PENALTY,
        drowsiness_event_penalty=DROWSY_PENALTY,
        attention_event_penalty=ATTENTION_PENALTY,
        away_event_penalty=AWAY_PENALTY,
    )
    defaults.update(overrides)
    return ScoreConfig(**defaults)


def transition(state: FocusState, timestamp: float, previous: FocusState | None = None) -> StateTransition:
    prev = previous if previous is not None else state
    return StateTransition(previous_state=prev, state=state, changed=prev != state, timestamp=timestamp)


def make_event(event_type: EventType, timestamp: float, severity: Severity = Severity.INFO) -> Event:
    return Event(event_type=event_type, timestamp=timestamp, severity=severity)


def make_manager(**score_overrides) -> SessionManager:
    return SessionManager(make_score_config(**score_overrides))


# --- Lifecycle: start ------------------------------------------------------------------


def test_not_active_by_default() -> None:
    manager = make_manager()
    assert manager.is_active is False
    assert manager.is_paused is False


def test_start_session_activates() -> None:
    manager = make_manager()

    manager.start_session(0.0)

    assert manager.is_active is True
    assert manager.is_paused is False


def test_start_session_initializes_score_to_starting_score() -> None:
    manager = make_manager(starting_score=77)
    manager.start_session(0.0)

    assert manager.focus_score == 77


def test_start_session_is_idempotent() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_event(make_event(EventType.PHONE_DETECTED, 1.0))
    assert manager.focus_score == STARTING_SCORE - PHONE_PENALTY

    manager.start_session(5.0)  # must NOT reset already-accumulated state

    assert manager.focus_score == STARTING_SCORE - PHONE_PENALTY
    assert manager.phone_distraction_count == 1


# --- Lifecycle: pause/resume ------------------------------------------------------------


def test_pause_before_start_is_a_noop() -> None:
    manager = make_manager()
    manager.pause_session(0.0)
    assert manager.is_paused is False


def test_pause_session_sets_paused() -> None:
    manager = make_manager()
    manager.start_session(0.0)

    manager.pause_session(1.0)

    assert manager.is_paused is True


def test_double_pause_is_a_noop() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.FOCUSED, 1.0))
    manager.pause_session(2.0)

    manager.pause_session(3.0)  # must not double-count or raise on stale timestamp semantics

    assert manager.is_paused is True
    # [0,1) credited to IDLE (pre-seed), [1,2) credited to FOCUSED = 1.0; the
    # second pause() is a no-op and must not add [2,3) again.
    assert manager.focused_duration_seconds == pytest.approx(1.0)


def test_resume_before_start_is_a_noop() -> None:
    manager = make_manager()
    manager.resume_session(0.0)
    assert manager.is_paused is False


def test_resume_without_pause_is_a_noop() -> None:
    manager = make_manager()
    manager.start_session(0.0)

    manager.resume_session(1.0)

    assert manager.is_paused is False


def test_resume_clears_paused() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.pause_session(1.0)

    manager.resume_session(2.0)

    assert manager.is_paused is False


# --- Lifecycle: end ----------------------------------------------------------------------


def test_end_session_without_start_raises_session_error() -> None:
    manager = make_manager()

    with pytest.raises(SessionError):
        manager.end_session(0.0)


def test_end_session_returns_summary_and_deactivates() -> None:
    manager = make_manager()
    manager.start_session(0.0)

    summary = manager.end_session(10.0)

    assert isinstance(summary, SessionSummary)
    assert manager.is_active is False


def test_end_session_total_duration() -> None:
    manager = make_manager()
    manager.start_session(0.0)

    summary = manager.end_session(125.0)

    assert summary.total_duration_seconds == pytest.approx(125.0)


def test_reset_returns_to_blank_slate() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_event(make_event(EventType.PHONE_DETECTED, 1.0))
    manager.pause_session(2.0)

    manager.reset()

    assert manager.is_active is False
    assert manager.is_paused is False
    assert manager.focus_score == STARTING_SCORE
    assert manager.phone_distraction_count == 0


# --- record_transition / record_event: defensive no-op while inactive/paused --------


def test_record_transition_before_start_is_a_noop() -> None:
    manager = make_manager()

    manager.record_transition(transition(FocusState.FOCUSED, 0.0))  # must not raise

    assert manager.focused_duration_seconds == 0.0


def test_record_event_before_start_is_a_noop() -> None:
    manager = make_manager()

    manager.record_event(make_event(EventType.PHONE_DETECTED, 0.0))  # must not raise

    assert manager.focus_score == STARTING_SCORE
    assert manager.phone_distraction_count == 0


def test_record_transition_while_paused_is_a_noop() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.FOCUSED, 1.0))
    manager.pause_session(2.0)

    manager.record_transition(transition(FocusState.FOCUSED, 100.0))  # ignored while paused

    # [0,1) credited to IDLE (pre-seed), [1,2) credited to FOCUSED = 1.0
    assert manager.focused_duration_seconds == pytest.approx(1.0)


def test_record_event_while_paused_is_a_noop() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.pause_session(1.0)

    manager.record_event(make_event(EventType.PHONE_DETECTED, 50.0))

    assert manager.focus_score == STARTING_SCORE
    assert manager.phone_distraction_count == 0


def test_record_event_while_paused_is_not_appended_to_summary_events() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.pause_session(1.0)
    manager.record_event(make_event(EventType.PHONE_DETECTED, 50.0))
    manager.resume_session(60.0)

    summary = manager.end_session(61.0)

    assert summary.events == ()


# --- Duration accounting: FOCUSED ---------------------------------------------------------


def test_focused_duration_accumulates_across_multiple_transitions() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.FOCUSED, 0.0))  # seed: FOCUSED as of t=0

    manager.record_transition(transition(FocusState.FOCUSED, 1.0))  # +1
    manager.record_transition(transition(FocusState.FOCUSED, 2.0))  # changed=False, still counts: +1
    manager.record_transition(transition(FocusState.FOCUSED, 3.0))  # +1

    assert manager.focused_duration_seconds == pytest.approx(3.0)


def test_focused_duration_stops_accumulating_after_leaving_focused() -> None:
    """A transition INTO a new state still credits the interval leading up
    to it to the OLD state - the new state only starts accruing duration
    from its own transition's timestamp onward."""
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.FOCUSED, 0.0))  # seed
    manager.record_transition(transition(FocusState.FOCUSED, 5.0))  # +5 -> focused=5

    # [5,10) is still credited to FOCUSED (state only changes AT t=10)
    manager.record_transition(transition(FocusState.PHONE_DISTRACTION, 10.0, previous=FocusState.FOCUSED))
    # [10,20) credited to PHONE_DISTRACTION
    manager.record_transition(transition(FocusState.PHONE_DISTRACTION, 20.0))

    assert manager.focused_duration_seconds == pytest.approx(10.0)
    assert manager.phone_distraction_duration_seconds == pytest.approx(10.0)


def test_non_focused_non_phone_states_accumulate_nothing() -> None:
    manager = make_manager()
    manager.start_session(0.0)

    for state in (FocusState.DROWSINESS_SIGNAL, FocusState.ATTENTION_DIVERTED, FocusState.AWAY, FocusState.UNKNOWN):
        manager.record_transition(transition(state, 10.0))

    assert manager.focused_duration_seconds == 0.0
    assert manager.phone_distraction_duration_seconds == 0.0


# --- Duration accounting: PHONE_DISTRACTION ------------------------------------------------


def test_phone_distraction_duration_accumulates() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.PHONE_DISTRACTION, 1.0))

    manager.record_transition(transition(FocusState.PHONE_DISTRACTION, 8.0))

    assert manager.phone_distraction_duration_seconds == pytest.approx(7.0)


# --- Longest focus streak ----------------------------------------------------------------


def test_longest_streak_single_continuous_period() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.FOCUSED, 0.0))  # seed

    manager.record_transition(transition(FocusState.FOCUSED, 10.0))

    assert manager.longest_focus_streak_seconds == pytest.approx(10.0)


def test_longest_streak_keeps_the_longer_of_two_periods() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.FOCUSED, 0.0))  # seed: FOCUSED as of t=0

    # [0,5) FOCUSED -> streak 1 = 5s
    manager.record_transition(transition(FocusState.PHONE_DISTRACTION, 5.0, previous=FocusState.FOCUSED))
    # [5,6) PHONE_DISTRACTION -> phone += 1, streak resets
    manager.record_transition(transition(FocusState.FOCUSED, 6.0, previous=FocusState.PHONE_DISTRACTION))
    # [6,19) FOCUSED -> streak 2 = 13s
    manager.record_transition(transition(FocusState.FOCUSED, 19.0))

    assert manager.longest_focus_streak_seconds == pytest.approx(13.0)
    assert manager.focused_duration_seconds == pytest.approx(18.0)  # 5 + 13, cumulative not max
    assert manager.phone_distraction_duration_seconds == pytest.approx(1.0)


def test_streak_resets_immediately_on_leaving_focused() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.FOCUSED, 0.0))  # seed
    # [0,5) FOCUSED -> streak = 5
    manager.record_transition(transition(FocusState.PHONE_DISTRACTION, 5.0, previous=FocusState.FOCUSED))

    # [5,100) PHONE_DISTRACTION -> streak stays reset at 0, longest stays 5
    manager.record_transition(transition(FocusState.PHONE_DISTRACTION, 100.0))

    assert manager.longest_focus_streak_seconds == pytest.approx(5.0)  # not extended by the distraction interval


def test_first_short_focus_streak_then_longer_one_reports_the_longer() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.FOCUSED, 0.0))  # seed
    # [0,2) FOCUSED -> streak = 2
    manager.record_transition(transition(FocusState.DROWSINESS_SIGNAL, 2.0, previous=FocusState.FOCUSED))
    # [2,3) DROWSINESS_SIGNAL -> streak resets
    manager.record_transition(transition(FocusState.FOCUSED, 3.0, previous=FocusState.DROWSINESS_SIGNAL))
    # [3,29) FOCUSED -> streak = 26
    manager.record_transition(transition(FocusState.FOCUSED, 29.0))

    assert manager.longest_focus_streak_seconds == pytest.approx(26.0)


# --- Pause/resume excludes the paused gap from duration accounting --------------------------


def test_pause_resume_excludes_gap_from_focused_duration() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.FOCUSED, 0.0))  # seed
    manager.record_transition(transition(FocusState.FOCUSED, 5.0))  # +5 -> focused=5
    manager.pause_session(5.0)

    manager.resume_session(1000.0)  # a huge real-world gap while paused
    manager.record_transition(transition(FocusState.FOCUSED, 1008.0))  # another 8s focused

    assert manager.focused_duration_seconds == pytest.approx(13.0)  # 5 + 8, NOT 1013


def test_pause_credits_the_pre_pause_interval_before_pausing() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.FOCUSED, 0.0))  # seed
    manager.record_transition(transition(FocusState.FOCUSED, 3.0))  # +3 -> focused=3

    manager.pause_session(7.0)  # 3 -> 7 still credited to FOCUSED: +4 -> focused=7

    assert manager.focused_duration_seconds == pytest.approx(7.0)


def test_end_session_while_paused_does_not_double_count_or_add_gap() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.FOCUSED, 0.0))  # seed
    manager.record_transition(transition(FocusState.FOCUSED, 5.0))  # +5 -> focused=5
    manager.pause_session(5.0)

    summary = manager.end_session(500.0)  # ended while still paused

    assert summary.focused_duration_seconds == pytest.approx(5.0)
    assert summary.total_duration_seconds == pytest.approx(500.0)  # total still reflects full wall time


def test_end_session_while_active_credits_final_interval() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.FOCUSED, 0.0))  # seed
    manager.record_transition(transition(FocusState.FOCUSED, 5.0))  # +5 -> focused=5

    summary = manager.end_session(9.0)  # final 5 -> 9 interval credited on end: +4 -> focused=9

    assert summary.focused_duration_seconds == pytest.approx(9.0)


# --- Score + counts: each penalty-bearing event type -----------------------------------------


def test_phone_detected_increments_count_and_applies_penalty() -> None:
    manager = make_manager()
    manager.start_session(0.0)

    manager.record_event(make_event(EventType.PHONE_DETECTED, 1.0))

    assert manager.phone_distraction_count == 1
    assert manager.focus_score == STARTING_SCORE - PHONE_PENALTY


def test_drowsiness_signal_increments_count_and_applies_penalty() -> None:
    manager = make_manager()
    manager.start_session(0.0)

    manager.record_event(make_event(EventType.DROWSINESS_SIGNAL, 1.0))

    assert manager.drowsiness_count == 1
    assert manager.focus_score == STARTING_SCORE - DROWSY_PENALTY


def test_attention_diverted_increments_count_and_applies_penalty() -> None:
    manager = make_manager()
    manager.start_session(0.0)

    manager.record_event(make_event(EventType.ATTENTION_DIVERTED, 1.0))

    assert manager.attention_diversion_count == 1
    assert manager.focus_score == STARTING_SCORE - ATTENTION_PENALTY


def test_person_left_increments_away_count_and_applies_penalty() -> None:
    manager = make_manager()
    manager.start_session(0.0)

    manager.record_event(make_event(EventType.PERSON_LEFT, 1.0))

    assert manager.away_count == 1
    assert manager.focus_score == STARTING_SCORE - AWAY_PENALTY


@pytest.mark.parametrize(
    "event_type",
    [
        EventType.PHONE_CLEARED,
        EventType.DROWSINESS_CLEARED,
        EventType.ATTENTION_RESTORED,
        EventType.PERSON_RETURNED,
        EventType.FOCUS_RESTORED,
        EventType.SESSION_STARTED,
        EventType.SESSION_ENDED,
        EventType.CAMERA_ERROR,
        EventType.MODEL_ERROR,
        EventType.VISION_ERROR,
    ],
)
def test_non_penalty_event_types_do_not_change_score_or_counts(event_type: EventType) -> None:
    manager = make_manager()
    manager.start_session(0.0)

    manager.record_event(make_event(event_type, 1.0))

    assert manager.focus_score == STARTING_SCORE
    assert manager.phone_distraction_count == 0
    assert manager.drowsiness_count == 0
    assert manager.attention_diversion_count == 0
    assert manager.away_count == 0


def test_non_penalty_event_is_still_recorded_in_summary_events() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_event(make_event(EventType.FOCUS_RESTORED, 1.0))

    summary = manager.end_session(2.0)

    assert len(summary.events) == 1
    assert summary.events[0].event_type == EventType.FOCUS_RESTORED


def test_score_never_goes_below_zero() -> None:
    manager = make_manager(phone_event_penalty=60)
    manager.start_session(0.0)

    manager.record_event(make_event(EventType.PHONE_DETECTED, 1.0))
    manager.record_event(make_event(EventType.PHONE_DETECTED, 2.0))

    assert manager.focus_score == 0  # 100 - 60 - 60 clamps at 0, not -20


def test_multiple_event_types_compound_correctly() -> None:
    manager = make_manager()
    manager.start_session(0.0)

    manager.record_event(make_event(EventType.PHONE_DETECTED, 1.0))
    manager.record_event(make_event(EventType.DROWSINESS_SIGNAL, 2.0))
    manager.record_event(make_event(EventType.ATTENTION_DIVERTED, 3.0))
    manager.record_event(make_event(EventType.PERSON_LEFT, 4.0))

    expected = STARTING_SCORE - PHONE_PENALTY - DROWSY_PENALTY - ATTENTION_PENALTY - AWAY_PENALTY
    assert manager.focus_score == expected
    assert (
        manager.phone_distraction_count,
        manager.drowsiness_count,
        manager.attention_diversion_count,
        manager.away_count,
    ) == (1, 1, 1, 1)


# --- Monotonic timestamp validation --------------------------------------------------------


def test_record_transition_out_of_order_raises_value_error() -> None:
    manager = make_manager()
    manager.start_session(5.0)

    with pytest.raises(ValueError):
        manager.record_transition(transition(FocusState.FOCUSED, 4.0))


def test_record_event_out_of_order_raises_value_error() -> None:
    manager = make_manager()
    manager.start_session(5.0)

    with pytest.raises(ValueError):
        manager.record_event(make_event(EventType.PHONE_DETECTED, 4.0))


def test_pause_session_out_of_order_raises_value_error() -> None:
    manager = make_manager()
    manager.start_session(5.0)

    with pytest.raises(ValueError):
        manager.pause_session(4.0)


def test_resume_session_out_of_order_raises_value_error() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.pause_session(5.0)

    with pytest.raises(ValueError):
        manager.resume_session(4.0)


def test_end_session_out_of_order_raises_value_error() -> None:
    manager = make_manager()
    manager.start_session(5.0)

    with pytest.raises(ValueError):
        manager.end_session(4.0)


def test_event_at_exactly_the_last_transition_timestamp_is_accepted() -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.FOCUSED, 5.0))

    manager.record_event(make_event(EventType.PHONE_DETECTED, 5.0))  # equal, not raised

    assert manager.phone_distraction_count == 1


# --- elapsed_seconds() ----------------------------------------------------------------------


def test_elapsed_seconds_zero_when_not_active() -> None:
    manager = make_manager()
    assert manager.elapsed_seconds(100.0) == 0.0


def test_elapsed_seconds_reflects_time_since_start() -> None:
    manager = make_manager()
    manager.start_session(10.0)

    assert manager.elapsed_seconds(35.0) == pytest.approx(25.0)


# --- Full realistic end-to-end scenario (PRD section 27 analytics) --------------------------


def test_realistic_full_session_scenario() -> None:
    """Each record_transition's timestamp is the BOUNDARY where the new
    state begins - the interval leading up to it belongs to the OLD
    state. Plain absolute timestamps are used throughout (rather than
    incremental ts(start, duration) sums) so every interval boundary is
    unambiguous at a glance."""
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.FOCUSED, 0.0))  # seed: FOCUSED as of t=0

    # FOCUSED [0, 100)
    manager.record_transition(transition(FocusState.PHONE_DISTRACTION, 100.0, previous=FocusState.FOCUSED))
    manager.record_event(make_event(EventType.PHONE_DETECTED, 100.0, Severity.WARNING))

    # PHONE_DISTRACTION [100, 120)
    manager.record_transition(transition(FocusState.FOCUSED, 120.0, previous=FocusState.PHONE_DISTRACTION))
    manager.record_event(make_event(EventType.PHONE_CLEARED, 120.1))
    manager.record_event(make_event(EventType.FOCUS_RESTORED, 120.2))

    # FOCUSED [120, 170)
    manager.record_transition(transition(FocusState.DROWSINESS_SIGNAL, 170.0, previous=FocusState.FOCUSED))
    manager.record_event(make_event(EventType.DROWSINESS_SIGNAL, 170.0, Severity.WARNING))

    # DROWSINESS_SIGNAL [170, 171)
    manager.record_transition(transition(FocusState.FOCUSED, 171.0, previous=FocusState.DROWSINESS_SIGNAL))

    # FOCUSED [171, 201)
    manager.record_transition(transition(FocusState.AWAY, 201.0, previous=FocusState.FOCUSED))
    manager.record_event(make_event(EventType.PERSON_LEFT, 201.0, Severity.WARNING))

    # AWAY [201, 211) - away has no duration bucket per PRD section 25 (count only)
    manager.record_transition(transition(FocusState.FOCUSED, 211.0, previous=FocusState.AWAY))

    # FOCUSED [211, 216), then end
    summary = manager.end_session(216.0)

    assert summary.total_duration_seconds == pytest.approx(216.0)
    # focused: 100 + 50 + 30 + 5 = 185
    assert summary.focused_duration_seconds == pytest.approx(185.0)
    assert summary.phone_distraction_duration_seconds == pytest.approx(20.0)
    assert summary.phone_distraction_count == 1
    assert summary.drowsiness_count == 1
    assert summary.attention_diversion_count == 0
    assert summary.away_count == 1
    # longest streak: max(100, 50, 30, 5) = 100
    assert summary.longest_focus_streak_seconds == pytest.approx(100.0)
    expected_score = STARTING_SCORE - PHONE_PENALTY - DROWSY_PENALTY - AWAY_PENALTY
    assert summary.focus_score == expected_score
    assert len(summary.events) == 5  # PHONE_DETECTED, PHONE_CLEARED, FOCUS_RESTORED, DROWSINESS_SIGNAL, PERSON_LEFT


# --- JSON persistence -------------------------------------------------------------------------


def test_save_summary_json_writes_expected_filename(tmp_path) -> None:
    manager = make_manager()
    manager.start_session(0.0)
    summary = manager.end_session(10.0)

    path = SessionManager.save_summary_json(summary, directory=tmp_path)

    expected_name = f"session_{summary.started_at.strftime('%Y%m%d_%H%M%S')}.json"
    assert path.name == expected_name
    assert path.parent == tmp_path
    assert path.exists()


def test_save_summary_json_content_round_trips(tmp_path) -> None:
    manager = make_manager()
    manager.start_session(0.0)
    manager.record_transition(transition(FocusState.FOCUSED, 5.0))
    manager.record_event(make_event(EventType.PHONE_DETECTED, 5.0, Severity.WARNING))
    summary = manager.end_session(10.0)

    path = SessionManager.save_summary_json(summary, directory=tmp_path)

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    assert data["total_duration_seconds"] == pytest.approx(10.0)
    assert data["focused_duration_seconds"] == pytest.approx(5.0)
    assert data["phone_distraction_count"] == 1
    assert data["focus_score"] == STARTING_SCORE - PHONE_PENALTY
    assert len(data["events"]) == 1
    assert data["events"][0]["event_type"] == "PHONE_DETECTED"
    assert data["events"][0]["severity"] == "WARNING"
    assert data["events"][0]["timestamp"] == pytest.approx(5.0)


def test_save_summary_json_creates_missing_directory(tmp_path) -> None:
    manager = make_manager()
    manager.start_session(0.0)
    summary = manager.end_session(1.0)
    nested = tmp_path / "nested" / "logs"

    path = SessionManager.save_summary_json(summary, directory=nested)

    assert path.exists()
    assert nested.exists()


def test_save_summary_json_filename_uses_session_start_not_end_time(tmp_path) -> None:
    manager = make_manager()
    manager.start_session(0.0)
    summary = manager.end_session(999999.0)  # a huge duration - filename must not use end time

    path = SessionManager.save_summary_json(summary, directory=tmp_path)

    assert path.name == f"session_{summary.started_at.strftime('%Y%m%d_%H%M%S')}.json"
    assert path.name != f"session_{summary.ended_at.strftime('%Y%m%d_%H%M%S')}.json" or (
        summary.started_at.strftime("%Y%m%d_%H%M%S") == summary.ended_at.strftime("%Y%m%d_%H%M%S")
    )


def test_default_logs_directory_points_at_project_logs_folder() -> None:
    from src.session.session_manager import DEFAULT_LOGS_DIRECTORY, PROJECT_ROOT

    assert DEFAULT_LOGS_DIRECTORY == PROJECT_ROOT / "logs"
