"""Tests for DrowsinessFilter (FOCUSGUARD_PRD.md section 12).

Fully deterministic: synthetic EyeState/timestamp sequences only. No
webcam, GPU, or MediaPipe model.
"""

from __future__ import annotations

import math

import pytest

from src.core.config_manager import EyesConfig
from src.face.eye_metrics import EyeState
from src.state.drowsiness_filter import DrowsinessFilter
from src.state.temporal_filter import DurationConfirmerState

DROWSY = 1.20
BLINK_MAX = 0.45


def make_config(drowsiness_duration_seconds: float = DROWSY) -> EyesConfig:
    return EyesConfig(
        closed_threshold=0.21,
        open_threshold=0.24,
        blink_max_duration_seconds=BLINK_MAX,
        drowsiness_duration_seconds=drowsiness_duration_seconds,
    )


def ts(*parts: float) -> float:
    return round(math.fsum(parts), 9)


# --- Initial state -----------------------------------------------------------


def test_initial_state_is_inactive_and_not_drowsy() -> None:
    filt = DrowsinessFilter(make_config())

    assert filt.state == DurationConfirmerState.INACTIVE
    result = filt.update(EyeState.OPEN, 0.0)
    assert result.is_drowsy is False


# --- PRD section 36: short closure -> blink (no confirmed event) ------------


def test_short_closure_shorter_than_blink_max_never_confirms_drowsy() -> None:
    filt = DrowsinessFilter(make_config())

    filt.update(EyeState.OPEN, 0.0)
    filt.update(EyeState.CLOSED, 0.1)
    result = filt.update(EyeState.OPEN, ts(0.1, BLINK_MAX - 0.1))  # closed for ~0.35s, a normal blink

    assert result.is_drowsy is False
    assert result.just_confirmed is False


def test_closure_between_blink_max_and_drowsiness_duration_does_not_confirm_yet() -> None:
    """The PRD only defines events at the blink boundary (no event) and
    the drowsiness boundary (DROWSINESS_SIGNAL) - the gap between them is
    simply 'still closed, not yet confirmed', with no event either way."""
    filt = DrowsinessFilter(make_config())
    filt.update(EyeState.CLOSED, 0.0)

    result = filt.update(EyeState.CLOSED, BLINK_MAX + 0.1)  # past blink_max, still well under drowsy threshold

    assert result.is_drowsy is False
    assert result.just_confirmed is False


# --- PRD section 36: long closure -> drowsiness -----------------------------


def test_closure_exactly_at_drowsiness_threshold_confirms() -> None:
    filt = DrowsinessFilter(make_config())
    filt.update(EyeState.CLOSED, 0.0)

    result = filt.update(EyeState.CLOSED, DROWSY)

    assert result.is_drowsy is True
    assert result.just_confirmed is True


def test_closure_above_drowsiness_threshold_confirms() -> None:
    filt = DrowsinessFilter(make_config())
    filt.update(EyeState.CLOSED, 0.0)

    result = filt.update(EyeState.CLOSED, DROWSY + 0.5)

    assert result.is_drowsy is True
    assert result.just_confirmed is True


def test_just_confirmed_fires_exactly_once() -> None:
    filt = DrowsinessFilter(make_config())
    filt.update(EyeState.CLOSED, 0.0)

    confirmed = filt.update(EyeState.CLOSED, DROWSY)
    repeat = filt.update(EyeState.CLOSED, DROWSY + 0.1)

    assert confirmed.just_confirmed is True
    assert repeat.just_confirmed is False


# --- Clearing: immediate once eyes reopen (no grace period) -----------------


def test_eyes_reopening_after_confirmed_drowsiness_clears_immediately() -> None:
    filt = DrowsinessFilter(make_config())
    filt.update(EyeState.CLOSED, 0.0)
    filt.update(EyeState.CLOSED, DROWSY)

    result = filt.update(EyeState.OPEN, ts(DROWSY, 0.01))

    assert result.is_drowsy is False
    assert result.just_cleared is True


# --- PRD section 36 / section 10: missing landmarks -> UNKNOWN never CLOSED --


def test_unknown_eye_state_never_starts_or_continues_confirmation() -> None:
    filt = DrowsinessFilter(make_config())

    result = filt.update(EyeState.UNKNOWN, 0.0)
    assert result.is_drowsy is False
    assert result.state == DurationConfirmerState.INACTIVE

    result = filt.update(EyeState.UNKNOWN, 100.0)
    assert result.is_drowsy is False
    assert result.state == DurationConfirmerState.INACTIVE


def test_unknown_during_closed_confirmation_resets_progress() -> None:
    filt = DrowsinessFilter(make_config())
    filt.update(EyeState.CLOSED, 0.0)

    # Landmarks lost partway through - must not silently continue counting
    # toward drowsiness while state is unreliable.
    filt.update(EyeState.UNKNOWN, ts(0.0, DROWSY / 2))
    result = filt.update(EyeState.CLOSED, ts(0.0, DROWSY / 2, 0.1))

    assert result.is_drowsy is False
    assert result.just_confirmed is False


# --- Rapid oscillation never confirms ----------------------------------------


def test_rapid_blinking_never_confirms_drowsiness() -> None:
    filt = DrowsinessFilter(make_config())
    t = 0.0
    step = DROWSY / 8

    for _ in range(20):
        result = filt.update(EyeState.CLOSED, t)
        assert result.is_drowsy is False
        t = ts(t, step)
        result = filt.update(EyeState.OPEN, t)
        assert result.is_drowsy is False
        t = ts(t, step)


# --- Timestamp validation ------------------------------------------------------


def test_out_of_order_timestamp_raises_value_error() -> None:
    filt = DrowsinessFilter(make_config())
    filt.update(EyeState.CLOSED, 5.0)

    with pytest.raises(ValueError):
        filt.update(EyeState.CLOSED, 4.0)


def test_timestamp_propagates_into_result() -> None:
    filt = DrowsinessFilter(make_config())

    result = filt.update(EyeState.OPEN, 42.5)

    assert result.timestamp == 42.5


# --- Zero-duration configuration ------------------------------------------------


def test_zero_drowsiness_duration_confirms_immediately() -> None:
    filt = DrowsinessFilter(make_config(drowsiness_duration_seconds=0.0))

    result = filt.update(EyeState.CLOSED, 0.0)

    assert result.is_drowsy is True
    assert result.just_confirmed is True
