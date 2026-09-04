"""Tests for HeadOrientationFilter (FOCUSGUARD_PRD.md section 13).

Fully deterministic: synthetic HeadOrientation/timestamp sequences only.
No webcam, GPU, MediaPipe, or real-time waiting.
"""

from __future__ import annotations

import math

import pytest

from src.core.config_manager import HeadConfig
from src.face.head_pose import HeadOrientation
from src.state.head_orientation_filter import HeadOrientationFilter, HeadOrientationFilterState

CONFIRM = 0.80


def make_config(confirmation_seconds: float = CONFIRM) -> HeadConfig:
    return HeadConfig(
        yaw_threshold_degrees=20.0,
        pitch_threshold_degrees=18.0,
        confirmation_seconds=confirmation_seconds,
        calibration_seconds=0.0,
    )


def ts(*parts: float) -> float:
    """Precise timestamp summation - avoids the binary float drift that
    chained `a + b + c` addition can introduce at exact boundaries."""
    return round(math.fsum(parts), 9)


def divert(filt: HeadOrientationFilter, start_ts: float = 0.0, orientation: HeadOrientation = HeadOrientation.LEFT) -> float:
    """Drive filt from CENTERED to DIVERTED. Returns the confirmation timestamp."""
    filt.update(orientation, start_ts)
    confirmed_ts = ts(start_ts, CONFIRM)
    result = filt.update(orientation, confirmed_ts)
    assert result.state == HeadOrientationFilterState.DIVERTED
    return confirmed_ts


# --- 1. Initial state -----------------------------------------------------------


def test_initial_state_is_centered() -> None:
    filt = HeadOrientationFilter(make_config())

    assert filt.state == HeadOrientationFilterState.CENTERED


# --- 2. First off-center reading enters DIVERTING -------------------------------


def test_first_off_center_enters_diverting() -> None:
    filt = HeadOrientationFilter(make_config())

    result = filt.update(HeadOrientation.LEFT, 0.0)

    assert result.state == HeadOrientationFilterState.DIVERTING
    assert result.is_diverted is False
    assert result.just_diverted is False


# --- 3/4/5. Confirmation threshold behavior -------------------------------------


def test_below_confirmation_threshold_never_diverts_yet() -> None:
    filt = HeadOrientationFilter(make_config())
    filt.update(HeadOrientation.LEFT, 0.0)

    result = filt.update(HeadOrientation.LEFT, CONFIRM - 0.1)

    assert result.state == HeadOrientationFilterState.DIVERTING
    assert result.is_diverted is False


def test_exactly_at_confirmation_threshold_diverts() -> None:
    filt = HeadOrientationFilter(make_config())
    filt.update(HeadOrientation.LEFT, 0.0)

    result = filt.update(HeadOrientation.LEFT, CONFIRM)

    assert result.state == HeadOrientationFilterState.DIVERTED
    assert result.is_diverted is True
    assert result.just_diverted is True


def test_exactly_at_confirmation_threshold_diverts_at_realistic_large_timestamp_magnitude() -> None:
    """Regression test: at large timestamp magnitudes (realistic for
    time.monotonic(), which reports seconds-since-boot), plain float
    subtraction can land a hair below the exact threshold - e.g.
    100.8 - 100.0 == 0.7999999999999972 in IEEE754, not exactly 0.8 - even
    though both operands were produced by ordinary decimal arithmetic. This
    confirms the boundary comparison tolerates that noise."""
    base = 100_000.0  # realistic system-uptime-scale timestamp
    filt = HeadOrientationFilter(make_config())
    filt.update(HeadOrientation.LEFT, base)

    result = filt.update(HeadOrientation.LEFT, base + CONFIRM)

    assert result.state == HeadOrientationFilterState.DIVERTED
    assert result.just_diverted is True


def test_above_confirmation_threshold_diverts() -> None:
    filt = HeadOrientationFilter(make_config())
    filt.update(HeadOrientation.LEFT, 0.0)

    result = filt.update(HeadOrientation.LEFT, CONFIRM + 0.2)

    assert result.state == HeadOrientationFilterState.DIVERTED
    assert result.just_diverted is True


# --- 6. just_diverted fires exactly once -----------------------------------------


def test_just_diverted_true_exactly_once() -> None:
    filt = HeadOrientationFilter(make_config())
    filt.update(HeadOrientation.LEFT, 0.0)

    diverted_result = filt.update(HeadOrientation.LEFT, CONFIRM)
    repeat_result = filt.update(HeadOrientation.LEFT, CONFIRM + 0.1)

    assert diverted_result.just_diverted is True
    assert repeat_result.just_diverted is False


# --- 7. Remains DIVERTED while continuously non-centered ------------------------


def test_remains_diverted_while_continuously_off_center() -> None:
    filt = HeadOrientationFilter(make_config())
    confirmed_ts = divert(filt)

    result = filt.update(HeadOrientation.LEFT, ts(confirmed_ts, 1.0))

    assert result.state == HeadOrientationFilterState.DIVERTED
    assert result.is_diverted is True


# --- 8/9. Direction switching does not reset timer or clear diversion ----------


def test_switching_direction_while_diverting_does_not_reset_timer() -> None:
    filt = HeadOrientationFilter(make_config())
    filt.update(HeadOrientation.LEFT, 0.0)
    filt.update(HeadOrientation.RIGHT, ts(0.0, CONFIRM / 2))  # switch direction, still off-center

    result = filt.update(HeadOrientation.UP, CONFIRM)  # total elapsed since t=0 reaches threshold

    assert result.state == HeadOrientationFilterState.DIVERTED
    assert result.just_diverted is True


def test_switching_direction_while_diverted_does_not_restore() -> None:
    filt = HeadOrientationFilter(make_config())
    confirmed_ts = divert(filt, orientation=HeadOrientation.LEFT)

    result = filt.update(HeadOrientation.RIGHT, ts(confirmed_ts, 0.1))

    assert result.state == HeadOrientationFilterState.DIVERTED
    assert result.just_restored is False
    assert result.just_diverted is False


# --- 10. Return to CENTER while DIVERTING never diverts -------------------------


def test_return_to_center_while_diverting_never_diverts() -> None:
    filt = HeadOrientationFilter(make_config())
    filt.update(HeadOrientation.LEFT, 0.0)

    result = filt.update(HeadOrientation.CENTER, CONFIRM - 0.1)

    assert result.state == HeadOrientationFilterState.CENTERED
    assert result.just_diverted is False


# --- 11. Return to CENTER while DIVERTED restores immediately ------------------


def test_return_to_center_while_diverted_restores_immediately() -> None:
    filt = HeadOrientationFilter(make_config())
    confirmed_ts = divert(filt)

    result = filt.update(HeadOrientation.CENTER, ts(confirmed_ts, 0.01))

    assert result.state == HeadOrientationFilterState.CENTERED
    assert result.is_diverted is False
    assert result.just_restored is True


def test_just_restored_true_exactly_once() -> None:
    filt = HeadOrientationFilter(make_config())
    confirmed_ts = divert(filt)
    restored_ts = ts(confirmed_ts, 0.01)

    restored_result = filt.update(HeadOrientation.CENTER, restored_ts)
    repeat_result = filt.update(HeadOrientation.CENTER, ts(restored_ts, 0.1))

    assert restored_result.just_restored is True
    assert repeat_result.just_restored is False


# --- 12/13. UNKNOWN treated as centered for timer purposes ----------------------


def test_unknown_during_diverting_resets_timer() -> None:
    filt = HeadOrientationFilter(make_config())
    filt.update(HeadOrientation.LEFT, 0.0)
    filt.update(HeadOrientation.UNKNOWN, ts(0.0, CONFIRM / 2))

    # Re-diverting from here needs a fresh full confirmation duration.
    result = filt.update(HeadOrientation.LEFT, ts(0.0, CONFIRM / 2 + 0.1))

    assert result.state == HeadOrientationFilterState.DIVERTING
    assert result.just_diverted is False


def test_unknown_during_diverted_restores_immediately() -> None:
    filt = HeadOrientationFilter(make_config())
    confirmed_ts = divert(filt)

    result = filt.update(HeadOrientation.UNKNOWN, ts(confirmed_ts, 0.01))

    assert result.state == HeadOrientationFilterState.CENTERED
    assert result.is_diverted is False
    assert result.just_restored is True


# --- 14. Zero-duration configuration --------------------------------------------


def test_zero_confirmation_duration_diverts_immediately() -> None:
    filt = HeadOrientationFilter(make_config(confirmation_seconds=0.0))

    result = filt.update(HeadOrientation.LEFT, 0.0)

    assert result.state == HeadOrientationFilterState.DIVERTED
    assert result.is_diverted is True
    assert result.just_diverted is True


# --- 15/16. Timestamp validation -------------------------------------------------


def test_duplicate_timestamps_are_valid_with_zero_elapsed() -> None:
    filt = HeadOrientationFilter(make_config())

    filt.update(HeadOrientation.LEFT, 5.0)
    result = filt.update(HeadOrientation.LEFT, 5.0)

    assert result.state == HeadOrientationFilterState.DIVERTING
    assert result.just_diverted is False


def test_out_of_order_timestamp_raises_value_error() -> None:
    filt = HeadOrientationFilter(make_config())
    filt.update(HeadOrientation.LEFT, 5.0)

    with pytest.raises(ValueError):
        filt.update(HeadOrientation.LEFT, 4.0)


# --- 17. Timestamp-based, not call-count-based ----------------------------------


def test_long_gap_between_updates_uses_timestamp_not_call_count() -> None:
    filt = HeadOrientationFilter(make_config())
    filt.update(HeadOrientation.LEFT, 0.0)

    result = filt.update(HeadOrientation.LEFT, 100.0)

    assert result.state == HeadOrientationFilterState.DIVERTED
    assert result.just_diverted is True


def test_many_calls_within_short_timestamp_span_do_not_divert() -> None:
    filt = HeadOrientationFilter(make_config())
    filt.update(HeadOrientation.LEFT, 0.0)

    result = None
    for i in range(1, 50):
        result = filt.update(HeadOrientation.LEFT, i * (CONFIRM / 100))

    assert result.state == HeadOrientationFilterState.DIVERTING
    assert result.just_diverted is False


# --- 18. Rapid oscillation never diverts ----------------------------------------


def test_rapid_oscillation_shorter_than_confirmation_duration_never_diverts() -> None:
    filt = HeadOrientationFilter(make_config())
    t = 0.0
    step = CONFIRM / 4

    for _ in range(20):
        result = filt.update(HeadOrientation.LEFT, t)
        assert result.state != HeadOrientationFilterState.DIVERTED
        t = ts(t, step)
        result = filt.update(HeadOrientation.CENTER, t)
        assert result.state == HeadOrientationFilterState.CENTERED
        t = ts(t, step)


# --- 19. Multiple cycles do not leak stale state --------------------------------


def test_multiple_complete_cycles_do_not_leak_stale_state() -> None:
    filt = HeadOrientationFilter(make_config())

    # Each cycle starts from a clean, independent round-number base (rather
    # than chaining arithmetic off the previous cycle's already-rounded
    # values) so per-cycle boundary checks stay exact - large gaps between
    # cycles make the precise numeric relationship between cycles
    # irrelevant to what this test actually verifies (no stale state leaks).
    for cycle_index in range(3):
        cycle_start = float(cycle_index * 100)

        filt.update(HeadOrientation.LEFT, cycle_start)
        diverted = filt.update(HeadOrientation.LEFT, ts(cycle_start, CONFIRM))
        assert diverted.state == HeadOrientationFilterState.DIVERTED
        assert diverted.just_diverted is True

        restored_ts = ts(cycle_start, CONFIRM, 0.1)
        restored = filt.update(HeadOrientation.CENTER, restored_ts)
        assert restored.state == HeadOrientationFilterState.CENTERED
        assert restored.just_restored is True

    assert filt.state == HeadOrientationFilterState.CENTERED


# --- 20. raw_orientation and timestamp propagation ------------------------------


def test_raw_orientation_and_timestamp_propagate() -> None:
    filt = HeadOrientationFilter(make_config())

    result = filt.update(HeadOrientation.UP, 42.5)

    assert result.raw_orientation == HeadOrientation.UP
    assert result.timestamp == 42.5


# --- 21. is_diverted matches logical state --------------------------------------


@pytest.mark.parametrize(
    "state,expected_is_diverted",
    [
        (HeadOrientationFilterState.CENTERED, False),
        (HeadOrientationFilterState.DIVERTING, False),
        (HeadOrientationFilterState.DIVERTED, True),
    ],
)
def test_is_diverted_matches_logical_state(state: HeadOrientationFilterState, expected_is_diverted: bool) -> None:
    filt = HeadOrientationFilter(make_config())

    if state == HeadOrientationFilterState.CENTERED:
        result = filt.update(HeadOrientation.CENTER, 0.0)
    elif state == HeadOrientationFilterState.DIVERTING:
        result = filt.update(HeadOrientation.LEFT, 0.0)
    else:  # DIVERTED
        filt.update(HeadOrientation.LEFT, 0.0)
        result = filt.update(HeadOrientation.LEFT, CONFIRM)

    assert result.state == state
    assert result.is_diverted is expected_is_diverted


# --- 22. elapsed_in_state_seconds (Phase 8: debug-mode timers) --------------------


def test_elapsed_in_state_seconds_is_none_while_centered() -> None:
    filt = HeadOrientationFilter(make_config())

    assert filt.elapsed_in_state_seconds(0.0) is None


def test_elapsed_in_state_seconds_while_diverting() -> None:
    filt = HeadOrientationFilter(make_config())
    filt.update(HeadOrientation.LEFT, 0.0)

    assert filt.elapsed_in_state_seconds(0.1) == pytest.approx(0.1)


def test_elapsed_in_state_seconds_is_none_once_diverted() -> None:
    filt = HeadOrientationFilter(make_config())
    confirmed_ts = divert(filt)

    assert filt.elapsed_in_state_seconds(ts(confirmed_ts, 5.0)) is None
