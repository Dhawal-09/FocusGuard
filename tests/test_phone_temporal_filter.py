"""Tests for PhoneTemporalFilter (FOCUSGUARD_PRD.md section 9).

Fully deterministic: synthetic boolean/timestamp sequences only. No
webcam, GPU, YOLO, MediaPipe, model downloads, or real-time waiting.
"""

from __future__ import annotations

import math

import pytest

from src.core.config_manager import PhoneConfig
from src.state.phone_temporal_filter import PhoneFilterState, PhoneTemporalFilter

CONFIRM = 0.35
CLEAR = 0.60


def make_config(confirm_duration_seconds: float = CONFIRM, clear_duration_seconds: float = CLEAR) -> PhoneConfig:
    return PhoneConfig(
        confirm_duration_seconds=confirm_duration_seconds,
        clear_duration_seconds=clear_duration_seconds,
        warning_cooldown_seconds=10.0,
    )


def ts(*parts: float) -> float:
    """Sum timestamp components with precise (non-drifting) summation.

    Chained `a + b + c` float addition can land a hair below an exact
    threshold (e.g. 0.35 + 0.1 + 0.60 != 1.05 bit-for-bit), which would
    make "exactly at threshold" tests flaky by construction rather than
    by anything the implementation does. math.fsum avoids that drift.
    """
    return round(math.fsum(parts), 9)


def confirm(filt: PhoneTemporalFilter, start_ts: float = 0.0) -> float:
    """Drive filt from NOT_DETECTED to CONFIRMED via a two-step detection
    (start, then exactly at threshold). Returns the timestamp of confirmation."""
    filt.update(True, start_ts)
    confirmed_ts = ts(start_ts, CONFIRM)
    result = filt.update(True, confirmed_ts)
    assert result.state == PhoneFilterState.CONFIRMED
    return confirmed_ts


# --- 1. Initial state ----------------------------------------------------------


def test_initial_state_is_not_detected() -> None:
    filt = PhoneTemporalFilter(make_config())

    assert filt.state == PhoneFilterState.NOT_DETECTED


# --- 2. First detection enters CONFIRMING --------------------------------------


def test_first_true_enters_confirming() -> None:
    filt = PhoneTemporalFilter(make_config())

    result = filt.update(True, 0.0)

    assert result.state == PhoneFilterState.CONFIRMING
    assert result.is_confirmed is False
    assert result.just_confirmed is False


# --- 3/4/5. Confirmation threshold behavior ------------------------------------


def test_detection_below_confirmation_threshold_never_confirms_yet() -> None:
    filt = PhoneTemporalFilter(make_config())
    filt.update(True, 0.0)

    result = filt.update(True, CONFIRM - 0.05)

    assert result.state == PhoneFilterState.CONFIRMING
    assert result.is_confirmed is False


def test_detection_exactly_at_confirmation_threshold_confirms() -> None:
    filt = PhoneTemporalFilter(make_config())
    filt.update(True, 0.0)

    result = filt.update(True, CONFIRM)

    assert result.state == PhoneFilterState.CONFIRMED
    assert result.is_confirmed is True
    assert result.just_confirmed is True


def test_detection_above_confirmation_threshold_confirms() -> None:
    filt = PhoneTemporalFilter(make_config())
    filt.update(True, 0.0)

    result = filt.update(True, CONFIRM + 0.10)

    assert result.state == PhoneFilterState.CONFIRMED
    assert result.just_confirmed is True


# --- 6/7. just_confirmed fires exactly once ------------------------------------


def test_just_confirmed_true_exactly_once() -> None:
    filt = PhoneTemporalFilter(make_config())
    filt.update(True, 0.0)

    confirming_result = filt.update(True, CONFIRM)
    repeat_result = filt.update(True, CONFIRM + 0.1)

    assert confirming_result.just_confirmed is True
    assert repeat_result.just_confirmed is False


def test_subsequent_confirmed_frames_do_not_repeat_just_confirmed() -> None:
    filt = PhoneTemporalFilter(make_config())
    confirm(filt)

    for offset in (0.1, 0.2, 0.3):
        result = filt.update(True, CONFIRM + offset)
        assert result.just_confirmed is False
        assert result.state == PhoneFilterState.CONFIRMED


# --- 8. Confirmed phone remains confirmed while detected -----------------------


def test_confirmed_phone_remains_confirmed_while_detected() -> None:
    filt = PhoneTemporalFilter(make_config())
    confirm(filt)

    result = filt.update(True, CONFIRM + 1.0)

    assert result.state == PhoneFilterState.CONFIRMED
    assert result.is_confirmed is True


# --- 9/10. CONFIRMED + absence enters CLEARING, stays confirmed ----------------


def test_confirmed_plus_absence_enters_clearing() -> None:
    filt = PhoneTemporalFilter(make_config())
    confirmed_ts = confirm(filt)

    result = filt.update(False, ts(confirmed_ts, 0.1))

    assert result.state == PhoneFilterState.CLEARING


def test_is_confirmed_remains_true_during_clearing() -> None:
    filt = PhoneTemporalFilter(make_config())
    confirmed_ts = confirm(filt)
    filt.update(False, ts(confirmed_ts, 0.1))

    result = filt.update(False, ts(confirmed_ts, 0.2))

    assert result.state == PhoneFilterState.CLEARING
    assert result.is_confirmed is True


# --- 11/12/13. Clear threshold behavior -----------------------------------------


def test_absence_below_clear_threshold_does_not_clear() -> None:
    filt = PhoneTemporalFilter(make_config())
    confirmed_ts = confirm(filt)
    clearing_start = ts(confirmed_ts, 0.1)
    filt.update(False, clearing_start)

    result = filt.update(False, ts(clearing_start, CLEAR - 0.05))

    assert result.state == PhoneFilterState.CLEARING
    assert result.just_cleared is False


def test_absence_exactly_at_clear_threshold_clears() -> None:
    filt = PhoneTemporalFilter(make_config())
    confirmed_ts = confirm(filt)
    clearing_start = ts(confirmed_ts, 0.1)
    filt.update(False, clearing_start)

    result = filt.update(False, ts(clearing_start, CLEAR))

    assert result.state == PhoneFilterState.NOT_DETECTED
    assert result.is_confirmed is False
    assert result.just_cleared is True


def test_absence_above_clear_threshold_clears() -> None:
    filt = PhoneTemporalFilter(make_config())
    confirmed_ts = confirm(filt)
    clearing_start = ts(confirmed_ts, 0.1)
    filt.update(False, clearing_start)

    result = filt.update(False, ts(clearing_start, CLEAR, 0.2))

    assert result.state == PhoneFilterState.NOT_DETECTED
    assert result.just_cleared is True


# --- 14/15. just_cleared fires exactly once -------------------------------------


def test_just_cleared_true_exactly_once() -> None:
    filt = PhoneTemporalFilter(make_config())
    confirmed_ts = confirm(filt)
    clearing_start = ts(confirmed_ts, 0.1)
    filt.update(False, clearing_start)

    cleared_result = filt.update(False, ts(clearing_start, CLEAR))
    repeat_result = filt.update(False, ts(clearing_start, CLEAR, 0.1))

    assert cleared_result.just_cleared is True
    assert repeat_result.just_cleared is False


def test_subsequent_not_detected_frames_do_not_repeat_just_cleared() -> None:
    filt = PhoneTemporalFilter(make_config())
    confirmed_ts = confirm(filt)
    clearing_start = ts(confirmed_ts, 0.1)
    filt.update(False, clearing_start)
    filt.update(False, ts(clearing_start, CLEAR))

    for offset in (0.1, 0.2, 0.3):
        result = filt.update(False, ts(clearing_start, CLEAR, offset))
        assert result.just_cleared is False
        assert result.state == PhoneFilterState.NOT_DETECTED


# --- 16/17. Reappearance during CLEARING ----------------------------------------


def test_phone_reappears_during_clearing_returns_immediately_to_confirmed() -> None:
    filt = PhoneTemporalFilter(make_config())
    confirmed_ts = confirm(filt)
    clearing_start = ts(confirmed_ts, 0.1)
    filt.update(False, clearing_start)

    result = filt.update(True, clearing_start + 0.1)

    assert result.state == PhoneFilterState.CONFIRMED
    assert result.is_confirmed is True


def test_reappearance_during_clearing_does_not_restart_confirmation() -> None:
    filt = PhoneTemporalFilter(make_config())
    confirmed_ts = confirm(filt)
    clearing_start = ts(confirmed_ts, 0.1)
    filt.update(False, clearing_start)

    result = filt.update(True, clearing_start + 0.1)

    assert result.state != PhoneFilterState.CONFIRMING
    assert result.just_confirmed is False


# --- 18. New confirmation cycle after a genuine clear ---------------------------


def test_new_confirmation_cycle_after_genuine_clear() -> None:
    filt = PhoneTemporalFilter(make_config())
    confirmed_ts = confirm(filt)
    clearing_start = ts(confirmed_ts, 0.1)
    filt.update(False, clearing_start)
    cleared_result = filt.update(False, ts(clearing_start, CLEAR))
    assert cleared_result.state == PhoneFilterState.NOT_DETECTED

    new_cycle_start = ts(clearing_start, CLEAR, 1.0)
    filt.update(True, new_cycle_start)
    result = filt.update(True, ts(new_cycle_start, CONFIRM))

    assert result.state == PhoneFilterState.CONFIRMED
    assert result.just_confirmed is True


# --- 19. Rapid oscillation never confirms ---------------------------------------


def test_rapid_oscillation_shorter_than_confirm_duration_never_confirms() -> None:
    filt = PhoneTemporalFilter(make_config())
    ts = 0.0
    step = CONFIRM / 4  # each True/False period is well under confirm_duration_seconds

    for _ in range(20):
        result = filt.update(True, ts)
        assert result.state != PhoneFilterState.CONFIRMED
        assert result.just_confirmed is False
        ts += step
        result = filt.update(False, ts)
        assert result.state != PhoneFilterState.CONFIRMED
        ts += step

    assert filt.state in (PhoneFilterState.NOT_DETECTED, PhoneFilterState.CONFIRMING)


# --- 20/21. Zero-duration configuration -----------------------------------------


def test_zero_confirm_duration_confirms_immediately() -> None:
    filt = PhoneTemporalFilter(make_config(confirm_duration_seconds=0.0))

    result = filt.update(True, 0.0)

    assert result.state == PhoneFilterState.CONFIRMED
    assert result.is_confirmed is True
    assert result.just_confirmed is True


def test_zero_clear_duration_clears_immediately() -> None:
    filt = PhoneTemporalFilter(make_config(clear_duration_seconds=0.0))
    filt.update(True, 0.0)  # confirm_duration_seconds default (0.35) still applies
    confirmed = filt.update(True, CONFIRM)
    assert confirmed.state == PhoneFilterState.CONFIRMED

    result = filt.update(False, CONFIRM + 0.1)

    assert result.state == PhoneFilterState.NOT_DETECTED
    assert result.is_confirmed is False
    assert result.just_cleared is True


# --- 22/23. Timestamp validation -------------------------------------------------


def test_duplicate_timestamps_are_valid_with_zero_elapsed() -> None:
    filt = PhoneTemporalFilter(make_config())

    result = filt.update(True, 5.0)
    result = filt.update(True, 5.0)

    # Zero elapsed since state entry -> not confirmed yet (unless threshold is 0).
    assert result.state == PhoneFilterState.CONFIRMING
    assert result.just_confirmed is False


def test_out_of_order_timestamp_raises_value_error() -> None:
    filt = PhoneTemporalFilter(make_config())
    filt.update(True, 5.0)

    with pytest.raises(ValueError):
        filt.update(True, 4.0)


# --- 24. Timestamp-based, not frame-count-based ---------------------------------


def test_long_gap_between_updates_uses_timestamp_not_call_count() -> None:
    filt = PhoneTemporalFilter(make_config())
    filt.update(True, 0.0)

    # Only two calls total, but the timestamp gap alone exceeds confirm_duration.
    result = filt.update(True, 100.0)

    assert result.state == PhoneFilterState.CONFIRMED
    assert result.just_confirmed is True


def test_many_calls_within_a_short_timestamp_span_do_not_confirm() -> None:
    filt = PhoneTemporalFilter(make_config())
    filt.update(True, 0.0)

    # Many calls, but all timestamps stay well under confirm_duration_seconds.
    result = None
    for i in range(1, 50):
        result = filt.update(True, i * (CONFIRM / 100))

    assert result.state == PhoneFilterState.CONFIRMING
    assert result.just_confirmed is False


# --- 25. Timestamp propagation ---------------------------------------------------


def test_timestamp_propagates_into_result() -> None:
    filt = PhoneTemporalFilter(make_config())

    result = filt.update(True, 42.5)

    assert result.timestamp == 42.5


# --- 26. is_confirmed matches logical state --------------------------------------


@pytest.mark.parametrize(
    "state,expected_is_confirmed",
    [
        (PhoneFilterState.NOT_DETECTED, False),
        (PhoneFilterState.CONFIRMING, False),
        (PhoneFilterState.CONFIRMED, True),
        (PhoneFilterState.CLEARING, True),
    ],
)
def test_is_confirmed_matches_logical_state(state: PhoneFilterState, expected_is_confirmed: bool) -> None:
    filt = PhoneTemporalFilter(make_config())

    if state == PhoneFilterState.NOT_DETECTED:
        result = filt.update(False, 0.0)
    elif state == PhoneFilterState.CONFIRMING:
        result = filt.update(True, 0.0)
    elif state == PhoneFilterState.CONFIRMED:
        filt.update(True, 0.0)
        result = filt.update(True, CONFIRM)
    else:  # CLEARING
        filt.update(True, 0.0)
        filt.update(True, CONFIRM)
        result = filt.update(False, CONFIRM + 0.1)

    assert result.state == state
    assert result.is_confirmed is expected_is_confirmed


# --- 27. Multiple cycles do not leak stale state ---------------------------------


def test_multiple_complete_cycles_do_not_leak_stale_state() -> None:
    filt = PhoneTemporalFilter(make_config())
    cycle_start = 0.0

    for cycle in range(3):
        filt.update(True, cycle_start)
        confirmed = filt.update(True, ts(cycle_start, CONFIRM))
        assert confirmed.state == PhoneFilterState.CONFIRMED
        assert confirmed.just_confirmed is True

        clearing_start = ts(cycle_start, CONFIRM, 0.1)
        filt.update(False, clearing_start)
        cleared = filt.update(False, ts(clearing_start, CLEAR))
        assert cleared.state == PhoneFilterState.NOT_DETECTED
        assert cleared.just_cleared is True

        cycle_start = ts(clearing_start, CLEAR, 1.0)  # gap before next cycle

    assert filt.state == PhoneFilterState.NOT_DETECTED
