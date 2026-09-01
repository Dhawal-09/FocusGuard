"""Tests for temporal_filter.py (FOCUSGUARD_PRD.md section 15).

Fully deterministic: synthetic boolean/value/timestamp sequences only. No
webcam, GPU, MediaPipe, or real-time waiting. Includes a parity check
against eye_metrics.classify_eye_state to prove hysteresis() faithfully
generalizes its dead-zone semantics without modifying that file.
"""

from __future__ import annotations

import math

import pytest

from src.face.eye_metrics import EyeState, classify_eye_state
from src.state.temporal_filter import (
    Cooldown,
    Debouncer,
    DurationConfirmer,
    DurationConfirmerState,
    hysteresis,
)

CONFIRM = 0.35
CLEAR = 0.60


def ts(*parts: float) -> float:
    """Precise timestamp summation - avoids binary float drift at exact
    boundaries (see Phase 4/5 test lessons: chained `a + b + c` addition
    can land a hair below an exact threshold)."""
    return round(math.fsum(parts), 9)


# =============================================================================
# DurationConfirmer
# =============================================================================


def confirm(filt: DurationConfirmer, start_ts: float = 0.0) -> float:
    filt.update(True, start_ts)
    confirmed_ts = ts(start_ts, CONFIRM)
    result = filt.update(True, confirmed_ts)
    assert result.state == DurationConfirmerState.CONFIRMED
    return confirmed_ts


def test_confirmer_initial_state_is_inactive() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)

    assert filt.state == DurationConfirmerState.INACTIVE


def test_confirmer_first_true_enters_confirming() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)

    result = filt.update(True, 0.0)

    assert result.state == DurationConfirmerState.CONFIRMING
    assert result.is_confirmed is False


def test_confirmer_below_threshold_never_confirms_yet() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)
    filt.update(True, 0.0)

    result = filt.update(True, CONFIRM - 0.05)

    assert result.state == DurationConfirmerState.CONFIRMING


def test_confirmer_exactly_at_threshold_confirms() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)
    filt.update(True, 0.0)

    result = filt.update(True, CONFIRM)

    assert result.state == DurationConfirmerState.CONFIRMED
    assert result.just_confirmed is True


def test_confirmer_above_threshold_confirms() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)
    filt.update(True, 0.0)

    result = filt.update(True, CONFIRM + 0.1)

    assert result.just_confirmed is True


def test_confirmer_just_confirmed_fires_exactly_once() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)
    filt.update(True, 0.0)

    first = filt.update(True, CONFIRM)
    second = filt.update(True, ts(CONFIRM, 0.1))

    assert first.just_confirmed is True
    assert second.just_confirmed is False


def test_confirmer_transient_detection_never_confirms() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)
    filt.update(True, 0.0)

    result = filt.update(False, CONFIRM - 0.1)

    assert result.state == DurationConfirmerState.INACTIVE
    assert result.just_confirmed is False


def test_confirmer_confirmed_plus_absence_enters_clearing() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)
    confirmed_ts = confirm(filt)

    result = filt.update(False, ts(confirmed_ts, 0.1))

    assert result.state == DurationConfirmerState.CLEARING
    assert result.is_confirmed is True  # still logically confirmed during grace period


def test_confirmer_absence_below_clear_threshold_does_not_clear() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)
    confirmed_ts = confirm(filt)
    clearing_start = ts(confirmed_ts, 0.1)
    filt.update(False, clearing_start)

    result = filt.update(False, ts(clearing_start, CLEAR - 0.1))

    assert result.state == DurationConfirmerState.CLEARING
    assert result.just_cleared is False


def test_confirmer_absence_exactly_at_clear_threshold_clears() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)
    confirmed_ts = confirm(filt)
    clearing_start = ts(confirmed_ts, 0.1)
    filt.update(False, clearing_start)

    result = filt.update(False, ts(clearing_start, CLEAR))

    assert result.state == DurationConfirmerState.INACTIVE
    assert result.is_confirmed is False
    assert result.just_cleared is True


def test_confirmer_just_cleared_fires_exactly_once() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)
    confirmed_ts = confirm(filt)
    clearing_start = ts(confirmed_ts, 0.1)
    filt.update(False, clearing_start)

    first = filt.update(False, ts(clearing_start, CLEAR))
    second = filt.update(False, ts(clearing_start, CLEAR, 0.1))

    assert first.just_cleared is True
    assert second.just_cleared is False


def test_confirmer_reappears_during_clearing_returns_to_confirmed_without_restarting() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)
    confirmed_ts = confirm(filt)
    clearing_start = ts(confirmed_ts, 0.1)
    filt.update(False, clearing_start)

    result = filt.update(True, ts(clearing_start, 0.1))

    assert result.state == DurationConfirmerState.CONFIRMED
    assert result.just_confirmed is False


def test_confirmer_zero_confirm_duration_confirms_immediately() -> None:
    filt = DurationConfirmer(confirm_duration_seconds=0.0, clear_duration_seconds=CLEAR)

    result = filt.update(True, 0.0)

    assert result.state == DurationConfirmerState.CONFIRMED
    assert result.just_confirmed is True


def test_confirmer_zero_clear_duration_clears_immediately_matching_head_orientation_shape() -> None:
    """clear_duration_seconds=0.0 reproduces head_orientation_filter.py's
    'immediate restore, no grace period' behavior."""
    filt = DurationConfirmer(confirm_duration_seconds=CONFIRM, clear_duration_seconds=0.0)
    confirm(filt)

    result = filt.update(False, ts(CONFIRM, 0.1))

    assert result.state == DurationConfirmerState.INACTIVE
    assert result.just_cleared is True


def test_confirmer_duplicate_timestamps_valid_with_zero_elapsed() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)

    filt.update(True, 5.0)
    result = filt.update(True, 5.0)

    assert result.state == DurationConfirmerState.CONFIRMING
    assert result.just_confirmed is False


def test_confirmer_out_of_order_timestamp_raises_value_error() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)
    filt.update(True, 5.0)

    with pytest.raises(ValueError):
        filt.update(True, 4.0)


def test_confirmer_long_gap_uses_timestamp_not_call_count() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)
    filt.update(True, 0.0)

    result = filt.update(True, 100.0)

    assert result.state == DurationConfirmerState.CONFIRMED


def test_confirmer_boundary_holds_at_realistic_large_timestamp_magnitude() -> None:
    base = 100_000.0
    filt = DurationConfirmer(CONFIRM, CLEAR)
    filt.update(True, base)

    result = filt.update(True, base + CONFIRM)

    assert result.state == DurationConfirmerState.CONFIRMED
    assert result.just_confirmed is True


def test_confirmer_rapid_oscillation_never_confirms() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)
    t = 0.0
    step = CONFIRM / 4

    for _ in range(20):
        result = filt.update(True, t)
        assert result.state != DurationConfirmerState.CONFIRMED
        t = ts(t, step)
        filt.update(False, t)
        t = ts(t, step)


def test_confirmer_negative_confirm_duration_raises_value_error() -> None:
    with pytest.raises(ValueError):
        DurationConfirmer(confirm_duration_seconds=-1.0)


def test_confirmer_negative_clear_duration_raises_value_error() -> None:
    with pytest.raises(ValueError):
        DurationConfirmer(confirm_duration_seconds=CONFIRM, clear_duration_seconds=-1.0)


def test_confirmer_multiple_cycles_do_not_leak_stale_state() -> None:
    filt = DurationConfirmer(CONFIRM, CLEAR)

    for cycle_index in range(3):
        base = float(cycle_index * 100)
        filt.update(True, base)
        confirmed = filt.update(True, ts(base, CONFIRM))
        assert confirmed.state == DurationConfirmerState.CONFIRMED
        assert confirmed.just_confirmed is True

        clearing_start = ts(base, CONFIRM, 0.1)
        filt.update(False, clearing_start)
        cleared = filt.update(False, ts(clearing_start, CLEAR))
        assert cleared.state == DurationConfirmerState.INACTIVE
        assert cleared.just_cleared is True

    assert filt.state == DurationConfirmerState.INACTIVE


# =============================================================================
# hysteresis()
# =============================================================================

CLOSED_THRESHOLD = 0.21
OPEN_THRESHOLD = 0.24


def test_hysteresis_below_low_threshold_is_false() -> None:
    assert hysteresis(0.10, True, CLOSED_THRESHOLD, OPEN_THRESHOLD) is False


def test_hysteresis_above_high_threshold_is_true() -> None:
    assert hysteresis(0.30, False, CLOSED_THRESHOLD, OPEN_THRESHOLD) is True


def test_hysteresis_dead_zone_retains_previous_true() -> None:
    assert hysteresis(0.225, True, CLOSED_THRESHOLD, OPEN_THRESHOLD) is True


def test_hysteresis_dead_zone_retains_previous_false() -> None:
    assert hysteresis(0.225, False, CLOSED_THRESHOLD, OPEN_THRESHOLD) is False


def test_hysteresis_exactly_at_low_threshold_is_dead_zone() -> None:
    """Matches eye_metrics.classify_eye_state's strict '<' (not '<=')."""
    assert hysteresis(CLOSED_THRESHOLD, True, CLOSED_THRESHOLD, OPEN_THRESHOLD) is True
    assert hysteresis(CLOSED_THRESHOLD, False, CLOSED_THRESHOLD, OPEN_THRESHOLD) is False


def test_hysteresis_exactly_at_high_threshold_is_dead_zone() -> None:
    """Matches eye_metrics.classify_eye_state's strict '>' (not '>=')."""
    assert hysteresis(OPEN_THRESHOLD, True, CLOSED_THRESHOLD, OPEN_THRESHOLD) is True
    assert hysteresis(OPEN_THRESHOLD, False, CLOSED_THRESHOLD, OPEN_THRESHOLD) is False


def test_hysteresis_invalid_threshold_order_raises_value_error() -> None:
    with pytest.raises(ValueError):
        hysteresis(0.2, True, low_threshold=0.5, high_threshold=0.5)
    with pytest.raises(ValueError):
        hysteresis(0.2, True, low_threshold=0.6, high_threshold=0.5)


def test_hysteresis_across_consecutive_calls_simulates_blink() -> None:
    state = True  # OPEN
    state = hysteresis(0.30, state, CLOSED_THRESHOLD, OPEN_THRESHOLD)
    assert state is True
    state = hysteresis(0.225, state, CLOSED_THRESHOLD, OPEN_THRESHOLD)
    assert state is True
    state = hysteresis(0.10, state, CLOSED_THRESHOLD, OPEN_THRESHOLD)
    assert state is False
    state = hysteresis(0.225, state, CLOSED_THRESHOLD, OPEN_THRESHOLD)
    assert state is False
    state = hysteresis(0.30, state, CLOSED_THRESHOLD, OPEN_THRESHOLD)
    assert state is True


@pytest.mark.parametrize(
    "metric,previous_open",
    [
        (0.05, True), (0.05, False),
        (0.21, True), (0.21, False),
        (0.225, True), (0.225, False),
        (0.24, True), (0.24, False),
        (0.35, True), (0.35, False),
    ],
)
def test_hysteresis_matches_eye_metrics_classify_eye_state(metric: float, previous_open: bool) -> None:
    """Parity check: hysteresis() must faithfully reproduce
    eye_metrics.classify_eye_state's behavior for equivalent inputs,
    without eye_metrics.py being modified. UNKNOWN is never passed here
    since classify_eye_state's own UNKNOWN-previous-state handling
    (defaulting to OPEN) is a caller-level concern outside hysteresis()'s
    boolean-only contract."""
    previous_eye_state = EyeState.OPEN if previous_open else EyeState.CLOSED

    expected = classify_eye_state(metric, previous_eye_state, CLOSED_THRESHOLD, OPEN_THRESHOLD)
    actual = hysteresis(metric, previous_open, CLOSED_THRESHOLD, OPEN_THRESHOLD)

    assert actual is (expected == EyeState.OPEN)


# =============================================================================
# Debouncer
# =============================================================================

DEBOUNCE = 0.20


def test_debouncer_first_reading_is_accepted_immediately() -> None:
    deb = Debouncer(DEBOUNCE)

    result = deb.update(True, 0.0)

    assert result is True
    assert deb.value is True


def test_debouncer_rejects_change_within_debounce_window() -> None:
    deb = Debouncer(DEBOUNCE)
    deb.update(False, 0.0)

    result = deb.update(True, DEBOUNCE - 0.05)

    assert result is False  # still debounced to the original value


def test_debouncer_accepts_change_exactly_at_debounce_window() -> None:
    deb = Debouncer(DEBOUNCE)
    deb.update(False, 0.0)

    result = deb.update(True, DEBOUNCE)

    assert result is True


def test_debouncer_accepts_change_after_debounce_window() -> None:
    deb = Debouncer(DEBOUNCE)
    deb.update(False, 0.0)

    result = deb.update(True, DEBOUNCE + 0.1)

    assert result is True


def test_debouncer_rapid_chatter_collapses_to_single_accepted_change() -> None:
    deb = Debouncer(DEBOUNCE)
    deb.update(False, 0.0)

    # Multiple rapid flips within the window: all rejected until the window passes.
    assert deb.update(True, 0.02) is False
    assert deb.update(False, 0.05) is False
    assert deb.update(True, 0.08) is False
    assert deb.update(True, ts(0.0, DEBOUNCE)) is True


def test_debouncer_repeated_same_value_does_not_reset_window() -> None:
    deb = Debouncer(DEBOUNCE)
    deb.update(False, 0.0)
    deb.update(False, 0.05)  # same value, no change - should not affect timing

    result = deb.update(True, DEBOUNCE)  # measured from t=0.0, not t=0.05

    assert result is True


def test_debouncer_duplicate_timestamps_valid() -> None:
    deb = Debouncer(DEBOUNCE)
    deb.update(False, 5.0)

    result = deb.update(True, 5.0)

    assert result is False  # zero elapsed, well within debounce window


def test_debouncer_out_of_order_timestamp_raises_value_error() -> None:
    deb = Debouncer(DEBOUNCE)
    deb.update(False, 5.0)

    with pytest.raises(ValueError):
        deb.update(True, 4.0)


def test_debouncer_negative_duration_raises_value_error() -> None:
    with pytest.raises(ValueError):
        Debouncer(-0.1)


def test_debouncer_large_timestamp_gap_uses_timestamp_not_call_count() -> None:
    deb = Debouncer(DEBOUNCE)
    deb.update(False, 0.0)

    result = deb.update(True, 1000.0)

    assert result is True


def test_debouncer_value_is_none_before_first_update() -> None:
    deb = Debouncer(DEBOUNCE)

    assert deb.value is None


# =============================================================================
# Cooldown
# =============================================================================

COOLDOWN = 10.0


def test_cooldown_first_fire_always_succeeds() -> None:
    cd = Cooldown(COOLDOWN)

    assert cd.try_fire(0.0) is True
    assert cd.last_fired_timestamp == 0.0


def test_cooldown_second_fire_within_window_fails() -> None:
    cd = Cooldown(COOLDOWN)
    cd.try_fire(0.0)

    assert cd.try_fire(COOLDOWN - 1.0) is False


def test_cooldown_fire_exactly_at_window_succeeds() -> None:
    cd = Cooldown(COOLDOWN)
    cd.try_fire(0.0)

    assert cd.try_fire(COOLDOWN) is True


def test_cooldown_fire_after_window_succeeds() -> None:
    cd = Cooldown(COOLDOWN)
    cd.try_fire(0.0)

    assert cd.try_fire(COOLDOWN + 5.0) is True


def test_cooldown_repeated_attempts_within_window_all_fail_until_elapsed() -> None:
    cd = Cooldown(COOLDOWN)
    cd.try_fire(0.0)

    assert cd.try_fire(2.0) is False
    assert cd.try_fire(5.0) is False
    assert cd.try_fire(9.9) is False
    assert cd.try_fire(10.0) is True
    assert cd.try_fire(10.1) is False  # window restarted from the successful fire at 10.0


def test_cooldown_boundary_holds_at_realistic_large_timestamp_magnitude() -> None:
    base = 100_000.0
    cd = Cooldown(COOLDOWN)
    cd.try_fire(base)

    assert cd.try_fire(base + COOLDOWN) is True


def test_cooldown_duplicate_timestamps_valid() -> None:
    cd = Cooldown(COOLDOWN)
    cd.try_fire(5.0)

    assert cd.try_fire(5.0) is False


def test_cooldown_out_of_order_timestamp_raises_value_error() -> None:
    cd = Cooldown(COOLDOWN)
    cd.try_fire(5.0)

    with pytest.raises(ValueError):
        cd.try_fire(4.0)


def test_cooldown_negative_duration_raises_value_error() -> None:
    with pytest.raises(ValueError):
        Cooldown(-1.0)


def test_cooldown_zero_duration_always_succeeds() -> None:
    cd = Cooldown(0.0)

    assert cd.try_fire(0.0) is True
    assert cd.try_fire(0.0) is True
    assert cd.try_fire(0.001) is True


def test_cooldown_never_fired_before_first_call_reports_none() -> None:
    cd = Cooldown(COOLDOWN)

    assert cd.last_fired_timestamp is None


# =============================================================================
# DurationConfirmer.elapsed_in_state_seconds (Phase 8: debug-mode timers)
# =============================================================================


def test_elapsed_in_state_seconds_is_none_while_inactive() -> None:
    dc = DurationConfirmer(confirm_duration_seconds=CONFIRM)

    assert dc.elapsed_in_state_seconds(0.0) is None


def test_elapsed_in_state_seconds_while_confirming() -> None:
    dc = DurationConfirmer(confirm_duration_seconds=CONFIRM)
    dc.update(True, 0.0)

    assert dc.elapsed_in_state_seconds(0.1) == pytest.approx(0.1)


def test_elapsed_in_state_seconds_is_none_once_confirmed() -> None:
    dc = DurationConfirmer(confirm_duration_seconds=CONFIRM)
    dc.update(True, 0.0)
    dc.update(True, CONFIRM)

    assert dc.state == DurationConfirmerState.CONFIRMED
    assert dc.elapsed_in_state_seconds(CONFIRM + 5.0) is None


def test_elapsed_in_state_seconds_while_clearing() -> None:
    dc = DurationConfirmer(confirm_duration_seconds=CONFIRM, clear_duration_seconds=CLEAR)
    dc.update(True, 0.0)
    dc.update(True, CONFIRM)
    dc.update(False, ts(CONFIRM, 0.1))

    assert dc.state == DurationConfirmerState.CLEARING
    assert dc.elapsed_in_state_seconds(ts(CONFIRM, 0.1, 0.2)) == pytest.approx(0.2)


def test_elapsed_in_state_seconds_does_not_affect_update_behavior() -> None:
    """Purely additive: calling the accessor must not perturb the
    confirm/clear state machine itself."""
    dc = DurationConfirmer(confirm_duration_seconds=CONFIRM)
    dc.update(True, 0.0)

    dc.elapsed_in_state_seconds(0.1)
    dc.elapsed_in_state_seconds(0.2)
    result = dc.update(True, CONFIRM)

    assert result.state == DurationConfirmerState.CONFIRMED
    assert result.just_confirmed is True
