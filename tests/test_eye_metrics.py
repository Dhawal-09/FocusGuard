"""Tests for eye_metrics.py (FOCUSGUARD_PRD.md section 11).

All pure functions - no MediaPipe, webcam, or GPU required.
"""

from __future__ import annotations

import pytest

from src.face.eye_metrics import (
    EyeState,
    classify_eye_state,
    combine_eye_metrics,
    compute_ear,
)

CLOSED_THRESHOLD = 0.21
OPEN_THRESHOLD = 0.24

# Synthetic 6-point (p1..p6) sets in (corner, top, top, corner, bottom, bottom)
# order. Using symmetric offsets keeps the hand-computed EAR exact.
OPEN_EYE_POINTS = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.1), (0.3, 0.0), (0.2, -0.1), (0.1, -0.1)]
CLOSED_EYE_POINTS = [(0.0, 0.0), (0.1, 0.01), (0.2, 0.01), (0.3, 0.0), (0.2, -0.01), (0.1, -0.01)]
MID_ZONE_EYE_POINTS = [(0.0, 0.0), (0.1, 0.03375), (0.2, 0.03375), (0.3, 0.0), (0.2, -0.03375), (0.1, -0.03375)]
DEGENERATE_EYE_POINTS = [(0.15, 0.0), (0.1, 0.1), (0.2, 0.1), (0.15, 0.0), (0.2, -0.1), (0.1, -0.1)]


# --- compute_ear --------------------------------------------------------------


def test_compute_ear_open_eye() -> None:
    ear = compute_ear(OPEN_EYE_POINTS)

    assert ear == pytest.approx(0.6667, abs=1e-3)
    assert ear > OPEN_THRESHOLD


def test_compute_ear_closed_eye() -> None:
    ear = compute_ear(CLOSED_EYE_POINTS)

    assert ear == pytest.approx(0.0667, abs=1e-3)
    assert ear < CLOSED_THRESHOLD


def test_compute_ear_mid_zone() -> None:
    ear = compute_ear(MID_ZONE_EYE_POINTS)

    assert ear == pytest.approx(0.225, abs=1e-3)
    assert CLOSED_THRESHOLD < ear < OPEN_THRESHOLD


def test_compute_ear_returns_none_for_degenerate_geometry() -> None:
    ear = compute_ear(DEGENERATE_EYE_POINTS)

    assert ear is None


def test_compute_ear_requires_exactly_six_points() -> None:
    with pytest.raises(ValueError):
        compute_ear([(0.0, 0.0), (1.0, 1.0)])


# --- combine_eye_metrics -------------------------------------------------------


def test_combine_eye_metrics_averages_both_eyes() -> None:
    assert combine_eye_metrics(0.30, 0.20) == pytest.approx(0.25)


def test_combine_eye_metrics_falls_back_to_left_only() -> None:
    assert combine_eye_metrics(0.30, None) == pytest.approx(0.30)


def test_combine_eye_metrics_falls_back_to_right_only() -> None:
    assert combine_eye_metrics(None, 0.20) == pytest.approx(0.20)


def test_combine_eye_metrics_returns_none_when_neither_usable() -> None:
    assert combine_eye_metrics(None, None) is None


# --- classify_eye_state (hysteresis) -------------------------------------------


def test_classify_open_above_open_threshold() -> None:
    state = classify_eye_state(0.30, EyeState.UNKNOWN, CLOSED_THRESHOLD, OPEN_THRESHOLD)

    assert state == EyeState.OPEN


def test_classify_closed_below_closed_threshold() -> None:
    state = classify_eye_state(0.10, EyeState.OPEN, CLOSED_THRESHOLD, OPEN_THRESHOLD)

    assert state == EyeState.CLOSED


def test_classify_dead_zone_with_no_previous_state_defaults_to_open() -> None:
    state = classify_eye_state(0.225, EyeState.UNKNOWN, CLOSED_THRESHOLD, OPEN_THRESHOLD)

    assert state == EyeState.OPEN


def test_classify_dead_zone_retains_previous_open_state() -> None:
    state = classify_eye_state(0.225, EyeState.OPEN, CLOSED_THRESHOLD, OPEN_THRESHOLD)

    assert state == EyeState.OPEN


def test_classify_dead_zone_retains_previous_closed_state() -> None:
    state = classify_eye_state(0.225, EyeState.CLOSED, CLOSED_THRESHOLD, OPEN_THRESHOLD)

    assert state == EyeState.CLOSED


def test_classify_hysteresis_across_consecutive_frames() -> None:
    """Simulates a blink: OPEN -> dead zone (retains OPEN) -> CLOSED -> dead
    zone (retains CLOSED) -> OPEN."""
    state = EyeState.UNKNOWN

    state = classify_eye_state(0.30, state, CLOSED_THRESHOLD, OPEN_THRESHOLD)
    assert state == EyeState.OPEN

    state = classify_eye_state(0.225, state, CLOSED_THRESHOLD, OPEN_THRESHOLD)
    assert state == EyeState.OPEN

    state = classify_eye_state(0.10, state, CLOSED_THRESHOLD, OPEN_THRESHOLD)
    assert state == EyeState.CLOSED

    state = classify_eye_state(0.225, state, CLOSED_THRESHOLD, OPEN_THRESHOLD)
    assert state == EyeState.CLOSED

    state = classify_eye_state(0.30, state, CLOSED_THRESHOLD, OPEN_THRESHOLD)
    assert state == EyeState.OPEN


def test_classify_boundary_values_are_exclusive_dead_zone() -> None:
    # Exactly at closed_threshold is not "< closed_threshold" -> not CLOSED.
    # Exactly at open_threshold is not "> open_threshold" -> not OPEN.
    state_at_closed = classify_eye_state(CLOSED_THRESHOLD, EyeState.CLOSED, CLOSED_THRESHOLD, OPEN_THRESHOLD)
    state_at_open = classify_eye_state(OPEN_THRESHOLD, EyeState.OPEN, CLOSED_THRESHOLD, OPEN_THRESHOLD)

    assert state_at_closed == EyeState.CLOSED  # dead zone, retains previous
    assert state_at_open == EyeState.OPEN  # dead zone, retains previous
