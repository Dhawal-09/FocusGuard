"""Tests for PersonAwayFilter (FOCUSGUARD_PRD.md section 14).

Fully deterministic: synthetic person_present/timestamp sequences only. No
webcam, GPU, or YOLO model.
"""

from __future__ import annotations

import math

import pytest

from src.core.config_manager import PersonConfig
from src.state.person_away_filter import PersonAwayFilter
from src.state.temporal_filter import DurationConfirmerState

AWAY = 3.0


def make_config(away_duration_seconds: float = AWAY) -> PersonConfig:
    return PersonConfig(away_duration_seconds=away_duration_seconds)


def ts(*parts: float) -> float:
    return round(math.fsum(parts), 9)


# --- Initial state -----------------------------------------------------------


def test_initial_state_is_inactive_and_not_away() -> None:
    filt = PersonAwayFilter(make_config())

    result = filt.update(True, 0.0)

    assert result.is_away is False
    assert filt.state in (DurationConfirmerState.INACTIVE, DurationConfirmerState.CONFIRMING)


# --- PRD section 36 / section 14: short absence -> no away ------------------


def test_short_absence_does_not_confirm_away() -> None:
    filt = PersonAwayFilter(make_config())
    filt.update(True, 0.0)

    filt.update(False, 0.1)
    result = filt.update(True, ts(0.1, AWAY - 1.0))  # returns well before the away threshold

    assert result.is_away is False
    assert result.just_confirmed is False


# --- PRD section 36 / section 14: long absence -> away ----------------------


def test_absence_exactly_at_away_threshold_confirms() -> None:
    filt = PersonAwayFilter(make_config())
    filt.update(True, 0.0)
    filt.update(False, 0.1)

    result = filt.update(False, ts(0.1, AWAY))

    assert result.is_away is True
    assert result.just_confirmed is True


def test_absence_above_away_threshold_confirms() -> None:
    filt = PersonAwayFilter(make_config())
    filt.update(False, 0.0)

    result = filt.update(False, AWAY + 1.0)

    assert result.is_away is True
    assert result.just_confirmed is True


def test_just_confirmed_fires_exactly_once() -> None:
    filt = PersonAwayFilter(make_config())
    filt.update(False, 0.0)

    confirmed = filt.update(False, AWAY)
    repeat = filt.update(False, AWAY + 0.5)

    assert confirmed.just_confirmed is True
    assert repeat.just_confirmed is False


# --- PRD section 36 / section 14: return -> person returned ------------------


def test_return_after_confirmed_away_clears_immediately_as_person_returned() -> None:
    filt = PersonAwayFilter(make_config())
    filt.update(False, 0.0)
    filt.update(False, AWAY)

    result = filt.update(True, ts(AWAY, 0.01))

    assert result.is_away is False
    assert result.just_cleared is True


def test_person_present_throughout_never_confirms_away() -> None:
    filt = PersonAwayFilter(make_config())

    for t in (0.0, 1.0, 5.0, 100.0):
        result = filt.update(True, t)
        assert result.is_away is False
        assert result.just_confirmed is False


# --- Rapid presence oscillation never confirms away --------------------------


def test_rapid_oscillation_shorter_than_away_duration_never_confirms() -> None:
    filt = PersonAwayFilter(make_config())
    t = 0.0
    step = AWAY / 8

    for _ in range(20):
        result = filt.update(False, t)
        assert result.is_away is False
        t = ts(t, step)
        result = filt.update(True, t)
        assert result.is_away is False
        t = ts(t, step)


# --- Timestamp validation ------------------------------------------------------


def test_out_of_order_timestamp_raises_value_error() -> None:
    filt = PersonAwayFilter(make_config())
    filt.update(False, 5.0)

    with pytest.raises(ValueError):
        filt.update(False, 4.0)


def test_timestamp_propagates_into_result() -> None:
    filt = PersonAwayFilter(make_config())

    result = filt.update(True, 42.5)

    assert result.timestamp == 42.5


# --- Zero-duration configuration ------------------------------------------------


def test_zero_away_duration_confirms_immediately() -> None:
    filt = PersonAwayFilter(make_config(away_duration_seconds=0.0))

    result = filt.update(False, 0.0)

    assert result.is_away is True
    assert result.just_confirmed is True
