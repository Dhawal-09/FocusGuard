"""Reusable timestamp-based temporal stabilization (PRD section 15).

Three independent, composable primitives generalizing the temporal
patterns already proven by concrete phase-specific implementations:

  - DurationConfirmer: generalizes the confirm+clear-with-grace-period
    state machine first built for phone detection
    (src/state/phone_temporal_filter.py), and the confirm-with-
    immediate-restore shape built for head orientation
    (src/state/head_orientation_filter.py - equivalent here to
    clear_duration_seconds=0.0).
  - hysteresis(): generalizes the dead-zone classification first built
    for eye openness (src/face/eye_metrics.py's classify_eye_state).
  - Debouncer / Cooldown: new primitives (nothing to generalize from) -
    debounce suppresses rapid raw-signal chatter before confirmation
    logic ever sees it; cooldown throttles how often a side effect
    (e.g. an audio warning) may fire, independent of the underlying
    state's own behavior.

This module does NOT modify or replace phone_temporal_filter.py,
head_orientation_filter.py, or eye_metrics.py - those remain independent,
already-approved implementations. This is the generalized toolkit for
future consumers (drowsiness, attention-diversion, away-detection, audio
warning cooldowns, etc), not a refactor of existing ones.

All timing is timestamp-based, never frame-count-based, per PRD section
15's explicit requirement and every prior phase's convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Absorbs float64 subtraction noise at realistic time.monotonic()
# magnitudes (e.g. 100.8 - 100.0 == 0.7999999999999972, not exactly 0.8)
# without ever meaningfully affecting real frame timing, which operates in
# millisecond increments - orders of magnitude larger than this tolerance.
# (Same fix applied in src/state/head_orientation_filter.py.)
_BOUNDARY_EPSILON_SECONDS = 1e-9


def _validate_monotonic(last_timestamp: float | None, timestamp: float) -> None:
    if last_timestamp is not None and timestamp < last_timestamp:
        raise ValueError(f"Timestamps must be monotonic: received {timestamp} after {last_timestamp}")


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


# --- DurationConfirmer -------------------------------------------------------


class DurationConfirmerState(Enum):
    INACTIVE = "INACTIVE"
    CONFIRMING = "CONFIRMING"
    CONFIRMED = "CONFIRMED"
    CLEARING = "CLEARING"


@dataclass(frozen=True)
class DurationConfirmerResult:
    state: DurationConfirmerState
    is_confirmed: bool
    just_confirmed: bool
    just_cleared: bool
    timestamp: float


class DurationConfirmer:
    """Generic timestamp-based confirm/clear state machine.

    INACTIVE -> CONFIRMING -> CONFIRMED -> CLEARING -> INACTIVE, with
    CLEARING -> CONFIRMED if `active` becomes True again before
    clear_duration_seconds elapses (a grace-period reappearance never
    restarts confirmation from scratch - this is what prevents
    CONFIRMED -> INACTIVE -> CONFIRMING -> CONFIRMED flapping from a
    single missed reading).

    clear_duration_seconds=0.0 (the default) means clearing is immediate
    once `active` goes False - equivalent to head_orientation_filter.py's
    behavior. A positive value adds a grace period - equivalent to
    phone_temporal_filter.py's behavior.
    """

    def __init__(self, confirm_duration_seconds: float, clear_duration_seconds: float = 0.0) -> None:
        _validate_non_negative("confirm_duration_seconds", confirm_duration_seconds)
        _validate_non_negative("clear_duration_seconds", clear_duration_seconds)
        self._confirm_duration_seconds = confirm_duration_seconds
        self._clear_duration_seconds = clear_duration_seconds
        self._state = DurationConfirmerState.INACTIVE
        self._state_start_timestamp: float | None = None
        self._last_timestamp: float | None = None

    @property
    def state(self) -> DurationConfirmerState:
        return self._state

    def elapsed_in_state_seconds(self, now: float) -> float | None:
        """Seconds elapsed since entering the current transitional state
        (CONFIRMING or CLEARING), for debug-mode timer display (PRD
        section 24). None while INACTIVE or CONFIRMED - there is no
        transitional timer running in those states. Purely additive,
        read-only access to bookkeeping update() already maintains; it
        changes no existing behavior."""
        if self._state_start_timestamp is None:
            return None
        return now - self._state_start_timestamp

    def update(self, active: bool, timestamp: float) -> DurationConfirmerResult:
        _validate_monotonic(self._last_timestamp, timestamp)

        just_confirmed = False
        just_cleared = False

        if self._state == DurationConfirmerState.INACTIVE and active:
            self._state = DurationConfirmerState.CONFIRMING
            self._state_start_timestamp = timestamp

        if self._state == DurationConfirmerState.CONFIRMING:
            if active:
                # Falls through from the transition above when
                # confirm_duration_seconds == 0: elapsed is 0, already
                # >= 0, so confirmation fires on this same call.
                elapsed = timestamp - self._state_start_timestamp
                if elapsed >= self._confirm_duration_seconds - _BOUNDARY_EPSILON_SECONDS:
                    self._state = DurationConfirmerState.CONFIRMED
                    self._state_start_timestamp = None
                    just_confirmed = True
            else:
                # Transient: never reached confirmation.
                self._state = DurationConfirmerState.INACTIVE
                self._state_start_timestamp = None
        elif self._state == DurationConfirmerState.CONFIRMED and not active:
            self._state = DurationConfirmerState.CLEARING
            self._state_start_timestamp = timestamp

        if self._state == DurationConfirmerState.CLEARING:
            if active:
                # Grace-period reappearance: back to CONFIRMED directly.
                self._state = DurationConfirmerState.CONFIRMED
                self._state_start_timestamp = None
            else:
                elapsed = timestamp - self._state_start_timestamp
                if elapsed >= self._clear_duration_seconds - _BOUNDARY_EPSILON_SECONDS:
                    self._state = DurationConfirmerState.INACTIVE
                    self._state_start_timestamp = None
                    just_cleared = True

        self._last_timestamp = timestamp

        return DurationConfirmerResult(
            state=self._state,
            is_confirmed=self._state in (DurationConfirmerState.CONFIRMED, DurationConfirmerState.CLEARING),
            just_confirmed=just_confirmed,
            just_cleared=just_cleared,
            timestamp=timestamp,
        )


# --- hysteresis ---------------------------------------------------------------


def hysteresis(value: float, previous_state: bool, low_threshold: float, high_threshold: float) -> bool:
    """Pure dead-zone hysteresis classification.

    Generalizes eye_metrics.classify_eye_state's exact boundary semantics:
    value < low_threshold -> False; value > high_threshold -> True;
    values in the dead zone (inclusive of both exact threshold values)
    retain previous_state. previous_state is required explicitly (no
    default) - callers own what to assume before any valid prior reading
    exists, the same way FaceAnalyzer/eye_metrics resolve an UNKNOWN prior
    eye state to True (OPEN) before ever calling the underlying
    classifier.
    """
    if high_threshold <= low_threshold:
        raise ValueError(f"high_threshold ({high_threshold}) must be greater than low_threshold ({low_threshold})")

    if value < low_threshold:
        return False
    if value > high_threshold:
        return True
    return previous_state


# --- Debouncer ------------------------------------------------------------------


class Debouncer:
    """Suppresses rapid raw-signal transitions ("switch bounce").

    Distinct from DurationConfirmer: this filters the RAW input signal
    itself, before any confirmation logic sees it, by ignoring a
    transition unless at least debounce_seconds has elapsed since the
    last ACCEPTED transition. DurationConfirmer instead requires a
    signal to stay continuously active for a duration before treating it
    as confirmed - a different mechanism serving a different purpose.
    """

    def __init__(self, debounce_seconds: float) -> None:
        _validate_non_negative("debounce_seconds", debounce_seconds)
        self._debounce_seconds = debounce_seconds
        self._accepted_value: bool | None = None
        self._last_change_timestamp: float | None = None
        self._last_timestamp: float | None = None

    def update(self, value: bool, timestamp: float) -> bool:
        """Feed one raw reading; returns the current debounced value."""
        _validate_monotonic(self._last_timestamp, timestamp)
        self._last_timestamp = timestamp

        if self._accepted_value is None:
            self._accepted_value = value
            self._last_change_timestamp = timestamp
            return self._accepted_value

        if value != self._accepted_value:
            elapsed = timestamp - self._last_change_timestamp
            if elapsed >= self._debounce_seconds - _BOUNDARY_EPSILON_SECONDS:
                self._accepted_value = value
                self._last_change_timestamp = timestamp

        return self._accepted_value

    @property
    def value(self) -> bool | None:
        return self._accepted_value


# --- Cooldown ---------------------------------------------------------------------


class Cooldown:
    """Timestamp-based gate for throttling repeated event/action firing.

    Distinct from Debouncer (filters an input signal) and
    DurationConfirmer (confirms a sustained state): Cooldown gates how
    often a side effect (e.g. playing an audio warning) may fire,
    independent of how the underlying state behaves - directly serving
    "the same event cannot repeatedly trigger audio/event every frame".
    """

    def __init__(self, cooldown_seconds: float) -> None:
        _validate_non_negative("cooldown_seconds", cooldown_seconds)
        self._cooldown_seconds = cooldown_seconds
        self._last_fired_timestamp: float | None = None
        self._last_timestamp: float | None = None

    def try_fire(self, timestamp: float) -> bool:
        """Returns True (and starts the cooldown) if enough time has
        elapsed since the last successful fire, or if this is the first
        call; False otherwise."""
        _validate_monotonic(self._last_timestamp, timestamp)
        self._last_timestamp = timestamp

        if self._last_fired_timestamp is None:
            self._last_fired_timestamp = timestamp
            return True

        elapsed = timestamp - self._last_fired_timestamp
        if elapsed >= self._cooldown_seconds - _BOUNDARY_EPSILON_SECONDS:
            self._last_fired_timestamp = timestamp
            return True

        return False

    @property
    def last_fired_timestamp(self) -> float | None:
        return self._last_fired_timestamp
