"""Phone-detection temporal confirmation and clearing (PRD section 9).

Prevents a single noisy/transient YOLO cell-phone detection from
immediately becoming a confirmed phone distraction, and prevents a
single missed detection from immediately clearing a confirmed one.

Completely independent of CameraManager, YOLODetector, FaceAnalyzer,
EventManager, StateManager, AudioManager, and the generic (future)
TemporalFilter. Callers derive a plain `phone_detected` boolean from
YOLO's Detection list (`any(d.class_name == "cell phone" for d in
detections)`) and feed it into update() along with the frame's own
timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.core.config_manager import PhoneConfig


class PhoneFilterState(Enum):
    NOT_DETECTED = "NOT_DETECTED"
    CONFIRMING = "CONFIRMING"
    CONFIRMED = "CONFIRMED"
    CLEARING = "CLEARING"


@dataclass(frozen=True)
class PhoneFilterResult:
    state: PhoneFilterState
    is_confirmed: bool
    just_confirmed: bool
    just_cleared: bool
    timestamp: float


class PhoneTemporalFilter:
    """Timestamp-based confirm/clear state machine for phone detection.

    NOT_DETECTED -> CONFIRMING -> CONFIRMED -> CLEARING -> NOT_DETECTED,
    with CLEARING -> CONFIRMED if the phone reappears before
    clear_duration_seconds elapses. That special transition is what
    prevents CONFIRMED -> NOT_DETECTED -> CONFIRMING -> CONFIRMED
    flapping from a single missed detection: reappearing during the
    grace period returns straight to CONFIRMED without restarting
    confirmation from scratch.
    """

    def __init__(self, config: PhoneConfig) -> None:
        self._config = config
        self._state = PhoneFilterState.NOT_DETECTED
        self._state_start_timestamp: float | None = None
        self._last_timestamp: float | None = None

    @property
    def state(self) -> PhoneFilterState:
        return self._state

    def update(self, phone_detected: bool, timestamp: float) -> PhoneFilterResult:
        self._validate_timestamp(timestamp)

        just_confirmed = False
        just_cleared = False

        if self._state == PhoneFilterState.NOT_DETECTED and phone_detected:
            self._state = PhoneFilterState.CONFIRMING
            self._state_start_timestamp = timestamp

        if self._state == PhoneFilterState.CONFIRMING:
            if phone_detected:
                # Falls through from the transition above when
                # confirm_duration_seconds == 0: elapsed is 0, which is
                # already >= 0, so confirmation fires on this same call.
                elapsed = timestamp - self._state_start_timestamp
                if elapsed >= self._config.confirm_duration_seconds:
                    self._state = PhoneFilterState.CONFIRMED
                    self._state_start_timestamp = None
                    just_confirmed = True
            else:
                # Transient detection: never reached confirmation.
                self._state = PhoneFilterState.NOT_DETECTED
                self._state_start_timestamp = None
        elif self._state == PhoneFilterState.CONFIRMED and not phone_detected:
            self._state = PhoneFilterState.CLEARING
            self._state_start_timestamp = timestamp

        if self._state == PhoneFilterState.CLEARING:
            if phone_detected:
                # Grace-period reappearance: back to CONFIRMED directly,
                # no re-confirmation.
                self._state = PhoneFilterState.CONFIRMED
                self._state_start_timestamp = None
            else:
                # Falls through from the transition above when
                # clear_duration_seconds == 0: elapsed is 0, which is
                # already >= 0, so clearing fires on this same call.
                elapsed = timestamp - self._state_start_timestamp
                if elapsed >= self._config.clear_duration_seconds:
                    self._state = PhoneFilterState.NOT_DETECTED
                    self._state_start_timestamp = None
                    just_cleared = True

        self._last_timestamp = timestamp

        return PhoneFilterResult(
            state=self._state,
            is_confirmed=self._state in (PhoneFilterState.CONFIRMED, PhoneFilterState.CLEARING),
            just_confirmed=just_confirmed,
            just_cleared=just_cleared,
            timestamp=timestamp,
        )

    def _validate_timestamp(self, timestamp: float) -> None:
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError(
                f"Timestamps must be monotonic: received {timestamp} after {self._last_timestamp}"
            )
