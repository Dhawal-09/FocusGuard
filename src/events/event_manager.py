"""Event generation and bounded event log (PRD section 20).

EventManager owns only event generation/logging (PRD section 33): turning
already-decided transition edges - a temporal filter's just_confirmed /
just_cleared / just_diverted / just_restored, or a StateManager
StateTransition with changed=True - into immutable Event records, and
maintaining a bounded, timestamp-ordered in-memory log
(session.max_event_log_entries, PRD section 28). It never decides
*whether* a signal is confirmed (each temporal filter's job) or *what* the
current FocusState is (StateManager's job) - every public method here
corresponds to a single specific transition PRD section 20 defines, so it
cannot itself generate an event from single-frame noise.

Two distinct event categories, both required by the approved Phase 7
design:
  - Signal-level events: PHONE_DETECTED/CLEARED, DROWSINESS_SIGNAL/
    CLEARED, ATTENTION_DIVERTED/RESTORED, PERSON_LEFT/RETURNED - one per
    filter's confirm/clear edge.
  - State-level events: FOCUS_RESTORED, SESSION_STARTED/ENDED - one per
    StateManager/session-lifecycle transition.

Cooldown: PRD section 21 defines exactly one configured cooldown value -
phone.warning_cooldown_seconds - and section 36 requires "the same event
cannot repeatedly trigger audio/event every frame". Reusing the generic
Cooldown primitive (src/state/temporal_filter.py, Phase 6), PHONE_DETECTED
specifically is gated: a phone reappearing sooner than the configured
cooldown after the previous PHONE_DETECTED still updates the underlying
PhoneTemporalFilter/StateManager state normally - only the *event log
entry* (and, later, whatever audio trigger consumes it) is suppressed.
No other event type has a PRD-defined cooldown value, so none is invented
for them (PRD section 40 rule 7: no magic numbers, thresholds live in
config) - their "no repeat while unchanged" guarantee already comes for
free from only ever being called on a genuine transition edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.core.config_manager import PhoneConfig, SessionConfig
from src.state.temporal_filter import Cooldown


class EventType(Enum):
    SESSION_STARTED = "SESSION_STARTED"
    SESSION_ENDED = "SESSION_ENDED"

    PHONE_DETECTED = "PHONE_DETECTED"
    PHONE_CLEARED = "PHONE_CLEARED"

    DROWSINESS_SIGNAL = "DROWSINESS_SIGNAL"
    DROWSINESS_CLEARED = "DROWSINESS_CLEARED"

    ATTENTION_DIVERTED = "ATTENTION_DIVERTED"
    ATTENTION_RESTORED = "ATTENTION_RESTORED"

    PERSON_LEFT = "PERSON_LEFT"
    PERSON_RETURNED = "PERSON_RETURNED"

    FOCUS_RESTORED = "FOCUS_RESTORED"

    CAMERA_ERROR = "CAMERA_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    VISION_ERROR = "VISION_ERROR"


class Severity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


_SEVERITY_BY_EVENT_TYPE: dict[EventType, Severity] = {
    EventType.SESSION_STARTED: Severity.INFO,
    EventType.SESSION_ENDED: Severity.INFO,
    EventType.PHONE_DETECTED: Severity.WARNING,
    EventType.PHONE_CLEARED: Severity.INFO,
    EventType.DROWSINESS_SIGNAL: Severity.WARNING,
    EventType.DROWSINESS_CLEARED: Severity.INFO,
    EventType.ATTENTION_DIVERTED: Severity.WARNING,
    EventType.ATTENTION_RESTORED: Severity.INFO,
    EventType.PERSON_LEFT: Severity.WARNING,
    EventType.PERSON_RETURNED: Severity.INFO,
    EventType.FOCUS_RESTORED: Severity.INFO,
    EventType.CAMERA_ERROR: Severity.ERROR,
    EventType.MODEL_ERROR: Severity.ERROR,
    EventType.VISION_ERROR: Severity.ERROR,
}


@dataclass(frozen=True)
class Event:
    event_type: EventType
    timestamp: float
    severity: Severity
    metadata: dict[str, object] = field(default_factory=dict)


class EventManager:
    """Generates Event records from confirmed filter/state transitions and
    maintains a bounded, timestamp-ordered in-memory event log."""

    def __init__(self, session_config: SessionConfig, phone_config: PhoneConfig) -> None:
        if session_config.max_event_log_entries < 1:
            raise ValueError(
                "session.max_event_log_entries must be >= 1, got: "
                f"{session_config.max_event_log_entries}"
            )
        self._max_entries = session_config.max_event_log_entries
        self._log: list[Event] = []
        self._phone_cooldown = Cooldown(phone_config.warning_cooldown_seconds)

    @property
    def events(self) -> list[Event]:
        """The current bounded log, oldest first."""
        return list(self._log)

    def _append(self, event: Event) -> Event:
        self._log.append(event)
        if len(self._log) > self._max_entries:
            del self._log[: len(self._log) - self._max_entries]
        return event

    def _emit(
        self,
        event_type: EventType,
        timestamp: float,
        metadata: dict[str, object] | None = None,
    ) -> Event:
        event = Event(
            event_type=event_type,
            timestamp=timestamp,
            severity=_SEVERITY_BY_EVENT_TYPE[event_type],
            metadata=metadata or {},
        )
        return self._append(event)

    # --- Session lifecycle -------------------------------------------------

    def session_started(self, timestamp: float) -> Event:
        return self._emit(EventType.SESSION_STARTED, timestamp)

    def session_ended(self, timestamp: float) -> Event:
        return self._emit(EventType.SESSION_ENDED, timestamp)

    # --- Signal-level events (from temporal filter confirm/clear edges) ----

    def phone_confirmed(self, timestamp: float) -> Event | None:
        """Emit PHONE_DETECTED, subject to phone.warning_cooldown_seconds.
        Returns None (nothing logged) if the cooldown is still active."""
        if not self._phone_cooldown.try_fire(timestamp):
            return None
        return self._emit(EventType.PHONE_DETECTED, timestamp)

    def phone_cleared(self, timestamp: float) -> Event:
        return self._emit(EventType.PHONE_CLEARED, timestamp)

    def drowsiness_confirmed(self, timestamp: float) -> Event:
        return self._emit(EventType.DROWSINESS_SIGNAL, timestamp)

    def drowsiness_cleared(self, timestamp: float) -> Event:
        return self._emit(EventType.DROWSINESS_CLEARED, timestamp)

    def attention_diverted(self, timestamp: float) -> Event:
        return self._emit(EventType.ATTENTION_DIVERTED, timestamp)

    def attention_restored(self, timestamp: float) -> Event:
        return self._emit(EventType.ATTENTION_RESTORED, timestamp)

    def person_left(self, timestamp: float) -> Event:
        return self._emit(EventType.PERSON_LEFT, timestamp)

    def person_returned(self, timestamp: float) -> Event:
        return self._emit(EventType.PERSON_RETURNED, timestamp)

    # --- State-level events (from StateManager transitions) ----------------

    def focus_restored(self, timestamp: float) -> Event:
        return self._emit(EventType.FOCUS_RESTORED, timestamp)

    # --- Error events (PRD section 35; raised by later phases' error handling) --

    def camera_error(self, timestamp: float, message: str) -> Event:
        return self._emit(EventType.CAMERA_ERROR, timestamp, {"message": message})

    def model_error(self, timestamp: float, message: str) -> Event:
        return self._emit(EventType.MODEL_ERROR, timestamp, {"message": message})

    def vision_error(self, timestamp: float, message: str) -> Event:
        return self._emit(EventType.VISION_ERROR, timestamp, {"message": message})
