"""FocusGuard state machine and transition logic (PRD sections 17-19).

StateManager owns only state transitions (PRD section 33): given the
already temporally-filtered signals for the current frame (each produced
by a dedicated filter - PhoneTemporalFilter, DrowsinessFilter,
HeadOrientationFilter, PersonAwayFilter - plus the frame's VisionQuality),
it deterministically decides the current FocusState. It never inspects
raw per-frame detector/analyzer output itself, so it cannot flap on
single-frame noise by construction; that guarantee lives entirely in the
filters that produce its inputs.

Priority (PRD section 18), most to least urgent:
    AWAY > PHONE_DISTRACTION > DROWSINESS_SIGNAL > ATTENTION_DIVERTED
    > FOCUSED > UNKNOWN

UNKNOWN placement: per the approved Phase 7 design decision, UNKNOWN is
used only when a person IS present but required face-derived perception
is unavailable (VisionQuality.DEGRADED) - never merely because the person
is momentarily not visible. A missing person is handled by
PersonAwayFilter and surfaces as AWAY once confirmed; before that
confirmation fires (the away-duration grace period), the previous state
is held rather than flapping into UNKNOWN or AWAY on a still-unconfirmed
absence - this is what "no AWAY event if the person returns before the
threshold" (PRD section 14) means at the state-machine level: nothing
visibly changes yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.core.types import VisionQuality


class FocusState(Enum):
    IDLE = "IDLE"
    FOCUSED = "FOCUSED"
    PHONE_DISTRACTION = "PHONE_DISTRACTION"
    DROWSINESS_SIGNAL = "DROWSINESS_SIGNAL"
    ATTENTION_DIVERTED = "ATTENTION_DIVERTED"
    AWAY = "AWAY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class StateTransition:
    previous_state: FocusState
    state: FocusState
    changed: bool
    timestamp: float


class StateManager:
    """Deterministic priority-based focus-state evaluation.

    Starts in IDLE (PRD section 17: "No active session"). start_session()
    and end_session() are pure state-transition triggers (no timers, no
    duration/score tracking - that remains SessionManager's job in a later
    phase) that move between IDLE and the active-session subgraph
    evaluate() operates over.
    """

    def __init__(self) -> None:
        self._state = FocusState.IDLE

    @property
    def state(self) -> FocusState:
        return self._state

    def start_session(self, timestamp: float) -> StateTransition:
        """IDLE -> UNKNOWN (PRD section 19): no perception has been
        evaluated yet, so a session must never falsely assume FOCUSED on
        its very first frame. Idempotent while already active."""
        previous = self._state
        if previous == FocusState.IDLE:
            self._state = FocusState.UNKNOWN
        return self._transition(previous, timestamp)

    def end_session(self, timestamp: float) -> StateTransition:
        previous = self._state
        self._state = FocusState.IDLE
        return self._transition(previous, timestamp)

    def evaluate(
        self,
        *,
        is_away: bool,
        is_phone_distraction: bool,
        is_drowsy: bool,
        is_diverted: bool,
        vision_quality: VisionQuality,
        timestamp: float,
    ) -> StateTransition:
        """Evaluate one frame's already-filtered signals into a FocusState.

        A no-op while IDLE (no active session to evaluate): call
        start_session() first.
        """
        previous = self._state

        if previous == FocusState.IDLE:
            new_state = FocusState.IDLE
        elif is_away:
            new_state = FocusState.AWAY
        elif is_phone_distraction:
            new_state = FocusState.PHONE_DISTRACTION
        elif is_drowsy:
            new_state = FocusState.DROWSINESS_SIGNAL
        elif is_diverted:
            new_state = FocusState.ATTENTION_DIVERTED
        elif vision_quality == VisionQuality.DEGRADED:
            new_state = FocusState.UNKNOWN
        elif vision_quality == VisionQuality.NO_PERSON:
            # Person not currently visible but away-confirmation has not
            # fired yet (grace period): hold the previous state rather
            # than flapping into UNKNOWN or AWAY on an unconfirmed
            # absence.
            new_state = previous
        else:
            new_state = FocusState.FOCUSED

        self._state = new_state
        return StateTransition(
            previous_state=previous,
            state=new_state,
            changed=new_state != previous,
            timestamp=timestamp,
        )

    def _transition(self, previous: FocusState, timestamp: float) -> StateTransition:
        return StateTransition(
            previous_state=previous,
            state=self._state,
            changed=self._state != previous,
            timestamp=timestamp,
        )
