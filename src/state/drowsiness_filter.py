"""Eye-closure-duration drowsiness confirmation (PRD section 12).

Wraps the generic DurationConfirmer (src/state/temporal_filter.py, Phase 6)
rather than reimplementing confirm/clear state-machine logic - the only
thing this module adds is domain framing: "active" means
EyeState.CLOSED, and the confirmation threshold is
eyes.drowsiness_duration_seconds.

A normal blink (a closure well under drowsiness_duration_seconds) never
reaches confirmation and therefore never generates a DROWSINESS_SIGNAL,
which satisfies PRD section 12's blink-vs-drowsiness requirement without
any separate blink-specific logic: eyes.blink_max_duration_seconds exists
in configuration purely as a tuning/documentation value for what counts as
a normal blink (used elsewhere for eye-state classification) and is not
consumed here - nothing changes behaviorally at that boundary, only at
drowsiness_duration_seconds.

Clearing is immediate once eyes are no longer CLOSED (no grace period):
the PRD's `eyes:` config defines no separate clear/grace duration, the
same reasoning src/state/head_orientation_filter.py documents for head
orientation.

Missing/unreliable eye data (EyeState.UNKNOWN) is never treated as
CLOSED, per PRD section 10's "never classify missing landmarks as closed
eyes" - it is treated the same as OPEN (not active), so it can never by
itself start or continue a drowsiness confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.config_manager import EyesConfig
from src.face.eye_metrics import EyeState
from src.state.temporal_filter import DurationConfirmer, DurationConfirmerState


@dataclass(frozen=True)
class DrowsinessFilterResult:
    state: DurationConfirmerState
    is_drowsy: bool
    just_confirmed: bool
    just_cleared: bool
    timestamp: float


class DrowsinessFilter:
    """Timestamp-based confirmation for sustained eye closure."""

    def __init__(self, config: EyesConfig) -> None:
        self._confirmer = DurationConfirmer(
            confirm_duration_seconds=config.drowsiness_duration_seconds,
            clear_duration_seconds=0.0,
        )

    @property
    def state(self) -> DurationConfirmerState:
        return self._confirmer.state

    def update(self, eyes_state: EyeState, timestamp: float) -> DrowsinessFilterResult:
        active = eyes_state == EyeState.CLOSED
        result = self._confirmer.update(active, timestamp)
        return DrowsinessFilterResult(
            state=result.state,
            is_drowsy=result.is_confirmed,
            just_confirmed=result.just_confirmed,
            just_cleared=result.just_cleared,
            timestamp=result.timestamp,
        )
