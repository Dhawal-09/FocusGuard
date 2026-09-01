"""Head-orientation temporal confirmation (PRD section 13).

Prevents a single noisy/transient off-center head-pose reading from
immediately being treated as a sustained attention diversion, using
head.confirmation_seconds. Restoring to CENTERED is immediate: the PRD's
`head:` config defines only one duration (confirmation_seconds), with no
separate clear/grace duration the way phone has confirm+clear - so this
filter does not invent an undocumented grace period.

Completely independent of FaceAnalyzer, head_pose.py's solvePnP internals,
CameraManager, YOLODetector, PhoneTemporalFilter, EventManager,
StateManager, AudioManager, and the generic (future) TemporalFilter.
Callers derive a HeadOrientation value (e.g. via
src.face.head_pose.estimate_head_pose) and feed it into update() along
with the frame's own timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.core.config_manager import HeadConfig
from src.face.head_pose import HeadOrientation

# Absorbs float64 subtraction noise at realistic time.monotonic() magnitudes
# (e.g. 100.8 - 100.0 == 0.7999999999999972, not exactly 0.8) without ever
# meaningfully affecting real frame timing, which operates in millisecond
# increments - orders of magnitude larger than this tolerance.
_BOUNDARY_EPSILON_SECONDS = 1e-9


class HeadOrientationFilterState(Enum):
    CENTERED = "CENTERED"
    DIVERTING = "DIVERTING"
    DIVERTED = "DIVERTED"


@dataclass(frozen=True)
class HeadOrientationFilterResult:
    state: HeadOrientationFilterState
    raw_orientation: HeadOrientation
    is_diverted: bool
    just_diverted: bool
    just_restored: bool
    timestamp: float


def _is_centered(orientation: HeadOrientation) -> bool:
    """UNKNOWN counts as centered for timer purposes: unreliable geometry
    must never count as evidence of diversion, mirroring the project-wide
    rule that missing/uncertain signal is never treated as the "worse"
    classification (e.g. missing eye landmarks are never CLOSED)."""
    return orientation in (HeadOrientation.CENTER, HeadOrientation.UNKNOWN)


class HeadOrientationFilter:
    """Timestamp-based confirmation for sustained off-center head orientation.

    CENTERED -> DIVERTING -> DIVERTED. DIVERTING or DIVERTED -> CENTERED
    immediately whenever the raw orientation returns to CENTER or UNKNOWN
    (no grace period). Switching between off-center directions (e.g. LEFT
    to RIGHT) without passing through CENTER does NOT reset the
    confirmation timer or clear a confirmed diversion, since PRD section
    17's ATTENTION_DIVERTED only cares about "outside the center
    threshold", not which specific direction.
    """

    def __init__(self, config: HeadConfig) -> None:
        self._config = config
        self._state = HeadOrientationFilterState.CENTERED
        self._diverting_start_timestamp: float | None = None
        self._last_timestamp: float | None = None

    @property
    def state(self) -> HeadOrientationFilterState:
        return self._state

    def elapsed_in_state_seconds(self, now: float) -> float | None:
        """Seconds elapsed since entering DIVERTING, for debug-mode timer
        display (PRD section 24). None while CENTERED or DIVERTED - no
        confirmation timer running in those states. Purely additive,
        read-only access to bookkeeping update() already maintains; it
        changes no existing behavior."""
        if self._diverting_start_timestamp is None:
            return None
        return now - self._diverting_start_timestamp

    def update(self, orientation: HeadOrientation, timestamp: float) -> HeadOrientationFilterResult:
        self._validate_timestamp(timestamp)

        just_diverted = False
        just_restored = False
        centered = _is_centered(orientation)

        if self._state == HeadOrientationFilterState.CENTERED and not centered:
            self._state = HeadOrientationFilterState.DIVERTING
            self._diverting_start_timestamp = timestamp

        if self._state == HeadOrientationFilterState.DIVERTING:
            if centered:
                self._state = HeadOrientationFilterState.CENTERED
                self._diverting_start_timestamp = None
            else:
                # Falls through from the transition above when
                # confirmation_seconds == 0: elapsed is 0, already >= 0,
                # so confirmation fires on this same call.
                elapsed = timestamp - self._diverting_start_timestamp
                if elapsed >= self._config.confirmation_seconds - _BOUNDARY_EPSILON_SECONDS:
                    self._state = HeadOrientationFilterState.DIVERTED
                    self._diverting_start_timestamp = None
                    just_diverted = True
        elif self._state == HeadOrientationFilterState.DIVERTED and centered:
            self._state = HeadOrientationFilterState.CENTERED
            just_restored = True

        self._last_timestamp = timestamp

        return HeadOrientationFilterResult(
            state=self._state,
            raw_orientation=orientation,
            is_diverted=self._state == HeadOrientationFilterState.DIVERTED,
            just_diverted=just_diverted,
            just_restored=just_restored,
            timestamp=timestamp,
        )

    def _validate_timestamp(self, timestamp: float) -> None:
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError(
                f"Timestamps must be monotonic: received {timestamp} after {self._last_timestamp}"
            )
