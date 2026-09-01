"""Dashboard view-model contract for the Pygame UI (PRD section 22-24).

DashboardView is the single input contract UIManager.render() consumes -
plain data, no Pygame types, no dependency on SessionManager (Phase 11,
not yet built) or AudioManager (Phase 10, not yet built). Whatever
eventually assembles a DashboardView each frame (Phase 12 integration,
not this phase) supplies session/score/fps/inference values directly;
UIManager only formats and draws them (PRD section 33: "UIManager - only
presentation/input").

Kept in its own module (rather than inside ui_manager.py) precisely so
this contract - and the pure formatting helpers below - are importable
and unit-testable without importing pygame at all. UIAction (the result
of UIManager.poll_input(), PRD section 23) lives here for the same
reason.

Event timestamp display: Event.timestamp (src/events/event_manager.py) is
time.monotonic()-based (see CameraManager.Frame), which has no fixed
relationship to wall-clock time. format_event_timestamp() therefore
renders elapsed time since session start ("MM:SS"), never a fabricated
wall-clock time - this module does not modify Event or EventManager.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from src.core.types import VisionQuality
from src.detection.detection_types import Detection
from src.events.event_manager import Event
from src.face.eye_metrics import EyeState, LandmarkPoint
from src.face.head_pose import HeadOrientation
from src.state.state_manager import FocusState

MAX_EVENT_LOG_LINES = 8


class UIAction(Enum):
    """One user intent per PRD section 23's controls (plus the window
    close button, mapped to EXIT - standard Pygame-app behavior, not a
    PRD conflict). UIManager only recognizes and returns these - it never
    acts on them; starting/pausing a session, muting audio, and resetting
    are all a future caller's job (Phase 10/11/12), not UIManager's."""

    START_PAUSE_RESUME = "START_PAUSE_RESUME"
    EXIT = "EXIT"
    TOGGLE_MUTE = "TOGGLE_MUTE"
    TOGGLE_DEBUG = "TOGGLE_DEBUG"
    RESET = "RESET"


@dataclass(frozen=True)
class DebugInfo:
    """Optional debug-mode-only fields (PRD section 24). Only meaningful,
    and only ever rendered, when DashboardView.debug is True."""

    detections: tuple[Detection, ...] = ()
    landmarks: tuple[LandmarkPoint, ...] | None = None
    eye_metric: float | None = None
    head_yaw: float | None = None
    head_pitch: float | None = None
    vision_quality: VisionQuality | None = None
    phone_timer_seconds: float | None = None
    drowsiness_timer_seconds: float | None = None
    attention_timer_seconds: float | None = None
    away_timer_seconds: float | None = None


@dataclass(frozen=True)
class DashboardView:
    """Everything UIManager.render() needs to draw one frame (PRD section 22)."""

    status: FocusState
    person_present: bool
    phone_detected: bool
    eyes_state: EyeState
    head_orientation: HeadOrientation
    session_elapsed_seconds: float
    focus_score: int
    fps: float
    inference_latency_ms: float
    recent_events: tuple[Event, ...] = ()
    session_start_timestamp: float | None = None
    debug: bool = False
    debug_info: DebugInfo | None = None
    paused: bool = False
    """True while a session is active but paused (PRD section 23's
    SPACE control). FocusState itself has no PAUSED value - the state
    simply stops being re-evaluated while paused, so this is the only
    signal that tells the UI to show "PAUSED" instead of the frozen,
    now-stale status label (Phase 12 integration)."""


def format_duration(total_seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS. Negative input clamps to
    zero (a duration can never legitimately be negative)."""
    whole_seconds = max(0, int(total_seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_presence(present: bool) -> str:
    return "Detected" if present else "Not Detected"


_EYE_STATE_LABELS: dict[EyeState, str] = {
    EyeState.OPEN: "Open",
    EyeState.CLOSED: "Closed",
    EyeState.UNKNOWN: "Unknown",
}


def format_eye_state(state: EyeState) -> str:
    return _EYE_STATE_LABELS[state]


_HEAD_ORIENTATION_LABELS: dict[HeadOrientation, str] = {
    HeadOrientation.CENTER: "Center",
    HeadOrientation.LEFT: "Left",
    HeadOrientation.RIGHT: "Right",
    HeadOrientation.UP: "Up",
    HeadOrientation.DOWN: "Down",
    HeadOrientation.UNKNOWN: "Unknown",
}


def format_head_orientation(orientation: HeadOrientation) -> str:
    return _HEAD_ORIENTATION_LABELS[orientation]


def format_status(state: FocusState) -> str:
    return state.value.replace("_", " ")


_VISION_QUALITY_LABELS: dict[VisionQuality, str] = {
    VisionQuality.GOOD: "Good",
    VisionQuality.DEGRADED: "Degraded",
    VisionQuality.NO_PERSON: "No Person",
}


def format_vision_quality(vision_quality: VisionQuality) -> str:
    return _VISION_QUALITY_LABELS[vision_quality]


def format_confidence(confidence: float) -> str:
    return f"{confidence:.2f}"


def format_timer(seconds: float | None) -> str:
    """Format an optional debug confirmation/away timer. None (no
    transitional timer currently running - PRD section 24's timers are
    only meaningful mid-confirmation) renders as a placeholder, never a
    fabricated 0.00."""
    if seconds is None:
        return "--"
    return f"{seconds:.2f}s"


def format_event_type(event: Event) -> str:
    return event.event_type.value.replace("_", " ").title()


def format_event_timestamp(event_timestamp: float, session_start_timestamp: float | None) -> str:
    """Format an event's monotonic timestamp as elapsed time since session
    start ("MM:SS"). session_start_timestamp is None only when no session
    has started yet (should not normally happen for a real event, but is
    handled explicitly rather than raising)."""
    if session_start_timestamp is None:
        return "--:--"
    elapsed = max(0.0, event_timestamp - session_start_timestamp)
    minutes, seconds = divmod(int(elapsed), 60)
    return f"{minutes:02d}:{seconds:02d}"


def recent_events(events: Sequence[Event], limit: int = MAX_EVENT_LOG_LINES) -> tuple[Event, ...]:
    """Return at most the latest `limit` events, oldest-of-the-kept-set
    first - matches EventManager.events' own oldest-first ordering, so
    the on-screen log simply reads top-to-bottom as "earlier to later"."""
    if limit <= 0:
        return ()
    return tuple(events[-limit:])
