"""Session statistics, focus score, and JSON summary persistence
(PRD sections 25-27).

SessionManager owns only session statistics and persistence (PRD section
33): given the already-decided FocusState transitions and Events a future
caller feeds it (record_transition()/record_event() - Phase 12
integration, not this phase), it accumulates duration/streak/count
statistics and a demonstration focus score, and can serialize the result
to JSON. It never decides what the current FocusState is (StateManager's
job) or generates events (EventManager's job) - it only reacts to
already-decided data, and it never imports UIManager or AudioManager.

Push model (PRD section 34's main-loop step order - "...evaluate state ->
generate events -> send events to audio -> update session -> render
UI"): record_transition() is called every time the state machine is
evaluated (even when nothing changed - most calls carry no state change
but still carry real elapsed time that must be counted), and
record_event() is called for each Event EventManager actually emits.

Duration/streak accounting integrates elapsed time between successive
record_transition() calls, crediting whichever state was active during
that just-elapsed interval - not the transition's new state. This keeps
memory flat (no transition history is stored) regardless of session
length. Pausing/resuming excludes the paused wall-clock gap from every
duration total by resetting the accounting clock's origin on resume,
rather than naively integrating straight through the pause.

Score penalties (PRD section 26) apply to exactly four Event types
(PHONE_DETECTED, DROWSINESS_SIGNAL, ATTENTION_DIVERTED, PERSON_LEFT),
clamped at 0; every other Event type is still recorded in the session's
own event list (for the JSON summary) but affects neither count nor
score. record_transition()/record_event() defensively no-op while the
session is inactive or paused, rather than raising - a safety net
independent of caller discipline, per the approved Phase 10 decision.

Never stores webcam frames (PRD section 27) - SessionSummary and Event
never contain image data by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.core.config_manager import ScoreConfig
from src.events.event_manager import Event, EventType
from src.state.state_manager import FocusState, StateTransition

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS_DIRECTORY = PROJECT_ROOT / "logs"


class SessionError(Exception):
    """Raised for invalid session-lifecycle usage (e.g. ending a session
    that was never started) with a human-readable message."""


@dataclass(frozen=True)
class SessionSummary:
    """Everything PRD section 27's end-of-session analytics display needs,
    plus the full event list for JSON persistence."""

    started_at: datetime
    ended_at: datetime
    total_duration_seconds: float
    focused_duration_seconds: float
    phone_distraction_duration_seconds: float
    phone_distraction_count: int
    drowsiness_count: int
    attention_diversion_count: int
    away_count: int
    longest_focus_streak_seconds: float
    focus_score: int
    events: tuple[Event, ...]


def _event_to_dict(event: Event) -> dict[str, object]:
    return {
        "event_type": event.event_type.value,
        "timestamp": event.timestamp,
        "severity": event.severity.value,
        "metadata": event.metadata,
    }


def _summary_to_dict(summary: SessionSummary) -> dict[str, object]:
    return {
        "started_at": summary.started_at.isoformat(),
        "ended_at": summary.ended_at.isoformat(),
        "total_duration_seconds": summary.total_duration_seconds,
        "focused_duration_seconds": summary.focused_duration_seconds,
        "phone_distraction_duration_seconds": summary.phone_distraction_duration_seconds,
        "phone_distraction_count": summary.phone_distraction_count,
        "drowsiness_count": summary.drowsiness_count,
        "attention_diversion_count": summary.attention_diversion_count,
        "away_count": summary.away_count,
        "longest_focus_streak_seconds": summary.longest_focus_streak_seconds,
        "focus_score": summary.focus_score,
        "events": [_event_to_dict(event) for event in summary.events],
    }


class SessionManager:
    """Accumulates session duration/streak/count statistics and a
    demonstration focus score from incrementally-fed FocusState
    transitions and Events."""

    def __init__(self, score_config: ScoreConfig) -> None:
        self._score_config = score_config
        self.reset()

    # --- Lifecycle -------------------------------------------------------------

    def reset(self) -> None:
        """Unconditionally clear all session state back to a blank slate.
        PRD section 23's "if safe" qualifier on the Reset control is a
        future caller/UI policy decision, not something SessionManager
        enforces itself."""
        self._is_active = False
        self._is_paused = False
        self._start_timestamp: float | None = None
        self._start_wall_clock: datetime | None = None
        self._last_timestamp: float | None = None
        self._last_state = FocusState.IDLE

        self._focus_score = self._score_config.starting_score
        self._focused_duration = 0.0
        self._phone_distraction_duration = 0.0
        self._current_focus_streak = 0.0
        self._longest_focus_streak = 0.0

        self._phone_distraction_count = 0
        self._drowsiness_count = 0
        self._attention_diversion_count = 0
        self._away_count = 0

        self._events: list[Event] = []

    def start_session(self, timestamp: float) -> None:
        """Start a new session. Idempotent - a no-op while already active,
        matching CameraManager/UIManager/AudioManager's "safe to call"
        precedent."""
        if self._is_active:
            return
        self.reset()
        self._is_active = True
        self._start_timestamp = timestamp
        self._start_wall_clock = datetime.now()
        self._last_timestamp = timestamp

    def pause_session(self, timestamp: float) -> None:
        """Pause. No-op if not active or already paused."""
        if not self._is_active or self._is_paused:
            return
        self._validate_timestamp(timestamp)
        self._accumulate(timestamp)
        self._is_paused = True

    def resume_session(self, timestamp: float) -> None:
        """Resume. No-op if not active or not paused. Resets the
        accounting clock's origin to `timestamp` so the paused wall-clock
        gap is excluded from every duration total."""
        if not self._is_active or not self._is_paused:
            return
        self._validate_timestamp(timestamp)
        self._last_timestamp = timestamp
        self._is_paused = False

    def end_session(self, timestamp: float) -> SessionSummary:
        """Finalize accounting and return a SessionSummary. Raises
        SessionError if no session is currently active - unlike
        record_transition()/record_event(), there is no sensible "no-op"
        return value for a summary that does not exist."""
        if not self._is_active:
            raise SessionError("Cannot end a session that has not been started.")
        self._validate_timestamp(timestamp)
        if not self._is_paused:
            self._accumulate(timestamp)

        assert self._start_timestamp is not None
        assert self._start_wall_clock is not None
        summary = SessionSummary(
            started_at=self._start_wall_clock,
            ended_at=datetime.now(),
            total_duration_seconds=timestamp - self._start_timestamp,
            focused_duration_seconds=self._focused_duration,
            phone_distraction_duration_seconds=self._phone_distraction_duration,
            phone_distraction_count=self._phone_distraction_count,
            drowsiness_count=self._drowsiness_count,
            attention_diversion_count=self._attention_diversion_count,
            away_count=self._away_count,
            longest_focus_streak_seconds=self._longest_focus_streak,
            focus_score=self._focus_score,
            events=tuple(self._events),
        )

        self._is_active = False
        self._is_paused = False
        return summary

    # --- Incremental recording ---------------------------------------------------

    def record_transition(self, transition: StateTransition) -> None:
        """Feed one StateManager.evaluate() result. Called every evaluated
        frame, not just on changed=True edges - most calls carry no state
        change but still carry real elapsed time that must be counted.
        Defensively no-ops while inactive or paused."""
        if not self._is_active or self._is_paused:
            return
        self._validate_timestamp(transition.timestamp)
        self._accumulate(transition.timestamp)
        self._last_state = transition.state

    def record_event(self, event: Event) -> None:
        """Feed one Event EventManager emitted. Defensively no-ops while
        inactive or paused."""
        if not self._is_active or self._is_paused:
            return
        self._validate_timestamp(event.timestamp)
        self._events.append(event)

        if event.event_type == EventType.PHONE_DETECTED:
            self._phone_distraction_count += 1
            self._apply_penalty(self._score_config.phone_event_penalty)
        elif event.event_type == EventType.DROWSINESS_SIGNAL:
            self._drowsiness_count += 1
            self._apply_penalty(self._score_config.drowsiness_event_penalty)
        elif event.event_type == EventType.ATTENTION_DIVERTED:
            self._attention_diversion_count += 1
            self._apply_penalty(self._score_config.attention_event_penalty)
        elif event.event_type == EventType.PERSON_LEFT:
            self._away_count += 1
            self._apply_penalty(self._score_config.away_event_penalty)
        # every other event type: recorded above, but no count/penalty.

    def _accumulate(self, timestamp: float) -> None:
        """Credit the interval [self._last_timestamp, timestamp] to
        whichever state was active during it (self._last_state - the
        state as of the *previous* call, not the new one)."""
        assert self._last_timestamp is not None
        elapsed = timestamp - self._last_timestamp

        if self._last_state == FocusState.FOCUSED:
            self._focused_duration += elapsed
            self._current_focus_streak += elapsed
            self._longest_focus_streak = max(self._longest_focus_streak, self._current_focus_streak)
        else:
            self._current_focus_streak = 0.0
            if self._last_state == FocusState.PHONE_DISTRACTION:
                self._phone_distraction_duration += elapsed

        self._last_timestamp = timestamp

    def _apply_penalty(self, penalty: int) -> None:
        self._focus_score = max(0, self._focus_score - penalty)

    def _validate_timestamp(self, timestamp: float) -> None:
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError(f"Timestamps must be monotonic: received {timestamp} after {self._last_timestamp}")

    # --- Live read-only state -----------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def focus_score(self) -> int:
        return self._focus_score

    @property
    def focused_duration_seconds(self) -> float:
        return self._focused_duration

    @property
    def phone_distraction_duration_seconds(self) -> float:
        return self._phone_distraction_duration

    @property
    def longest_focus_streak_seconds(self) -> float:
        return self._longest_focus_streak

    @property
    def phone_distraction_count(self) -> int:
        return self._phone_distraction_count

    @property
    def drowsiness_count(self) -> int:
        return self._drowsiness_count

    @property
    def attention_diversion_count(self) -> int:
        return self._attention_diversion_count

    @property
    def away_count(self) -> int:
        return self._away_count

    def elapsed_seconds(self, now: float) -> float:
        """Live session elapsed time for display (e.g.
        DashboardView.session_elapsed_seconds), independent of the
        internal per-state accounting buckets. 0.0 if not active."""
        if not self._is_active or self._start_timestamp is None:
            return 0.0
        return now - self._start_timestamp

    # --- Persistence (PRD section 27) ----------------------------------------------

    @staticmethod
    def save_summary_json(summary: SessionSummary, directory: Path | None = None) -> Path:
        """Persist a SessionSummary as logs/session_YYYYMMDD_HHMMSS.json,
        using the session's START wall-clock time for the filename.
        `directory` defaults to PROJECT_ROOT/logs but is overridable for
        testing. Never stores webcam frames - SessionSummary never
        contains image data by construction."""
        target_dir = directory if directory is not None else DEFAULT_LOGS_DIRECTORY
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"session_{summary.started_at.strftime('%Y%m%d_%H%M%S')}.json"
        path = target_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_summary_to_dict(summary), handle, indent=2)
        return path
