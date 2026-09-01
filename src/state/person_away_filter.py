"""Person-absence-duration AWAY confirmation (PRD section 14).

Wraps the generic DurationConfirmer (src/state/temporal_filter.py, Phase 6)
rather than reimplementing confirm/clear state-machine logic - the only
thing this module adds is domain framing: "active" means the primary
person is absent (person_present is False), and the confirmation
threshold is person.away_duration_seconds.

Clearing (PERSON_RETURNED) is immediate once the person is present again -
the PRD's `person:` config defines no separate clear/grace duration, the
same reasoning src/state/head_orientation_filter.py documents for head
orientation: a brief reappearance during the away-confirmation countdown
is handled by DurationConfirmer's own CONFIRMING -> INACTIVE transient
reset (never reaching AWAY at all), and a genuine return after AWAY has
been confirmed clears immediately rather than waiting out an artificial
grace period the PRD never specifies.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.config_manager import PersonConfig
from src.state.temporal_filter import DurationConfirmer, DurationConfirmerState


@dataclass(frozen=True)
class PersonAwayFilterResult:
    state: DurationConfirmerState
    is_away: bool
    just_confirmed: bool
    just_cleared: bool
    timestamp: float


class PersonAwayFilter:
    """Timestamp-based confirmation for sustained person absence."""

    def __init__(self, config: PersonConfig) -> None:
        self._confirmer = DurationConfirmer(
            confirm_duration_seconds=config.away_duration_seconds,
            clear_duration_seconds=0.0,
        )

    @property
    def state(self) -> DurationConfirmerState:
        return self._confirmer.state

    def elapsed_in_state_seconds(self, now: float) -> float | None:
        """Seconds elapsed since absence started confirming toward AWAY,
        for debug-mode away-timer display (PRD section 24). None while not
        currently mid-confirmation. Delegates directly to the wrapped
        DurationConfirmer - purely additive."""
        return self._confirmer.elapsed_in_state_seconds(now)

    def update(self, person_present: bool, timestamp: float) -> PersonAwayFilterResult:
        active = not person_present
        result = self._confirmer.update(active, timestamp)
        return PersonAwayFilterResult(
            state=result.state,
            is_away=result.is_confirmed,
            just_confirmed=result.just_confirmed,
            just_cleared=result.just_cleared,
            timestamp=result.timestamp,
        )
