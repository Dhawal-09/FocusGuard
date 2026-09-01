"""Pygame-mixer-based audio warnings and background music (PRD section 21).

AudioManager owns only audio (PRD section 33): loading warning sounds and
background music, playing them on request, applying volume/mute, and
gating the phone warning with its configured cooldown. It never decides
*when* a warning should play - it exposes granular play_*/music methods
and lets whatever calls them (Phase 12 integration, not this phase)
decide when a phone/drowsiness/attention/focus-restored/session-complete
moment has actually occurred. It never imports Event, EventType,
FocusState, or UIManager - the same module-independence discipline every
prior phase followed (e.g. PhoneTemporalFilter never imports
StateManager).

Missing or corrupt individual audio files must never crash the
application (PRD section 21/35): each sound/the music track is loaded
defensively at init() time, and a failed load simply means that specific
play_*/music call becomes a silent no-op forever after - it is never
retried and never raises. Only genuine mixer-*subsystem* initialization
failure (no audio device, driver unavailable, etc.) raises AudioError,
mirroring UIError's precedent for Pygame's display subsystem.

The mixer/music backends and the Sound loader are all injectable so this
class is fully unit-testable without real audio hardware or real audio
files - the same dependency-injection pattern CameraManager, YOLODetector,
and FaceAnalyzer already use for their own heavy external dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

import pygame

from src.core.config_manager import AudioConfig, PhoneConfig
from src.state.temporal_filter import Cooldown

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# PRD section 21's suggested asset layout. Not configurable - the PRD's
# `audio:` config schema (section 30) defines only enabled/volume/
# music_enabled/music_volume, no file-path keys.
DEFAULT_SOUND_PATHS: dict[str, Path] = {
    "phone_warning": PROJECT_ROOT / "assets" / "sounds" / "phone_warning.mp3",
    "drowsiness_warning": PROJECT_ROOT / "assets" / "sounds" / "drowsiness_warning.mp3",
    "attention_warning": PROJECT_ROOT / "assets" / "sounds" / "attention_warning.mp3",
    "focus_restored": PROJECT_ROOT / "assets" / "sounds" / "focus_restored.mp3",
    "session_complete": PROJECT_ROOT / "assets" / "sounds" / "session_complete.mp3",
}
DEFAULT_MUSIC_PATH = PROJECT_ROOT / "assets" / "music" / "focus_music.mp3"


class AudioError(Exception):
    """Raised only for genuine mixer-subsystem initialization failure
    (e.g. no audio device available) - never for a missing/corrupt
    individual sound or music file, which fail gracefully instead."""


class SoundLike(Protocol):
    """The subset of pygame.mixer.Sound's interface AudioManager depends on."""

    def play(self) -> object: ...

    def set_volume(self, volume: float) -> None: ...


class MixerBackend(Protocol):
    """The subset of the pygame.mixer module's interface AudioManager
    depends on for subsystem lifecycle."""

    def init(self) -> None: ...

    def quit(self) -> None: ...


class MusicBackend(Protocol):
    """The subset of the pygame.mixer.music module's interface AudioManager
    depends on for background music."""

    def load(self, path: str) -> None: ...

    def play(self, loops: int = 0) -> None: ...

    def pause(self) -> None: ...

    def unpause(self) -> None: ...

    def stop(self) -> None: ...

    def set_volume(self, volume: float) -> None: ...


SoundFactory = Callable[[str], SoundLike]


def _default_sound_factory(path: str) -> SoundLike:
    return pygame.mixer.Sound(path)


# Absorbs float64 subtraction noise at realistic time.monotonic() magnitudes,
# the same tolerance and reasoning src/state/temporal_filter.py documents for
# its own duration-boundary comparisons.
_REMINDER_EPSILON_SECONDS = 1e-9


class _PersistentReminder:
    """Tracks one continuously-active condition (e.g. "phone distraction is
    currently confirmed") and decides when a repeat reminder is due.

    The *initial* alert for a condition is a separate, pre-existing concern
    (play_phone_warning() etc., called by the caller on the filter's
    just_confirmed edge) - this class only handles repeats while the
    condition remains continuously true without ever clearing. On the
    first update(active=True, ...) call after being inactive, it records a
    baseline and returns False (never double-fires alongside the initial
    alert); on each later update(active=True, ...) call, once
    interval_seconds has elapsed since the last reminder (or the
    baseline), it returns True and re-baselines from that timestamp. Any
    update(active=False, ...) call resets it completely - the next onset
    starts a fresh interval, exactly as specified.

    Timestamp-based (not frame-count-based), monotonicity-validated, and
    epsilon-tolerant at the boundary - the same conventions every other
    temporal primitive in this codebase already follows (see
    src/state/temporal_filter.py).
    """

    def __init__(self, interval_seconds: float) -> None:
        if interval_seconds < 0:
            raise ValueError(f"interval_seconds must be >= 0, got {interval_seconds}")
        self._interval_seconds = interval_seconds
        self._last_timestamp: float | None = None
        self._last_reminder_timestamp: float | None = None

    def update(self, active: bool, now: float) -> bool:
        if self._last_timestamp is not None and now < self._last_timestamp:
            raise ValueError(f"Timestamps must be monotonic: received {now} after {self._last_timestamp}")
        self._last_timestamp = now

        if not active:
            self._last_reminder_timestamp = None
            return False

        if self._last_reminder_timestamp is None:
            self._last_reminder_timestamp = now
            return False

        elapsed = now - self._last_reminder_timestamp
        if elapsed >= self._interval_seconds - _REMINDER_EPSILON_SECONDS:
            self._last_reminder_timestamp = now
            return True
        return False

    def reset(self) -> None:
        """Discard in-progress tracking without needing an update() call -
        used when the caller knows monitoring itself paused/stopped (e.g.
        session pause), so a resumed condition never appears to have been
        continuously active across the gap."""
        self._last_reminder_timestamp = None


class AudioManager:
    """Loads warning sounds and background music, plays them on request,
    and applies mute/volume/cooldown - nothing else.

    All backends are injectable (defaulting to real pygame.mixer /
    pygame.mixer.music) so this class is fully unit-testable without a
    real audio device or real audio files.
    """

    def __init__(
        self,
        audio_config: AudioConfig,
        phone_config: PhoneConfig,
        *,
        mixer_backend: MixerBackend = pygame.mixer,
        music_backend: MusicBackend = pygame.mixer.music,
        sound_factory: SoundFactory = _default_sound_factory,
        sound_paths: dict[str, Path] | None = None,
        music_path: Path | None = None,
    ) -> None:
        self._audio_config = audio_config
        self._mixer_backend = mixer_backend
        self._music_backend = music_backend
        self._sound_factory = sound_factory
        self._sound_paths = sound_paths if sound_paths is not None else DEFAULT_SOUND_PATHS
        self._music_path = music_path if music_path is not None else DEFAULT_MUSIC_PATH

        self._phone_cooldown = Cooldown(phone_config.warning_cooldown_seconds)
        self._sounds: dict[str, SoundLike | None] = {}
        self._music_loaded = False
        self._muted = False
        self._is_initialized = False

        interval = audio_config.persistent_warning_interval_seconds
        self._phone_reminder = _PersistentReminder(interval)
        self._drowsiness_reminder = _PersistentReminder(interval)
        self._attention_reminder = _PersistentReminder(interval)

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def is_muted(self) -> bool:
        return self._muted

    def init(self) -> None:
        """Initialize the mixer subsystem and eagerly load every sound
        file. No-op if already initialized. Raises AudioError only for
        genuine mixer-subsystem failure - a missing/corrupt individual
        sound file is recorded as unavailable, not raised."""
        if self._is_initialized:
            return
        try:
            self._mixer_backend.init()
        except Exception as exc:
            raise AudioError(f"Failed to initialize audio mixer: {exc}") from exc
        self._is_initialized = True
        self._sounds = {key: self._load_sound(path) for key, path in self._sound_paths.items()}

    def shutdown(self) -> None:
        """Release mixer resources. Safe to call multiple times."""
        if self._is_initialized:
            try:
                self._music_backend.stop()
            except Exception:
                pass
            try:
                self._mixer_backend.quit()
            except Exception:
                pass
        self._sounds = {}
        self._music_loaded = False
        self._is_initialized = False

    def _load_sound(self, path: Path) -> SoundLike | None:
        try:
            sound = self._sound_factory(str(path))
            sound.set_volume(self._audio_config.volume)
        except Exception:
            return None
        return sound

    def _can_play(self) -> bool:
        return self._is_initialized and self._audio_config.enabled and not self._muted

    def _play_sound(self, key: str) -> bool:
        sound = self._sounds.get(key)
        if sound is None:
            return False
        try:
            sound.play()
        except Exception:
            return False
        return True

    # --- Warning sounds (PRD section 21) --------------------------------------

    def play_phone_warning(self, now: float) -> bool:
        """Play the phone-distraction warning, subject to
        phone.warning_cooldown_seconds (PRD: "audio warnings must have
        cooldowns"). Returns True only if actually played - False for any
        suppression reason (disabled, muted, not initialized, cooldown
        active, or the sound failed to load). Never raises."""
        if not self._can_play():
            return False
        if not self._phone_cooldown.try_fire(now):
            return False
        return self._play_sound("phone_warning")

    def play_drowsiness_warning(self) -> bool:
        if not self._can_play():
            return False
        return self._play_sound("drowsiness_warning")

    def play_attention_warning(self) -> bool:
        if not self._can_play():
            return False
        return self._play_sound("attention_warning")

    def play_focus_restored(self) -> bool:
        if not self._can_play():
            return False
        return self._play_sound("focus_restored")

    def play_session_complete(self) -> bool:
        if not self._can_play():
            return False
        return self._play_sound("session_complete")

    # --- Persistent condition reminders -----------------------------------------
    #
    # While a distraction condition (phone/drowsiness/attention) remains
    # continuously confirmed without ever clearing, repeat its warning every
    # audio.persistent_warning_interval_seconds - distinct from the initial
    # one-shot play_*_warning() call above (still triggered by the caller on
    # the filter's just_confirmed edge, unchanged). Call notify_*() every
    # frame with the condition's *current* confirmed/active state, not just
    # on edges - _PersistentReminder needs the continuous signal to measure
    # "has this stayed true long enough since the last reminder".
    #
    # The reminder *timer* always advances via _PersistentReminder.update(),
    # regardless of mute/enabled state - only the actual sound emission is
    # gated by _can_play(). This means muting never lets a backlog build up:
    # the cadence keeps progressing silently, and unmuting mid-cycle simply
    # resumes at the next naturally-due reminder rather than firing
    # immediately to "catch up".

    def _play_sound_if_allowed(self, key: str) -> bool:
        if not self._can_play():
            return False
        return self._play_sound(key)

    def notify_phone_distraction(self, active: bool, now: float) -> bool:
        """Call every frame with whether PHONE_DISTRACTION is currently
        confirmed. Returns True only if a repeat warning was actually
        played this call."""
        if not self._phone_reminder.update(active, now):
            return False
        return self._play_sound_if_allowed("phone_warning")

    def notify_drowsiness(self, active: bool, now: float) -> bool:
        """Call every frame with whether drowsiness is currently
        confirmed (DrowsinessFilter.is_drowsy)."""
        if not self._drowsiness_reminder.update(active, now):
            return False
        return self._play_sound_if_allowed("drowsiness_warning")

    def notify_attention_diverted(self, active: bool, now: float) -> bool:
        """Call every frame with whether attention is currently diverted
        (HeadOrientationFilter.is_diverted)."""
        if not self._attention_reminder.update(active, now):
            return False
        return self._play_sound_if_allowed("attention_warning")

    def reset_persistent_reminders(self) -> None:
        """Clear all persistent-reminder tracking - call whenever
        monitoring itself stops or restarts (session start, pause, end,
        reset) so a condition never appears to have "continued" across a
        gap it wasn't actually being observed through, and a new session
        never inherits a previous one's in-progress reminder cycle."""
        self._phone_reminder.reset()
        self._drowsiness_reminder.reset()
        self._attention_reminder.reset()

    # --- Background music (PRD section 21) ------------------------------------

    def _ensure_music_loaded(self) -> bool:
        if self._music_loaded:
            return True
        try:
            self._music_backend.load(str(self._music_path))
        except Exception:
            return False
        self._music_loaded = True
        return True

    def start_music(self) -> bool:
        """Start (looping) background focus music, if audio.enabled and
        audio.music_enabled are both true and not muted. Returns True only
        if playback actually started. Never raises."""
        if not self._is_initialized or not self._audio_config.enabled:
            return False
        if not self._audio_config.music_enabled or self._muted:
            return False
        if not self._ensure_music_loaded():
            return False
        try:
            self._music_backend.set_volume(self._audio_config.music_volume)
            self._music_backend.play(loops=-1)
        except Exception:
            return False
        return True

    def pause_music(self) -> None:
        if not self._is_initialized:
            return
        try:
            self._music_backend.pause()
        except Exception:
            pass

    def resume_music(self) -> None:
        if not self._is_initialized:
            return
        try:
            self._music_backend.unpause()
        except Exception:
            pass

    def stop_music(self) -> None:
        if not self._is_initialized:
            return
        try:
            self._music_backend.stop()
        except Exception:
            pass

    # --- Mute / volume ------------------------------------------------------

    def set_muted(self, muted: bool) -> None:
        """Set mute state. Immediately pauses/unpauses already-loaded
        background music to reflect the new state (mute must silence
        audio now, not merely block future play_* calls) - warning sounds
        need no equivalent action since they are short one-shots with
        nothing "in flight" to silence retroactively."""
        self._muted = muted
        if not self._is_initialized or not self._music_loaded:
            return
        try:
            if muted:
                self._music_backend.pause()
            else:
                self._music_backend.unpause()
        except Exception:
            pass

    def toggle_mute(self) -> bool:
        self.set_muted(not self._muted)
        return self._muted

    def set_volume(self, volume: float) -> None:
        """Update warning-sound volume at runtime, including already-loaded
        sounds. PRD defines no runtime volume-adjustment control (only `M`
        for mute exists in section 23) - this exists for completeness and
        testability, not to satisfy a specific control."""
        for sound in self._sounds.values():
            if sound is None:
                continue
            try:
                sound.set_volume(volume)
            except Exception:
                pass

    def set_music_volume(self, volume: float) -> None:
        if not self._is_initialized:
            return
        try:
            self._music_backend.set_volume(volume)
        except Exception:
            pass

    def __enter__(self) -> "AudioManager":
        self.init()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
