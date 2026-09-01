"""Tests for AudioManager (FOCUSGUARD_PRD.md section 21, 33, 35, 37).

Fully deterministic: fake mixer/music backends and a fake Sound factory
are injected throughout, so no test here touches real pygame.mixer, a
real audio device, or a real audio file. A separate real-driver smoke
test (using a locally-generated, non-repository tone file) is performed
manually outside this suite; see the Phase 9 implementation report.
"""

from __future__ import annotations

import math

import pytest

from src.core.config_manager import AudioConfig, PhoneConfig
from src.audio.audio_manager import (
    DEFAULT_MUSIC_PATH,
    DEFAULT_SOUND_PATHS,
    AudioError,
    AudioManager,
    _PersistentReminder,
)

COOLDOWN = 10.0
REMINDER_INTERVAL = 10.0


def ts(*parts: float) -> float:
    return round(math.fsum(parts), 9)


def make_audio_config(**overrides) -> AudioConfig:
    defaults = dict(
        enabled=True,
        volume=0.7,
        music_enabled=True,
        music_volume=0.25,
        persistent_warning_interval_seconds=REMINDER_INTERVAL,
    )
    defaults.update(overrides)
    return AudioConfig(**defaults)


def make_phone_config(warning_cooldown_seconds: float = COOLDOWN) -> PhoneConfig:
    return PhoneConfig(
        confirm_duration_seconds=0.35,
        clear_duration_seconds=0.60,
        warning_cooldown_seconds=warning_cooldown_seconds,
    )


class FakeSound:
    def __init__(self, path: str) -> None:
        self.path = path
        self.played_count = 0
        self.volume: float | None = None

    def play(self) -> None:
        self.played_count += 1

    def set_volume(self, volume: float) -> None:
        self.volume = volume


class RaisingSound(FakeSound):
    def play(self) -> None:
        raise RuntimeError("playback failed")


class FakeMixerBackend:
    def __init__(self, fail_init: bool = False, fail_quit: bool = False) -> None:
        self.fail_init = fail_init
        self.fail_quit = fail_quit
        self.init_calls = 0
        self.quit_calls = 0

    def init(self) -> None:
        self.init_calls += 1
        if self.fail_init:
            raise RuntimeError("no audio device")

    def quit(self) -> None:
        self.quit_calls += 1
        if self.fail_quit:
            raise RuntimeError("quit failed")


class FakeMusicBackend:
    def __init__(self, fail_load: bool = False, fail_play: bool = False) -> None:
        self.fail_load = fail_load
        self.fail_play = fail_play
        self.load_calls: list[str] = []
        self.play_calls: list[int] = []
        self.paused = False
        self.stop_calls = 0
        self.volume: float | None = None

    def load(self, path: str) -> None:
        if self.fail_load:
            raise RuntimeError("bad music file")
        self.load_calls.append(path)

    def play(self, loops: int = 0) -> None:
        if self.fail_play:
            raise RuntimeError("play failed")
        self.play_calls.append(loops)

    def pause(self) -> None:
        self.paused = True

    def unpause(self) -> None:
        self.paused = False

    def stop(self) -> None:
        self.stop_calls += 1

    def set_volume(self, volume: float) -> None:
        self.volume = volume


def make_sound_factory(fail_keys: frozenset[str] = frozenset()):
    def factory(path: str) -> FakeSound:
        for key in fail_keys:
            if key in path:
                raise RuntimeError(f"missing file: {path}")
        return FakeSound(path)

    return factory


def make_manager(
    *,
    audio_config: AudioConfig | None = None,
    phone_config: PhoneConfig | None = None,
    mixer_backend: FakeMixerBackend | None = None,
    music_backend: FakeMusicBackend | None = None,
    fail_keys: frozenset[str] = frozenset(),
) -> tuple[AudioManager, FakeMixerBackend, FakeMusicBackend]:
    mixer = mixer_backend or FakeMixerBackend()
    music = music_backend or FakeMusicBackend()
    manager = AudioManager(
        audio_config or make_audio_config(),
        phone_config or make_phone_config(),
        mixer_backend=mixer,
        music_backend=music,
        sound_factory=make_sound_factory(fail_keys),
    )
    return manager, mixer, music


def sounds_of(manager: AudioManager) -> dict[str, FakeSound]:
    return manager._sounds  # type: ignore[attr-defined]


# --- Default asset paths (PRD section 21 suggested layout) --------------------------


def test_default_sound_paths_match_prd_suggested_layout() -> None:
    assert set(DEFAULT_SOUND_PATHS.keys()) == {
        "phone_warning",
        "drowsiness_warning",
        "attention_warning",
        "focus_restored",
        "session_complete",
    }
    for key, path in DEFAULT_SOUND_PATHS.items():
        assert path.parts[-3:] == ("assets", "sounds", f"{key}.mp3")


def test_default_music_path_matches_prd_suggested_layout() -> None:
    assert DEFAULT_MUSIC_PATH.parts[-3:] == ("assets", "music", "focus_music.mp3")


# --- Lifecycle ------------------------------------------------------------------------


def test_not_initialized_by_default() -> None:
    manager, _, _ = make_manager()
    assert manager.is_initialized is False


def test_init_sets_initialized_and_calls_mixer_init() -> None:
    manager, mixer, _ = make_manager()

    manager.init()

    assert manager.is_initialized is True
    assert mixer.init_calls == 1


def test_init_loads_every_configured_sound() -> None:
    manager, _, _ = make_manager()

    manager.init()

    loaded = sounds_of(manager)
    assert set(loaded.keys()) == set(DEFAULT_SOUND_PATHS.keys())
    assert all(sound is not None for sound in loaded.values())


def test_init_is_idempotent() -> None:
    manager, mixer, _ = make_manager()
    manager.init()

    manager.init()

    assert mixer.init_calls == 1  # not called again


def test_init_applies_configured_volume_to_loaded_sounds() -> None:
    manager, _, _ = make_manager(audio_config=make_audio_config(volume=0.42))

    manager.init()

    for sound in sounds_of(manager).values():
        assert sound.volume == pytest.approx(0.42)


def test_init_failure_raises_audio_error() -> None:
    manager, _, _ = make_manager(mixer_backend=FakeMixerBackend(fail_init=True))

    with pytest.raises(AudioError):
        manager.init()

    assert manager.is_initialized is False


def test_shutdown_is_safe_to_call_multiple_times() -> None:
    manager, mixer, _ = make_manager()
    manager.init()

    manager.shutdown()
    manager.shutdown()

    assert manager.is_initialized is False
    assert mixer.quit_calls == 1  # second shutdown is a no-op (not already initialized)


def test_shutdown_without_init_does_not_raise() -> None:
    manager, _, _ = make_manager()
    manager.shutdown()
    assert manager.is_initialized is False


def test_shutdown_survives_backend_quit_failure() -> None:
    manager, _, _ = make_manager(mixer_backend=FakeMixerBackend(fail_quit=True))
    manager.init()

    manager.shutdown()  # must not raise

    assert manager.is_initialized is False


def test_context_manager_initializes_and_shuts_down() -> None:
    mixer = FakeMixerBackend()
    manager = AudioManager(make_audio_config(), make_phone_config(), mixer_backend=mixer, music_backend=FakeMusicBackend())

    with manager as m:
        assert m.is_initialized is True

    assert manager.is_initialized is False


# --- Missing/corrupt individual sound files never crash (PRD section 21/35) ---------


def test_missing_sound_file_is_recorded_as_unavailable_not_raised() -> None:
    manager, _, _ = make_manager(fail_keys=frozenset({"phone_warning"}))

    manager.init()  # must not raise

    assert sounds_of(manager)["phone_warning"] is None
    assert sounds_of(manager)["drowsiness_warning"] is not None


def test_play_warning_with_missing_file_returns_false_and_does_not_raise() -> None:
    manager, _, _ = make_manager(fail_keys=frozenset({"phone_warning"}))
    manager.init()

    result = manager.play_phone_warning(0.0)

    assert result is False


def test_play_sound_that_raises_on_play_returns_false() -> None:
    manager, mixer, music = make_manager()
    manager.init()
    sounds_of(manager)["drowsiness_warning"] = RaisingSound("x")

    result = manager.play_drowsiness_warning()

    assert result is False


# --- Playback: each warning method ----------------------------------------------------


def test_play_phone_warning_succeeds_when_ready() -> None:
    manager, _, _ = make_manager()
    manager.init()

    result = manager.play_phone_warning(0.0)

    assert result is True
    assert sounds_of(manager)["phone_warning"].played_count == 1


def test_play_drowsiness_warning_succeeds_when_ready() -> None:
    manager, _, _ = make_manager()
    manager.init()

    result = manager.play_drowsiness_warning()

    assert result is True
    assert sounds_of(manager)["drowsiness_warning"].played_count == 1


def test_play_attention_warning_succeeds_when_ready() -> None:
    manager, _, _ = make_manager()
    manager.init()

    result = manager.play_attention_warning()

    assert result is True
    assert sounds_of(manager)["attention_warning"].played_count == 1


def test_play_focus_restored_succeeds_when_ready() -> None:
    manager, _, _ = make_manager()
    manager.init()

    result = manager.play_focus_restored()

    assert result is True
    assert sounds_of(manager)["focus_restored"].played_count == 1


def test_play_session_complete_succeeds_when_ready() -> None:
    manager, _, _ = make_manager()
    manager.init()

    result = manager.play_session_complete()

    assert result is True
    assert sounds_of(manager)["session_complete"].played_count == 1


@pytest.mark.parametrize(
    "method_name,needs_now",
    [
        ("play_phone_warning", True),
        ("play_drowsiness_warning", False),
        ("play_attention_warning", False),
        ("play_focus_restored", False),
        ("play_session_complete", False),
    ],
)
def test_play_before_init_returns_false(method_name: str, needs_now: bool) -> None:
    manager, _, _ = make_manager()
    method = getattr(manager, method_name)

    result = method(0.0) if needs_now else method()

    assert result is False


@pytest.mark.parametrize(
    "method_name,needs_now",
    [
        ("play_phone_warning", True),
        ("play_drowsiness_warning", False),
        ("play_attention_warning", False),
        ("play_focus_restored", False),
        ("play_session_complete", False),
    ],
)
def test_play_when_disabled_returns_false(method_name: str, needs_now: bool) -> None:
    manager, _, _ = make_manager(audio_config=make_audio_config(enabled=False))
    manager.init()
    method = getattr(manager, method_name)

    result = method(0.0) if needs_now else method()

    assert result is False


@pytest.mark.parametrize(
    "method_name,needs_now",
    [
        ("play_phone_warning", True),
        ("play_drowsiness_warning", False),
        ("play_attention_warning", False),
        ("play_focus_restored", False),
        ("play_session_complete", False),
    ],
)
def test_play_when_muted_returns_false(method_name: str, needs_now: bool) -> None:
    manager, _, _ = make_manager()
    manager.init()
    manager.set_muted(True)
    method = getattr(manager, method_name)

    result = method(0.0) if needs_now else method()

    assert result is False


# --- Phone warning cooldown (PRD section 21/36: only phone has a cooldown) ---------


def test_second_phone_warning_within_cooldown_is_suppressed() -> None:
    manager, _, _ = make_manager()
    manager.init()
    manager.play_phone_warning(0.0)

    result = manager.play_phone_warning(ts(COOLDOWN - 1.0))

    assert result is False
    assert sounds_of(manager)["phone_warning"].played_count == 1


def test_phone_warning_exactly_at_cooldown_boundary_plays() -> None:
    manager, _, _ = make_manager()
    manager.init()
    manager.play_phone_warning(0.0)

    result = manager.play_phone_warning(COOLDOWN)

    assert result is True
    assert sounds_of(manager)["phone_warning"].played_count == 2


def test_phone_warning_after_cooldown_elapses_plays() -> None:
    manager, _, _ = make_manager()
    manager.init()
    manager.play_phone_warning(0.0)

    result = manager.play_phone_warning(ts(COOLDOWN, 5.0))

    assert result is True


def test_other_warnings_are_not_subject_to_phone_cooldown() -> None:
    manager, _, _ = make_manager()
    manager.init()

    manager.play_drowsiness_warning()
    manager.play_drowsiness_warning()
    manager.play_attention_warning()
    manager.play_attention_warning()

    assert sounds_of(manager)["drowsiness_warning"].played_count == 2
    assert sounds_of(manager)["attention_warning"].played_count == 2


def test_zero_cooldown_never_suppresses_phone_warning() -> None:
    manager, _, _ = make_manager(phone_config=make_phone_config(warning_cooldown_seconds=0.0))
    manager.init()

    first = manager.play_phone_warning(0.0)
    second = manager.play_phone_warning(0.0)

    assert first is True
    assert second is True


# --- Mute ------------------------------------------------------------------------------


def test_set_muted_true_then_false() -> None:
    manager, _, _ = make_manager()
    manager.init()

    manager.set_muted(True)
    assert manager.is_muted is True

    manager.set_muted(False)
    assert manager.is_muted is False


def test_toggle_mute_flips_state_and_returns_new_state() -> None:
    manager, _, _ = make_manager()
    manager.init()

    first = manager.toggle_mute()
    second = manager.toggle_mute()

    assert first is True
    assert second is False


def test_mute_blocks_start_music() -> None:
    manager, _, _ = make_manager()
    manager.init()
    manager.set_muted(True)

    result = manager.start_music()

    assert result is False


def test_muting_pauses_already_playing_music() -> None:
    manager, _, music = make_manager()
    manager.init()
    manager.start_music()
    assert music.paused is False

    manager.set_muted(True)

    assert music.paused is True


def test_unmuting_resumes_paused_music() -> None:
    manager, _, music = make_manager()
    manager.init()
    manager.start_music()
    manager.set_muted(True)

    manager.set_muted(False)

    assert music.paused is False


def test_mute_before_music_ever_started_does_not_touch_backend() -> None:
    manager, _, music = make_manager()
    manager.init()

    manager.set_muted(True)

    assert music.paused is False  # never touched - nothing was loaded/playing


def test_mute_before_init_does_not_raise() -> None:
    manager, _, _ = make_manager()

    manager.set_muted(True)  # must not raise

    assert manager.is_muted is True


# --- Volume ------------------------------------------------------------------------------


def test_set_volume_updates_all_loaded_sounds() -> None:
    manager, _, _ = make_manager()
    manager.init()

    manager.set_volume(0.33)

    for sound in sounds_of(manager).values():
        assert sound.volume == pytest.approx(0.33)


def test_set_volume_before_init_does_not_raise() -> None:
    manager, _, _ = make_manager()

    manager.set_volume(0.5)  # no sounds loaded yet - must not raise


def test_set_volume_skips_missing_sounds_without_raising() -> None:
    manager, _, _ = make_manager(fail_keys=frozenset({"phone_warning"}))
    manager.init()

    manager.set_volume(0.5)  # must not raise despite phone_warning being None

    assert sounds_of(manager)["drowsiness_warning"].volume == pytest.approx(0.5)


def test_set_music_volume_calls_backend() -> None:
    manager, _, music = make_manager()
    manager.init()

    manager.set_music_volume(0.6)

    assert music.volume == pytest.approx(0.6)


def test_set_music_volume_before_init_does_not_raise() -> None:
    manager, _, _ = make_manager()

    manager.set_music_volume(0.6)  # must not raise


# --- Music: audio.enabled / audio.music_enabled respected (PRD section 37) ---------


def test_start_music_succeeds_when_enabled_and_music_enabled() -> None:
    manager, _, music = make_manager()
    manager.init()

    result = manager.start_music()

    assert result is True
    assert music.play_calls == [-1]  # looping


def test_start_music_fails_when_audio_disabled() -> None:
    manager, _, _ = make_manager(audio_config=make_audio_config(enabled=False))
    manager.init()

    assert manager.start_music() is False


def test_start_music_fails_when_music_disabled() -> None:
    manager, _, _ = make_manager(audio_config=make_audio_config(music_enabled=False))
    manager.init()

    assert manager.start_music() is False


def test_start_music_fails_before_init() -> None:
    manager, _, _ = make_manager()

    assert manager.start_music() is False


def test_start_music_applies_configured_music_volume() -> None:
    manager, _, music = make_manager(audio_config=make_audio_config(music_volume=0.11))
    manager.init()

    manager.start_music()

    assert music.volume == pytest.approx(0.11)


def test_start_music_loads_file_only_once_across_multiple_starts() -> None:
    manager, _, music = make_manager()
    manager.init()

    manager.start_music()
    manager.stop_music()
    manager.start_music()

    assert len(music.load_calls) == 1


def test_start_music_with_missing_file_returns_false() -> None:
    manager, _, _ = make_manager(music_backend=FakeMusicBackend(fail_load=True))
    manager.init()

    assert manager.start_music() is False


def test_start_music_with_play_failure_returns_false() -> None:
    manager, _, _ = make_manager(music_backend=FakeMusicBackend(fail_play=True))
    manager.init()

    assert manager.start_music() is False


# --- Music controls ------------------------------------------------------------------------


def test_pause_music_calls_backend() -> None:
    manager, _, music = make_manager()
    manager.init()
    manager.start_music()

    manager.pause_music()

    assert music.paused is True


def test_resume_music_calls_backend() -> None:
    manager, _, music = make_manager()
    manager.init()
    manager.start_music()
    manager.pause_music()

    manager.resume_music()

    assert music.paused is False


def test_stop_music_calls_backend() -> None:
    manager, _, music = make_manager()
    manager.init()
    manager.start_music()

    manager.stop_music()

    assert music.stop_calls == 1


def test_music_controls_before_init_do_not_raise() -> None:
    manager, _, _ = make_manager()

    manager.pause_music()
    manager.resume_music()
    manager.stop_music()  # none of these should raise


def test_shutdown_stops_music() -> None:
    manager, _, music = make_manager()
    manager.init()
    manager.start_music()

    manager.shutdown()

    assert music.stop_calls == 1


# =========================================================================================
# _PersistentReminder (direct, isolated tests of the pure timer logic)
# =========================================================================================


def test_persistent_reminder_rejects_negative_interval() -> None:
    with pytest.raises(ValueError):
        _PersistentReminder(-1.0)


def test_persistent_reminder_first_onset_does_not_fire() -> None:
    reminder = _PersistentReminder(REMINDER_INTERVAL)

    assert reminder.update(True, 0.0) is False


def test_persistent_reminder_does_not_fire_before_interval_elapses() -> None:
    reminder = _PersistentReminder(REMINDER_INTERVAL)
    reminder.update(True, 0.0)

    result = reminder.update(True, REMINDER_INTERVAL - 1.0)

    assert result is False


def test_persistent_reminder_fires_exactly_at_interval_boundary() -> None:
    reminder = _PersistentReminder(REMINDER_INTERVAL)
    reminder.update(True, 0.0)

    result = reminder.update(True, REMINDER_INTERVAL)

    assert result is True


def test_persistent_reminder_fires_again_after_a_second_interval() -> None:
    reminder = _PersistentReminder(REMINDER_INTERVAL)
    reminder.update(True, 0.0)
    reminder.update(True, REMINDER_INTERVAL)

    result = reminder.update(True, REMINDER_INTERVAL * 2)

    assert result is True


def test_persistent_reminder_does_not_fire_again_immediately_after_firing() -> None:
    reminder = _PersistentReminder(REMINDER_INTERVAL)
    reminder.update(True, 0.0)
    reminder.update(True, REMINDER_INTERVAL)

    result = reminder.update(True, REMINDER_INTERVAL + 1.0)

    assert result is False


def test_persistent_reminder_condition_clearing_resets_it() -> None:
    reminder = _PersistentReminder(REMINDER_INTERVAL)
    reminder.update(True, 0.0)
    reminder.update(True, REMINDER_INTERVAL)  # fires

    cleared = reminder.update(False, REMINDER_INTERVAL + 0.1)
    assert cleared is False

    # A fresh onset right after clearing must NOT fire immediately - it's a
    # new cycle, not a continuation of the old one.
    fresh_onset = reminder.update(True, REMINDER_INTERVAL + 0.2)
    assert fresh_onset is False


def test_persistent_reminder_explicit_reset_forces_a_fresh_onset() -> None:
    reminder = _PersistentReminder(REMINDER_INTERVAL)
    reminder.update(True, 0.0)

    reminder.reset()

    # Without the reset, this timestamp would have fired (it's exactly the
    # interval boundary from the original onset). After reset, it must be
    # treated as a brand new onset instead.
    result = reminder.update(True, REMINDER_INTERVAL)
    assert result is False


def test_persistent_reminder_out_of_order_timestamp_raises_value_error() -> None:
    reminder = _PersistentReminder(REMINDER_INTERVAL)
    reminder.update(True, 5.0)

    with pytest.raises(ValueError):
        reminder.update(True, 4.0)


def test_persistent_reminder_zero_interval_fires_every_active_call_after_onset() -> None:
    reminder = _PersistentReminder(0.0)
    reminder.update(True, 0.0)  # onset, never fires

    assert reminder.update(True, 0.0) is True
    assert reminder.update(True, 0.0001) is True


# =========================================================================================
# AudioManager.notify_*() / reset_persistent_reminders() (PRD-adjacent feature)
# =========================================================================================


def test_notify_phone_distraction_first_onset_does_not_play() -> None:
    manager, _, _ = make_manager()
    manager.init()

    result = manager.notify_phone_distraction(True, 0.0)

    assert result is False
    assert sounds_of(manager)["phone_warning"].played_count == 0


def test_notify_phone_distraction_no_repeat_before_interval() -> None:
    manager, _, _ = make_manager()
    manager.init()
    manager.notify_phone_distraction(True, 0.0)

    result = manager.notify_phone_distraction(True, REMINDER_INTERVAL - 1.0)

    assert result is False
    assert sounds_of(manager)["phone_warning"].played_count == 0


def test_notify_phone_distraction_plays_at_interval_boundary() -> None:
    manager, _, _ = make_manager()
    manager.init()
    manager.notify_phone_distraction(True, 0.0)

    result = manager.notify_phone_distraction(True, REMINDER_INTERVAL)

    assert result is True
    assert sounds_of(manager)["phone_warning"].played_count == 1


def test_notify_phone_distraction_repeats_while_condition_persists() -> None:
    manager, _, _ = make_manager()
    manager.init()
    manager.notify_phone_distraction(True, 0.0)
    manager.notify_phone_distraction(True, REMINDER_INTERVAL)
    manager.notify_phone_distraction(True, REMINDER_INTERVAL * 2 - 1.0)  # too soon, no repeat

    manager.notify_phone_distraction(True, REMINDER_INTERVAL * 2)

    assert sounds_of(manager)["phone_warning"].played_count == 2


def test_notify_phone_distraction_resets_when_condition_clears() -> None:
    manager, _, _ = make_manager()
    manager.init()
    manager.notify_phone_distraction(True, 0.0)
    manager.notify_phone_distraction(True, REMINDER_INTERVAL)  # 1st reminder

    manager.notify_phone_distraction(False, REMINDER_INTERVAL + 1.0)  # cleared
    manager.notify_phone_distraction(True, REMINDER_INTERVAL + 2.0)  # fresh onset, no fire
    result = manager.notify_phone_distraction(True, REMINDER_INTERVAL + 2.0 + REMINDER_INTERVAL - 1.0)

    assert result is False  # would have fired only if the old cycle had continued
    assert sounds_of(manager)["phone_warning"].played_count == 1  # still just the one from before


def test_notify_drowsiness_and_notify_attention_diverted_follow_the_same_pattern() -> None:
    manager, _, _ = make_manager()
    manager.init()

    manager.notify_drowsiness(True, 0.0)
    drowsy_result = manager.notify_drowsiness(True, REMINDER_INTERVAL)
    manager.notify_attention_diverted(True, 0.0)
    attention_result = manager.notify_attention_diverted(True, REMINDER_INTERVAL)

    assert drowsy_result is True
    assert attention_result is True
    assert sounds_of(manager)["drowsiness_warning"].played_count == 1
    assert sounds_of(manager)["attention_warning"].played_count == 1


def test_phone_drowsiness_and_attention_reminders_have_independent_timers() -> None:
    manager, _, _ = make_manager()
    manager.init()

    manager.notify_phone_distraction(True, 0.0)
    manager.notify_drowsiness(True, 5.0)  # different onset time, independent timer

    phone_due = manager.notify_phone_distraction(True, REMINDER_INTERVAL)  # 10.0
    drowsy_not_due = manager.notify_drowsiness(True, REMINDER_INTERVAL)  # only 5s since its own onset

    assert phone_due is True
    assert drowsy_not_due is False
    assert sounds_of(manager)["phone_warning"].played_count == 1
    assert sounds_of(manager)["drowsiness_warning"].played_count == 0


def test_notify_while_muted_does_not_play_but_timer_still_advances() -> None:
    manager, _, _ = make_manager()
    manager.init()
    manager.notify_phone_distraction(True, 0.0)
    manager.set_muted(True)

    due_but_muted = manager.notify_phone_distraction(True, REMINDER_INTERVAL)
    assert due_but_muted is False
    assert sounds_of(manager)["phone_warning"].played_count == 0

    manager.set_muted(False)
    # The cadence must have progressed even while muted - the next natural
    # due time is REMINDER_INTERVAL after the muted trigger, not immediate.
    too_soon = manager.notify_phone_distraction(True, REMINDER_INTERVAL + 1.0)
    assert too_soon is False
    assert sounds_of(manager)["phone_warning"].played_count == 0

    right_on_time = manager.notify_phone_distraction(True, REMINDER_INTERVAL * 2)
    assert right_on_time is True
    assert sounds_of(manager)["phone_warning"].played_count == 1


def test_notify_when_disabled_never_plays() -> None:
    manager, _, _ = make_manager(audio_config=make_audio_config(enabled=False))
    manager.init()
    manager.notify_phone_distraction(True, 0.0)

    result = manager.notify_phone_distraction(True, REMINDER_INTERVAL)

    assert result is False


def test_notify_before_init_never_plays_and_does_not_raise() -> None:
    manager, _, _ = make_manager()
    manager.notify_phone_distraction(True, 0.0)

    result = manager.notify_phone_distraction(True, REMINDER_INTERVAL)  # must not raise

    assert result is False


def test_notify_with_missing_sound_file_returns_false_and_does_not_raise() -> None:
    manager, _, _ = make_manager(fail_keys=frozenset({"phone_warning"}))
    manager.init()
    manager.notify_phone_distraction(True, 0.0)

    result = manager.notify_phone_distraction(True, REMINDER_INTERVAL)  # must not raise

    assert result is False


def test_reset_persistent_reminders_clears_all_three_conditions() -> None:
    manager, _, _ = make_manager()
    manager.init()
    manager.notify_phone_distraction(True, 0.0)
    manager.notify_drowsiness(True, 0.0)
    manager.notify_attention_diverted(True, 0.0)

    manager.reset_persistent_reminders()

    # Without the reset, all three would fire at REMINDER_INTERVAL (their
    # shared onset). After reset, each must be treated as a fresh onset.
    assert manager.notify_phone_distraction(True, REMINDER_INTERVAL) is False
    assert manager.notify_drowsiness(True, REMINDER_INTERVAL) is False
    assert manager.notify_attention_diverted(True, REMINDER_INTERVAL) is False


def test_reset_persistent_reminders_before_init_does_not_raise() -> None:
    manager, _, _ = make_manager()

    manager.reset_persistent_reminders()  # must not raise


def test_existing_one_shot_play_phone_warning_unaffected_by_persistent_reminder_state() -> None:
    """The pre-existing one-shot method and its own cooldown must behave
    identically to before this feature existed - fully independent of the
    new notify_*()/reset_persistent_reminders() machinery."""
    manager, _, _ = make_manager()
    manager.init()

    manager.notify_phone_distraction(True, 0.0)  # onset only, does not play
    first = manager.play_phone_warning(0.0)
    second_within_cooldown = manager.play_phone_warning(COOLDOWN - 1.0)

    assert first is True
    assert second_within_cooldown is False
    assert sounds_of(manager)["phone_warning"].played_count == 1
