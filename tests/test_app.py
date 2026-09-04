"""Tests for FocusGuardApp (FOCUSGUARD_PRD.md sections 5, 23, 34, 35).

Every manager FocusGuardApp composes is real, wired to fake hardware
backends via the exact same injection points each manager's own test
suite already established (CameraManager's capture_factory, YOLODetector's
model_factory, FaceAnalyzer's landmarker_factory, AudioManager's
mixer/music/sound backends, UIManager under SDL_VIDEODRIVER=dummy) - this
exercises the REAL orchestration code in src/core/app.py against fake
hardware, not a separately-mocked orchestrator. No webcam, GPU, real YOLO
weights, real MediaPipe model, or audio device is required. JSON summary
persistence always uses a pytest tmp_path, never the repository's real
logs/ directory.

Two testing strategies are used:
  - Direct calls to FocusGuardApp._process_frame() with hand-constructed
    Frame objects at exact chosen timestamps, for deterministic coverage
    of the CV pipeline -> filters -> state -> events -> session -> audio
    wiring (CameraManager.read_frame() uses real time.monotonic(), which
    is unsuitable for precise duration-threshold testing - every prior
    phase's tests avoid this the same way, with synthetic timestamps).
  - A handful of full run()/_main_loop() smoke tests using a fake camera
    that yields a fixed number of frames, for genuine startup/shutdown/
    plumbing coverage.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import math
from pathlib import Path

import numpy as np
import pygame
import pytest

from src.audio.audio_manager import AudioManager
from src.camera.camera_manager import CameraError, CameraManager, Frame
from src.core.app import FocusGuardApp
from src.core.config_manager import (
    AppConfig,
    AudioConfig,
    CameraConfig,
    EyesConfig,
    FaceConfig,
    HeadConfig,
    PersonConfig,
    PhoneConfig,
    ScoreConfig,
    SessionConfig,
    UIConfig,
    YoloConfig,
)
from src.detection.yolo_detector import CELL_PHONE_CLASS_NAME, DetectionError, PERSON_CLASS_NAME, YOLODetector
from src.events.event_manager import EventType
from src.face.eye_metrics import LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from src.face.face_analyzer import FaceAnalysisError, FaceAnalyzer
from src.state.state_manager import FocusState
from src.ui.dashboard_view import UIAction
from src.ui.ui_manager import UIError, UIManager

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 48

PERSON_BOX = ((5.0, 5.0, 55.0, 45.0), 0.9, 0)
PHONE_BOX = ((20.0, 15.0, 35.0, 25.0), 0.9, 1)  # inside PERSON_BOX, for association
YOLO_NAMES = {0: PERSON_CLASS_NAME, 1: CELL_PHONE_CLASS_NAME}

# Same synthetic eye points as test_eye_metrics.py / test_face_analyzer.py.
OPEN_POINTS = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.1), (0.3, 0.0), (0.2, -0.1), (0.1, -0.1)]  # EAR ~0.667
CLOSED_POINTS = [(0.0, 0.0), (0.1, 0.01), (0.2, 0.01), (0.3, 0.0), (0.2, -0.01), (0.1, -0.01)]  # EAR ~0.067


def ts(*parts: float) -> float:
    return round(math.fsum(parts), 9)


@pytest.fixture(autouse=True)
def _default_head_pose(monkeypatch):
    """Neutral CENTER head pose by default for every test in this file.
    estimate_head_pose's own geometry correctness (yaw/pitch/orientation
    from real landmark positions) is exhaustively tested in
    tests/test_head_pose.py - hand-crafting geometrically consistent
    landmarks here would be redundant and fragile (a fully degenerate
    all-same-point landmark set was tried first and found to make
    solvePnP converge to a spurious non-CENTER rotation rather than
    reporting failure). The one test that specifically exercises
    ATTENTION_DIVERTED overrides this with its own monkeypatch call."""
    from src.face.head_pose import HeadOrientation, HeadPoseResult

    monkeypatch.setattr(
        "src.core.app.estimate_head_pose",
        lambda *a, **k: HeadPoseResult(orientation=HeadOrientation.CENTER, yaw_degrees=0.0, pitch_degrees=0.0),
    )


def make_image() -> np.ndarray:
    return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)


def make_frame(timestamp: float, index: int = 0) -> Frame:
    return Frame(image=make_image(), timestamp=timestamp, index=index)


# --- Fake camera (mirrors tests/test_camera_manager.py's FakeVideoCapture) -----------


class FakeVideoCapture:
    def __init__(self, frame_count: int = 0, opened: bool = True) -> None:
        self._remaining = frame_count
        self._opened = opened
        self.release_calls = 0

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop_id, value) -> bool:
        return True

    def read(self):
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, make_image()

    def release(self) -> None:
        self.release_calls += 1
        self._opened = False


def make_camera(
    frame_count: int = 0, open_fails: bool = False, config: CameraConfig | None = None
) -> tuple[CameraManager, FakeVideoCapture]:
    cam_config = config or CameraConfig(index=0, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, target_fps=30)
    capture = FakeVideoCapture(frame_count=frame_count, opened=not open_fails)
    return CameraManager(cam_config, capture_factory=lambda index: capture), capture


# --- Fake YOLO model (mirrors tests/test_yolo_detector.py) ---------------------------


class FakeYoloBoxes:
    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class FakeYoloResult:
    def __init__(self, boxes, names) -> None:
        self.boxes = boxes
        self.names = names


def _make_yolo_result(entries) -> FakeYoloResult:
    if not entries:
        return FakeYoloResult(FakeYoloBoxes([], [], []), YOLO_NAMES)
    xyxy = [e[0] for e in entries]
    conf = [e[1] for e in entries]
    cls = [e[2] for e in entries]
    return FakeYoloResult(FakeYoloBoxes(xyxy, conf, cls), YOLO_NAMES)


class FakeYoloModel:
    """entries is mutated directly by tests between _process_frame() calls
    to control what the next detect() call reports."""

    def __init__(self, entries=(), raise_error: Exception | None = None) -> None:
        self.entries = list(entries)
        self.raise_error = raise_error
        self.calls = 0

    def __call__(self, source, conf, device, verbose=False):
        self.calls += 1
        if self.raise_error is not None:
            raise self.raise_error
        return [_make_yolo_result(self.entries)]


def make_yolo(model: FakeYoloModel, config: YoloConfig | None = None) -> YOLODetector:
    cfg = config or YoloConfig(model="models/yolo11n.pt", confidence=0.45, phone_confidence=0.55, device="cpu", detection_interval_seconds=0.0)
    return YOLODetector(cfg, model_factory=lambda path: model)


# --- Fake FaceLandmarker (mirrors tests/test_face_analyzer.py) -----------------------


class FakeLandmark:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def make_landmarks(left_points=None, right_points=None) -> list[FakeLandmark]:
    landmarks = [FakeLandmark(0.5, 0.5) for _ in range(400)]
    if left_points is not None:
        for idx, (x, y) in zip(LEFT_EYE_INDICES, left_points):
            landmarks[idx] = FakeLandmark(x, y)
    if right_points is not None:
        for idx, (x, y) in zip(RIGHT_EYE_INDICES, right_points):
            landmarks[idx] = FakeLandmark(x, y)
    return landmarks


class FakeFaceResult:
    def __init__(self, face_landmarks) -> None:
        self.face_landmarks = face_landmarks


class FakeFaceLandmarker:
    """face_landmarks is mutated directly by tests between _process_frame()
    calls; [] means "no face detected"."""

    def __init__(self, face_landmarks=None, raise_error: Exception | None = None) -> None:
        self.face_landmarks = face_landmarks if face_landmarks is not None else []
        self.raise_error = raise_error
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        if self.raise_error is not None:
            raise self.raise_error
        return FakeFaceResult(self.face_landmarks)


def make_face(landmarker: FakeFaceLandmarker) -> FaceAnalyzer:
    face_config = FaceConfig(model="models/face_landmarker.task")
    eyes_config = EyesConfig(closed_threshold=0.21, open_threshold=0.24, blink_max_duration_seconds=0.05, drowsiness_duration_seconds=0.1)
    return FaceAnalyzer(face_config, eyes_config, landmarker_factory=lambda path: landmarker)


# --- Fake audio backends (mirrors tests/test_audio_manager.py) -----------------------


class FakeSound:
    def __init__(self, path: str) -> None:
        self.path = path
        self.played_count = 0
        self.volume: float | None = None

    def play(self) -> None:
        self.played_count += 1

    def set_volume(self, volume: float) -> None:
        self.volume = volume


class FakeMixerBackend:
    def __init__(self, fail_init: bool = False) -> None:
        self.fail_init = fail_init
        self.init_calls = 0
        self.quit_calls = 0

    def init(self) -> None:
        self.init_calls += 1
        if self.fail_init:
            raise RuntimeError("no audio device")

    def quit(self) -> None:
        self.quit_calls += 1


class FakeMusicBackend:
    def __init__(self) -> None:
        self.load_calls: list[str] = []
        self.play_calls: list[int] = []
        self.paused = False
        self.stop_calls = 0
        self.volume: float | None = None

    def load(self, path: str) -> None:
        self.load_calls.append(path)

    def play(self, loops: int = 0) -> None:
        self.play_calls.append(loops)

    def pause(self) -> None:
        self.paused = True

    def unpause(self) -> None:
        self.paused = False

    def stop(self) -> None:
        self.stop_calls += 1

    def set_volume(self, volume: float) -> None:
        self.volume = volume


def _sound_factory(path: str) -> FakeSound:
    return FakeSound(path)


def make_audio(
    audio_config: AudioConfig | None = None, phone_config: PhoneConfig | None = None, fail_init: bool = False
) -> tuple[AudioManager, FakeMixerBackend, FakeMusicBackend]:
    mixer = FakeMixerBackend(fail_init=fail_init)
    music = FakeMusicBackend()
    manager = AudioManager(
        audio_config or AudioConfig(enabled=True, volume=0.5, music_enabled=True, music_volume=0.25, persistent_warning_interval_seconds=10.0),
        phone_config or PhoneConfig(confirm_duration_seconds=0.1, clear_duration_seconds=0.1, warning_cooldown_seconds=1.0),
        mixer_backend=mixer,
        music_backend=music,
        sound_factory=_sound_factory,
    )
    return manager, mixer, music


def sounds_of(audio: AudioManager) -> dict[str, FakeSound]:
    return audio._sounds  # type: ignore[attr-defined]


# --- AppConfig ------------------------------------------------------------------------


def make_app_config(**overrides) -> AppConfig:
    # detection_interval_seconds=0.0 by default: every _process_frame() call
    # re-runs (fake) YOLO detection, matching every existing test's
    # assumption. Dedicated interval-throttling tests override this field
    # explicitly (see the "Detection interval throttling" section below).
    defaults = dict(
        camera=CameraConfig(index=0, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, target_fps=30),
        yolo=YoloConfig(model="models/yolo11n.pt", confidence=0.45, phone_confidence=0.55, device="cpu", detection_interval_seconds=0.0),
        phone=PhoneConfig(confirm_duration_seconds=0.1, clear_duration_seconds=0.1, warning_cooldown_seconds=1.0),
        face=FaceConfig(model="models/face_landmarker.task"),
        eyes=EyesConfig(closed_threshold=0.21, open_threshold=0.24, blink_max_duration_seconds=0.05, drowsiness_duration_seconds=0.1),
        head=HeadConfig(yaw_threshold_degrees=20.0, pitch_threshold_degrees=18.0, confirmation_seconds=0.1, calibration_seconds=0.0),
        person=PersonConfig(away_duration_seconds=0.1),
        audio=AudioConfig(enabled=True, volume=0.5, music_enabled=True, music_volume=0.25, persistent_warning_interval_seconds=10.0),
        ui=UIConfig(debug=False),
        score=ScoreConfig(starting_score=100, phone_event_penalty=10, drowsiness_event_penalty=5, attention_event_penalty=3, away_event_penalty=5),
        session=SessionConfig(max_event_log_entries=100),
        source_path=Path("config/config.yaml"),
    )
    defaults.update(overrides)
    return AppConfig(**defaults)


def make_app(
    tmp_path: Path,
    *,
    config: AppConfig | None = None,
    yolo_model: FakeYoloModel | None = None,
    landmarker: FakeFaceLandmarker | None = None,
    camera_frame_count: int = 0,
    camera_open_fails: bool = False,
    audio_fail_init: bool = False,
    init_subsystems: bool = True,
) -> tuple[FocusGuardApp, dict]:
    cfg = config or make_app_config()
    camera, camera_capture = make_camera(frame_count=camera_frame_count, open_fails=camera_open_fails, config=cfg.camera)
    yolo_model = yolo_model if yolo_model is not None else FakeYoloModel()
    yolo = make_yolo(yolo_model, config=cfg.yolo)
    landmarker = landmarker if landmarker is not None else FakeFaceLandmarker()
    face = make_face(landmarker)
    ui = UIManager()
    audio, mixer, music = make_audio(cfg.audio, cfg.phone, fail_init=audio_fail_init)

    app = FocusGuardApp(
        cfg,
        camera_manager=camera,
        yolo_detector=yolo,
        face_analyzer=face,
        ui_manager=ui,
        audio_manager=audio,
        logs_directory=tmp_path,
    )

    if init_subsystems:
        yolo.load()
        face.load()
        try:
            audio.init()
        except Exception:
            pass

    fakes = dict(
        camera=camera, camera_capture=camera_capture, yolo_model=yolo_model, landmarker=landmarker, ui=ui,
        audio=audio, mixer=mixer, music=music,
    )
    return app, fakes


# =========================================================================================
# Startup / shutdown
# =========================================================================================


def test_startup_calls_ui_init_camera_open_yolo_load_face_load(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path, init_subsystems=False)

    app._startup()

    assert fakes["ui"].is_initialized is True
    assert app._camera.is_open is True
    assert app._yolo.is_loaded is True
    assert app._face.is_loaded is True
    app._shutdown()


def test_startup_audio_failure_is_caught_and_does_not_raise(tmp_path: Path, capsys) -> None:
    app, fakes = make_app(tmp_path, init_subsystems=False, audio_fail_init=True)

    app._startup()  # must not raise despite audio failing

    captured = capsys.readouterr()
    assert "continuing without sound" in captured.out
    app._shutdown()


def test_run_returns_one_on_camera_open_failure(tmp_path: Path, capsys) -> None:
    app, fakes = make_app(tmp_path, init_subsystems=False, camera_open_fails=True)

    result = app.run()

    assert result == 1
    captured = capsys.readouterr()
    assert "Fatal startup error" in captured.out


def test_run_returns_one_on_yolo_load_failure(tmp_path: Path, capsys) -> None:
    yolo_model = FakeYoloModel()
    app, fakes = make_app(tmp_path, init_subsystems=False, yolo_model=yolo_model)
    app._yolo = YOLODetector(
        app._config.yolo, model_factory=lambda path: (_ for _ in ()).throw(DetectionError("bad weights"))
    )

    result = app.run()

    assert result == 1


def test_run_returns_one_on_face_load_failure(tmp_path: Path, capsys) -> None:
    app, fakes = make_app(tmp_path, init_subsystems=False)
    app._face = FaceAnalyzer(
        app._config.face, app._config.eyes, landmarker_factory=lambda path: (_ for _ in ()).throw(FaceAnalysisError("bad model"))
    )

    result = app.run()

    assert result == 1


def test_run_returns_one_on_ui_init_failure(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path, init_subsystems=False)

    class FailingUIManager(UIManager):
        def init(self) -> None:
            raise UIError("no display")

    app._ui = FailingUIManager()

    result = app.run()

    assert result == 1


def test_shutdown_is_safe_after_partial_startup_failure(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path, init_subsystems=False, camera_open_fails=True)

    app.run()  # camera.open() fails inside _startup, _shutdown() still runs

    assert fakes["camera_capture"].release_calls == 1  # release() called even though open() never succeeded


def test_shutdown_releases_camera_audio_ui(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path)
    app._startup()

    app._shutdown()

    assert fakes["camera_capture"].release_calls == 1
    assert fakes["mixer"].quit_calls == 1
    assert fakes["ui"].is_initialized is False


# =========================================================================================
# _process_frame wiring: phone
# =========================================================================================


def test_phone_confirmed_updates_state_event_session_and_audio(tmp_path: Path) -> None:
    yolo_model = FakeYoloModel(entries=[PERSON_BOX, PHONE_BOX])
    app, fakes = make_app(tmp_path, yolo_model=yolo_model)
    app._start_session(0.0)

    app._process_frame(make_frame(0.0))
    result = app._process_frame(make_frame(ts(0.1)))  # crosses confirm_duration_seconds=0.1

    assert app._state_manager.state == FocusState.PHONE_DISTRACTION
    assert app._session_manager.phone_distraction_count == 1
    assert app._session_manager.focus_score == 100 - 10
    assert sounds_of(fakes["audio"])["phone_warning"].played_count == 1
    assert any(e.event_type == EventType.PHONE_DETECTED for e in app._event_manager.events)


def test_phone_cleared_after_phone_disappears(tmp_path: Path) -> None:
    """A real (non-degenerate) face+open-eyes landmark set is provided so
    vision_quality is GOOD once the phone clears - without it, no face
    data means DEGRADED vision_quality, and the state would correctly
    land on UNKNOWN rather than FOCUSED (that would be testing a
    different thing than this test intends)."""
    yolo_model = FakeYoloModel(entries=[PERSON_BOX, PHONE_BOX])
    landmarker = FakeFaceLandmarker(face_landmarks=[make_landmarks(OPEN_POINTS, OPEN_POINTS)])
    app, fakes = make_app(tmp_path, yolo_model=yolo_model, landmarker=landmarker)
    app._start_session(0.0)
    app._process_frame(make_frame(0.0))
    app._process_frame(make_frame(ts(0.1)))
    assert app._state_manager.state == FocusState.PHONE_DISTRACTION

    yolo_model.entries = [PERSON_BOX]  # phone removed
    app._process_frame(make_frame(ts(0.1, 0.1)))
    # Generous margin past clear_duration_seconds=0.1 - PhoneTemporalFilter
    # has no epsilon tolerance on this boundary (unlike DurationConfirmer/
    # HeadOrientationFilter), so landing exactly on it can lose to float
    # subtraction noise; that boundary precision is already exhaustively
    # covered in tests/test_phone_temporal_filter.py.
    app._process_frame(make_frame(ts(0.1, 0.5)))

    assert app._state_manager.state == FocusState.FOCUSED
    assert any(e.event_type == EventType.PHONE_CLEARED for e in app._event_manager.events)


# =========================================================================================
# _process_frame wiring: drowsiness
# =========================================================================================


def test_drowsiness_confirmed_after_sustained_eye_closure(tmp_path: Path) -> None:
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    landmarker = FakeFaceLandmarker(face_landmarks=[make_landmarks(CLOSED_POINTS, CLOSED_POINTS)])
    app, fakes = make_app(tmp_path, yolo_model=yolo_model, landmarker=landmarker)
    app._start_session(0.0)

    app._process_frame(make_frame(0.0))
    app._process_frame(make_frame(ts(0.1)))  # crosses drowsiness_duration_seconds=0.1

    assert app._state_manager.state == FocusState.DROWSINESS_SIGNAL
    assert app._session_manager.drowsiness_count == 1
    assert app._session_manager.focus_score == 100 - 5
    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 1


# =========================================================================================
# _process_frame wiring: attention (head orientation via monkeypatch - see module docstring)
# =========================================================================================


def test_attention_diverted_after_sustained_off_center_head(tmp_path: Path, monkeypatch) -> None:
    """estimate_head_pose's own geometry correctness is exhaustively tested
    in tests/test_head_pose.py; here only the wiring (confirmed off-center
    orientation -> ATTENTION_DIVERTED -> event -> audio) is under test, so
    the function is monkeypatched to avoid reimplementing 3D landmark
    synthesis for a result that test_head_pose.py already guarantees."""
    from src.face.head_pose import HeadOrientation, HeadPoseResult

    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    landmarker = FakeFaceLandmarker(face_landmarks=[make_landmarks(OPEN_POINTS, OPEN_POINTS)])
    app, fakes = make_app(tmp_path, yolo_model=yolo_model, landmarker=landmarker)

    app._start_session(0.0)
    # head.calibration_seconds=0.0 in the test config, so this first frame
    # (still using the autouse CENTER/(0,0) fixture) instantly becomes the
    # calibration baseline before the LEFT override below takes effect.
    app._process_frame(make_frame(0.0))

    monkeypatch.setattr(
        "src.core.app.estimate_head_pose",
        lambda *a, **k: HeadPoseResult(orientation=HeadOrientation.LEFT, yaw_degrees=-30.0, pitch_degrees=0.0),
    )

    app._process_frame(make_frame(ts(0.1)))  # off-center begins, relative to the (0,0) baseline
    app._process_frame(make_frame(ts(0.2)))  # crosses head.confirmation_seconds=0.1

    assert app._state_manager.state == FocusState.ATTENTION_DIVERTED
    assert app._session_manager.attention_diversion_count == 1
    assert app._session_manager.focus_score == 100 - 3
    assert sounds_of(fakes["audio"])["attention_warning"].played_count == 1


# =========================================================================================
# _process_frame wiring: away
# =========================================================================================


def test_away_confirmed_after_sustained_absence(tmp_path: Path) -> None:
    yolo_model = FakeYoloModel(entries=[])  # no person at all
    app, fakes = make_app(tmp_path, yolo_model=yolo_model)
    app._start_session(0.0)

    app._process_frame(make_frame(0.0))
    app._process_frame(make_frame(ts(0.1)))  # crosses person.away_duration_seconds=0.1

    assert app._state_manager.state == FocusState.AWAY
    assert app._session_manager.away_count == 1
    assert app._session_manager.focus_score == 100 - 5
    assert any(e.event_type == EventType.PERSON_LEFT for e in app._event_manager.events)


# =========================================================================================
# _process_frame wiring: focus restored
# =========================================================================================


def test_focus_restored_does_not_fire_on_session_start_directly_into_focused(tmp_path: Path) -> None:
    """Regression guard: sitting down already-focused at session start
    (UNKNOWN -> FOCUSED, the very first evaluate() call) must not
    spuriously fire FOCUS_RESTORED - only clearing an actual distraction
    should (PRD section 38's demo only expects it after e.g. "put phone
    away")."""
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    landmarker = FakeFaceLandmarker(face_landmarks=[make_landmarks(OPEN_POINTS, OPEN_POINTS)])
    app, fakes = make_app(tmp_path, yolo_model=yolo_model, landmarker=landmarker)
    app._start_session(0.0)

    app._process_frame(make_frame(0.0))

    assert app._state_manager.state == FocusState.FOCUSED
    assert not any(e.event_type == EventType.FOCUS_RESTORED for e in app._event_manager.events)
    assert sounds_of(fakes["audio"])["focus_restored"].played_count == 0


def test_focus_restored_event_and_audio_on_return_to_focused(tmp_path: Path) -> None:
    """A real face+open-eyes landmark set is required so vision_quality is
    GOOD once the phone clears - see test_phone_cleared_after_phone_disappears."""
    yolo_model = FakeYoloModel(entries=[PERSON_BOX, PHONE_BOX])
    landmarker = FakeFaceLandmarker(face_landmarks=[make_landmarks(OPEN_POINTS, OPEN_POINTS)])
    app, fakes = make_app(tmp_path, yolo_model=yolo_model, landmarker=landmarker)
    app._start_session(0.0)
    app._process_frame(make_frame(0.0))
    app._process_frame(make_frame(ts(0.1)))
    assert app._state_manager.state == FocusState.PHONE_DISTRACTION

    yolo_model.entries = [PERSON_BOX]
    app._process_frame(make_frame(ts(0.1, 0.1)))
    # Generous margin past clear_duration_seconds=0.1 - PhoneTemporalFilter
    # has no epsilon tolerance on this boundary (unlike DurationConfirmer/
    # HeadOrientationFilter), so landing exactly on it can lose to float
    # subtraction noise; that boundary precision is already exhaustively
    # covered in tests/test_phone_temporal_filter.py.
    app._process_frame(make_frame(ts(0.1, 0.5)))

    assert app._state_manager.state == FocusState.FOCUSED
    assert any(e.event_type == EventType.FOCUS_RESTORED for e in app._event_manager.events)
    assert sounds_of(fakes["audio"])["focus_restored"].played_count == 1


# =========================================================================================
# Detection interval throttling (PRD section 29, Phase 13)
# =========================================================================================


def _yolo_config_with_interval(interval_seconds: float) -> YoloConfig:
    return YoloConfig(
        model="models/yolo11n.pt",
        confidence=0.45,
        phone_confidence=0.55,
        device="cpu",
        detection_interval_seconds=interval_seconds,
    )


def test_yolo_not_called_again_within_the_interval_window(tmp_path: Path) -> None:
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    cfg = make_app_config(yolo=_yolo_config_with_interval(0.1))
    app, fakes = make_app(tmp_path, config=cfg, yolo_model=yolo_model)
    app._start_session(0.0)

    app._process_frame(make_frame(0.0))
    assert yolo_model.calls == 1

    app._process_frame(make_frame(ts(0.05)))  # well within the 0.1s window

    assert yolo_model.calls == 1  # not called again


def test_yolo_reuses_last_detections_within_the_interval_window(tmp_path: Path) -> None:
    """The reused (stale) detections must still flow through the full
    pipeline (snapshot/filters/state) on the skipped frame - only the real
    YOLO call itself is skipped."""
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    cfg = make_app_config(yolo=_yolo_config_with_interval(0.1))
    app, fakes = make_app(tmp_path, config=cfg, yolo_model=yolo_model)
    app._start_session(0.0)
    app._process_frame(make_frame(0.0))

    yolo_model.entries = [PERSON_BOX, PHONE_BOX]  # would matter if re-detected, but interval blocks it
    app._process_frame(make_frame(ts(0.05)))

    assert yolo_model.calls == 1
    assert app._last_snapshot.person_present is True  # reused detection still flowed through
    assert app._last_snapshot.phone_detected is False  # the stale (pre-phone) result, not the new entries


def test_yolo_called_again_once_interval_elapses(tmp_path: Path) -> None:
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    cfg = make_app_config(yolo=_yolo_config_with_interval(0.1))
    app, fakes = make_app(tmp_path, config=cfg, yolo_model=yolo_model)
    app._start_session(0.0)
    app._process_frame(make_frame(0.0))
    assert yolo_model.calls == 1

    app._process_frame(make_frame(ts(0.1)))  # exactly at the boundary

    assert yolo_model.calls == 2


def test_yolo_called_again_well_past_interval(tmp_path: Path) -> None:
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    cfg = make_app_config(yolo=_yolo_config_with_interval(0.1))
    app, fakes = make_app(tmp_path, config=cfg, yolo_model=yolo_model)
    app._start_session(0.0)
    app._process_frame(make_frame(0.0))

    app._process_frame(make_frame(ts(0.5)))

    assert yolo_model.calls == 2


def test_zero_interval_calls_yolo_every_frame(tmp_path: Path) -> None:
    """detection_interval_seconds=0.0 must behave identically to the
    pre-Phase-13 (every-frame) behavior - this is also make_app_config()'s
    default, so every other test in this file relies on it implicitly."""
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    cfg = make_app_config(yolo=_yolo_config_with_interval(0.0))
    app, fakes = make_app(tmp_path, config=cfg, yolo_model=yolo_model)
    app._start_session(0.0)

    for i in range(5):
        app._process_frame(make_frame(ts(0.0, i * 0.01)))

    assert yolo_model.calls == 5


def test_face_analysis_still_runs_every_frame_regardless_of_yolo_interval(tmp_path: Path) -> None:
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    landmarker = FakeFaceLandmarker(face_landmarks=[make_landmarks(OPEN_POINTS, OPEN_POINTS)])
    cfg = make_app_config(yolo=_yolo_config_with_interval(0.1))
    app, fakes = make_app(tmp_path, config=cfg, yolo_model=yolo_model, landmarker=landmarker)
    app._start_session(0.0)
    app._process_frame(make_frame(0.0))
    assert landmarker.calls == 1

    app._process_frame(make_frame(ts(0.01)))  # well within the YOLO interval window

    assert landmarker.calls == 2  # face analysis is never throttled
    assert yolo_model.calls == 1  # YOLO was correctly skipped


def test_state_and_session_still_update_every_frame_while_yolo_is_throttled(tmp_path: Path) -> None:
    """Drowsiness confirmation (driven purely by eye state, independent of
    YOLO) must still work correctly while YOLO calls are being throttled -
    proving the filters/state/session pipeline is untouched by the interval."""
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    landmarker = FakeFaceLandmarker(face_landmarks=[make_landmarks(CLOSED_POINTS, CLOSED_POINTS)])
    cfg = make_app_config(yolo=_yolo_config_with_interval(10.0))  # huge interval: YOLO only ever called once
    app, fakes = make_app(tmp_path, config=cfg, yolo_model=yolo_model, landmarker=landmarker)
    app._start_session(0.0)

    app._process_frame(make_frame(0.0))
    app._process_frame(make_frame(ts(0.1)))  # crosses drowsiness_duration_seconds=0.1

    assert yolo_model.calls == 1  # YOLO never called a second time
    assert app._state_manager.state == FocusState.DROWSINESS_SIGNAL
    assert app._session_manager.drowsiness_count == 1


def test_detection_interval_state_resets_on_new_session(tmp_path: Path) -> None:
    """A brand new session must not inherit the previous session's YOLO
    timing/detection state - its very first frame must run a real
    detection regardless of how recently the prior session called YOLO."""
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    cfg = make_app_config(yolo=_yolo_config_with_interval(10.0))
    app, fakes = make_app(tmp_path, config=cfg, yolo_model=yolo_model)
    app._start_session(0.0)
    app._process_frame(make_frame(0.0))
    assert yolo_model.calls == 1

    app._handle_reset(1.0)  # idle/paused-safe reset per PRD section 23
    app._start_session(2.0)
    app._process_frame(make_frame(2.0))

    assert yolo_model.calls == 2


# =========================================================================================
# Persistent audio reminders (sustained conditions, not just the confirm edge)
# =========================================================================================


def _audio_config_with_reminder_interval(interval_seconds: float) -> AudioConfig:
    return AudioConfig(
        enabled=True,
        volume=0.5,
        music_enabled=True,
        music_volume=0.25,
        persistent_warning_interval_seconds=interval_seconds,
    )


def test_drowsiness_persistent_reminder_fires_after_interval_then_repeats(tmp_path: Path) -> None:
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    landmarker = FakeFaceLandmarker(face_landmarks=[make_landmarks(CLOSED_POINTS, CLOSED_POINTS)])
    cfg = make_app_config(audio=_audio_config_with_reminder_interval(0.1))
    app, fakes = make_app(tmp_path, config=cfg, yolo_model=yolo_model, landmarker=landmarker)
    app._start_session(0.0)

    app._process_frame(make_frame(0.0))
    app._process_frame(make_frame(ts(0.1)))  # crosses drowsiness_duration_seconds=0.1: confirmed
    assert app._state_manager.state == FocusState.DROWSINESS_SIGNAL
    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 1  # the initial one-shot alert

    app._process_frame(make_frame(ts(0.1, 0.05)))  # condition continues, not yet due for a reminder
    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 1

    app._process_frame(make_frame(ts(0.1, 0.1)))  # persistent interval elapsed since confirmation
    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 2

    app._process_frame(make_frame(ts(0.1, 0.15)))  # too soon for a third
    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 2

    app._process_frame(make_frame(ts(0.1, 0.2)))  # another full interval: repeats again
    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 3


def test_drowsiness_reminder_does_not_fire_if_condition_clears_first(tmp_path: Path) -> None:
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    landmarker = FakeFaceLandmarker(face_landmarks=[make_landmarks(CLOSED_POINTS, CLOSED_POINTS)])
    cfg = make_app_config(audio=_audio_config_with_reminder_interval(0.1))
    app, fakes = make_app(tmp_path, config=cfg, yolo_model=yolo_model, landmarker=landmarker)
    app._start_session(0.0)
    app._process_frame(make_frame(0.0))
    app._process_frame(make_frame(ts(0.1)))  # confirmed, initial warning
    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 1

    landmarker.face_landmarks = [make_landmarks(OPEN_POINTS, OPEN_POINTS)]  # eyes open again
    app._process_frame(make_frame(ts(0.1, 0.05)))  # clears before the reminder interval elapses

    app._process_frame(make_frame(ts(0.1, 0.2)))  # well past where a reminder would have fired

    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 1  # no spurious reminder


def test_phone_and_attention_persistent_reminders_also_repeat(tmp_path: Path, monkeypatch) -> None:
    from src.face.head_pose import HeadOrientation, HeadPoseResult

    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    landmarker = FakeFaceLandmarker(face_landmarks=[make_landmarks(OPEN_POINTS, OPEN_POINTS)])
    cfg = make_app_config(audio=_audio_config_with_reminder_interval(0.1))
    app, fakes = make_app(tmp_path, config=cfg, yolo_model=yolo_model, landmarker=landmarker)
    app._start_session(0.0)

    # Calibration frame: no phone yet, CENTER head (autouse fixture) -> baseline (0,0).
    app._process_frame(make_frame(0.0))

    yolo_model.entries = [PERSON_BOX, PHONE_BOX]
    monkeypatch.setattr(
        "src.core.app.estimate_head_pose",
        lambda *a, **k: HeadPoseResult(orientation=HeadOrientation.LEFT, yaw_degrees=-30.0, pitch_degrees=0.0),
    )

    app._process_frame(make_frame(ts(0.1)))  # both phone and attention timers start now
    app._process_frame(make_frame(ts(0.2)))  # both confirmed (0.1s elapsed)
    assert app._state_manager.state == FocusState.PHONE_DISTRACTION  # phone wins priority
    assert sounds_of(fakes["audio"])["phone_warning"].played_count == 1
    assert sounds_of(fakes["audio"])["attention_warning"].played_count == 1

    app._process_frame(make_frame(ts(0.2, 0.1)))  # persistent interval elapsed for both

    assert sounds_of(fakes["audio"])["phone_warning"].played_count == 2
    assert sounds_of(fakes["audio"])["attention_warning"].played_count == 2


def test_reminder_resets_on_pause_so_resume_does_not_fire_immediately(tmp_path: Path) -> None:
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    landmarker = FakeFaceLandmarker(face_landmarks=[make_landmarks(CLOSED_POINTS, CLOSED_POINTS)])
    cfg = make_app_config(audio=_audio_config_with_reminder_interval(0.1))
    app, fakes = make_app(tmp_path, config=cfg, yolo_model=yolo_model, landmarker=landmarker)
    app._start_session(0.0)
    app._process_frame(make_frame(0.0))
    app._process_frame(make_frame(ts(0.1)))  # confirmed, initial warning
    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 1

    app._pause_session(ts(0.1, 0.05))
    app._resume_session(1000.0)  # a huge real-world gap while paused

    # Without the pause reset, this would immediately look "overdue" (far
    # more than one interval has passed since confirmation) and fire right
    # away. It must not - the paused gap was never actually monitored.
    app._process_frame(make_frame(1000.0))
    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 1

    app._process_frame(make_frame(ts(1000.0, 0.1)))  # a fresh interval after resuming
    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 2


def test_reminder_state_resets_on_new_session(tmp_path: Path) -> None:
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    landmarker = FakeFaceLandmarker(face_landmarks=[make_landmarks(CLOSED_POINTS, CLOSED_POINTS)])
    cfg = make_app_config(audio=_audio_config_with_reminder_interval(0.1))
    app, fakes = make_app(tmp_path, config=cfg, yolo_model=yolo_model, landmarker=landmarker)
    app._start_session(0.0)
    app._process_frame(make_frame(0.0))
    app._process_frame(make_frame(ts(0.1)))  # confirmed, initial warning
    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 1

    app._end_session(ts(0.1, 0.05))
    app._start_session(ts(0.1, 0.1))  # a new session, well past where an old reminder would fire

    app._process_frame(make_frame(ts(0.1, 0.1)))  # first frame of the new session: fresh onset only
    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 1  # no stale reminder carried over


def test_reminder_state_resets_on_r_reset(tmp_path: Path) -> None:
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    landmarker = FakeFaceLandmarker(face_landmarks=[make_landmarks(CLOSED_POINTS, CLOSED_POINTS)])
    cfg = make_app_config(audio=_audio_config_with_reminder_interval(0.1))
    app, fakes = make_app(tmp_path, config=cfg, yolo_model=yolo_model, landmarker=landmarker)
    app._start_session(0.0)
    app._process_frame(make_frame(0.0))
    app._process_frame(make_frame(ts(0.1)))  # confirmed, initial warning
    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 1

    app._pause_session(ts(0.1, 0.01))  # R is only honored while paused/idle
    app._handle_reset(ts(0.1, 0.02))

    app._start_session(ts(0.1, 0.1))
    app._process_frame(make_frame(ts(0.1, 0.1)))  # fresh session, first frame: no immediate reminder
    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 1


def test_muted_persistent_reminder_does_not_play_at_app_level(tmp_path: Path) -> None:
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    landmarker = FakeFaceLandmarker(face_landmarks=[make_landmarks(CLOSED_POINTS, CLOSED_POINTS)])
    cfg = make_app_config(audio=_audio_config_with_reminder_interval(0.1))
    app, fakes = make_app(tmp_path, config=cfg, yolo_model=yolo_model, landmarker=landmarker)
    app._start_session(0.0)
    app._audio.set_muted(True)

    app._process_frame(make_frame(0.0))
    app._process_frame(make_frame(ts(0.1)))  # would confirm + warn if unmuted
    app._process_frame(make_frame(ts(0.1, 0.1)))  # would repeat if unmuted

    assert sounds_of(fakes["audio"])["drowsiness_warning"].played_count == 0


# =========================================================================================
# Per-frame error handling (PRD section 35)
# =========================================================================================


def test_yolo_inference_exception_emits_model_error_and_does_not_crash(tmp_path: Path, capsys) -> None:
    yolo_model = FakeYoloModel(raise_error=DetectionError("inference failed"))
    app, fakes = make_app(tmp_path, yolo_model=yolo_model)
    app._start_session(0.0)

    app._process_frame(make_frame(0.0))  # must not raise

    assert any(e.event_type == EventType.MODEL_ERROR for e in app._event_manager.events)
    assert app._last_detections == []
    captured = capsys.readouterr()
    assert "YOLO inference error" in captured.out


def test_face_analysis_exception_emits_vision_error_and_does_not_crash(tmp_path: Path, capsys) -> None:
    landmarker = FakeFaceLandmarker(raise_error=FaceAnalysisError("inference failed"))
    app, fakes = make_app(tmp_path, landmarker=landmarker)
    app._start_session(0.0)

    app._process_frame(make_frame(0.0))  # must not raise

    assert any(e.event_type == EventType.VISION_ERROR for e in app._event_manager.events)
    assert app._last_face_result.face_detected is False
    captured = capsys.readouterr()
    assert "Face analysis error" in captured.out


def test_camera_read_error_stops_the_app_cleanly(tmp_path: Path, capsys) -> None:
    app, fakes = make_app(tmp_path)

    class FailingCapture:
        def isOpened(self):
            return True

        def set(self, *a):
            return True

        def read(self):
            return False, None

        def release(self):
            pass

    app._camera = CameraManager(app._config.camera, capture_factory=lambda index: FailingCapture())
    app._camera.open()
    app._ui.init()

    app._run_one_iteration()

    assert app._running is False
    captured = capsys.readouterr()
    assert "Camera error" in captured.out
    app._shutdown()


# =========================================================================================
# Session control / actions (PRD section 23)
# =========================================================================================


def test_space_starts_session(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path)

    app._handle_actions([UIAction.START_PAUSE_RESUME], 0.0)

    assert app._session_manager.is_active is True
    assert app._state_manager.state == FocusState.UNKNOWN
    assert any(e.event_type == EventType.SESSION_STARTED for e in app._event_manager.events)
    assert fakes["music"].play_calls == [-1]


def test_space_start_then_pause_then_resume(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path)
    app._handle_actions([UIAction.START_PAUSE_RESUME], 0.0)

    app._handle_actions([UIAction.START_PAUSE_RESUME], 1.0)
    assert app._session_manager.is_paused is True
    assert fakes["music"].paused is True

    app._handle_actions([UIAction.START_PAUSE_RESUME], 2.0)
    assert app._session_manager.is_paused is False
    assert fakes["music"].paused is False


def test_reset_is_ignored_while_actively_running(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path)
    app._handle_actions([UIAction.START_PAUSE_RESUME], 0.0)  # active, not paused

    app._handle_actions([UIAction.RESET], 1.0)

    assert app._session_manager.is_active is True  # unaffected


def test_reset_allowed_while_paused(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path)
    app._handle_actions([UIAction.START_PAUSE_RESUME], 0.0)
    app._handle_actions([UIAction.START_PAUSE_RESUME], 1.0)  # pause

    app._handle_actions([UIAction.RESET], 2.0)

    assert app._session_manager.is_active is False
    assert app._state_manager.state == FocusState.IDLE


def test_reset_allowed_while_idle(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path)

    app._handle_actions([UIAction.RESET], 0.0)  # must not raise

    assert app._session_manager.is_active is False


def test_exit_while_idle_does_not_end_session(tmp_path: Path, capsys) -> None:
    app, fakes = make_app(tmp_path)

    app._handle_actions([UIAction.EXIT], 0.0)

    assert app._running is False
    captured = capsys.readouterr()
    assert "Session Summary" not in captured.out


def test_exit_while_active_ends_session_shows_and_saves_summary(tmp_path: Path, capsys) -> None:
    app, fakes = make_app(tmp_path)
    app._handle_actions([UIAction.START_PAUSE_RESUME], 0.0)

    app._handle_actions([UIAction.EXIT], 10.0)

    assert app._running is False
    assert app._session_manager.is_active is False
    captured = capsys.readouterr()
    assert "Session Summary" in captured.out
    assert "Session summary saved to" in captured.out
    saved_files = list(tmp_path.glob("session_*.json"))
    assert len(saved_files) == 1


def test_exit_while_paused_ends_session(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path)
    app._handle_actions([UIAction.START_PAUSE_RESUME], 0.0)
    app._handle_actions([UIAction.START_PAUSE_RESUME], 1.0)  # pause

    app._handle_actions([UIAction.EXIT], 2.0)

    assert app._running is False
    assert app._session_manager.is_active is False


def test_toggle_mute_action(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path)

    app._handle_actions([UIAction.TOGGLE_MUTE], 0.0)

    assert fakes["audio"].is_muted is True


def test_toggle_debug_action(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path)
    assert app._debug is False

    app._handle_actions([UIAction.TOGGLE_DEBUG], 0.0)

    assert app._debug is True


def test_exit_action_short_circuits_remaining_actions_in_same_batch(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path)

    app._handle_actions([UIAction.EXIT, UIAction.TOGGLE_DEBUG], 0.0)

    assert app._debug is False  # never reached


# =========================================================================================
# DashboardView assembly
# =========================================================================================


def test_dashboard_view_paused_reflects_session_manager(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path)
    app._handle_actions([UIAction.START_PAUSE_RESUME], 0.0)
    app._handle_actions([UIAction.START_PAUSE_RESUME], 1.0)  # pause

    view = app._build_dashboard_view(make_frame(2.0))

    assert view.paused is True


def test_dashboard_view_before_any_session_has_safe_defaults(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path)

    view = app._build_dashboard_view(make_frame(0.0))

    assert view.status == FocusState.IDLE
    assert view.person_present is False
    assert view.debug_info is None


def test_dashboard_view_debug_info_populated_only_when_debug_true(tmp_path: Path) -> None:
    yolo_model = FakeYoloModel(entries=[PERSON_BOX])
    app, fakes = make_app(tmp_path, yolo_model=yolo_model)
    app._start_session(0.0)
    app._process_frame(make_frame(0.0))

    view_no_debug = app._build_dashboard_view(make_frame(0.1))
    assert view_no_debug.debug_info is None

    app._debug = True
    view_debug = app._build_dashboard_view(make_frame(0.1))
    assert view_debug.debug_info is not None
    assert len(view_debug.debug_info.detections) == 1


def test_dashboard_view_inference_latency_sums_yolo_and_face(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path)
    app._start_session(0.0)
    app._process_frame(make_frame(0.0))

    view = app._build_dashboard_view(make_frame(0.1))

    expected = (app._yolo.last_inference_ms or 0.0) + (app._face.last_inference_ms or 0.0)
    assert view.inference_latency_ms == pytest.approx(expected)


# =========================================================================================
# FPS tracking
# =========================================================================================


def test_fps_updates_after_two_frames(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path)
    assert app._fps is None  # None before any frame has been tracked

    app._track_fps(0.0)
    assert app._fps is None  # first call only seeds _last_frame_time

    app._track_fps(0.1)  # 10 fps instantaneous
    assert app._fps == pytest.approx(10.0)


def test_fps_smooths_across_multiple_frames(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path)
    app._track_fps(0.0)
    app._track_fps(0.1)  # seeds fps=10.0
    first = app._fps

    app._track_fps(0.15)  # 20 fps instantaneous, smoothed toward it

    assert app._fps > first
    assert app._fps < 20.0


# =========================================================================================
# Full run()/_main_loop() smoke tests
# =========================================================================================


def test_run_completes_cleanly_with_camera_running_dry(tmp_path: Path) -> None:
    """No QUIT event is posted - the fake camera simply runs out of frames,
    which _handle_camera_error treats as a clean stop (PRD section 6)."""
    app, fakes = make_app(tmp_path, camera_frame_count=5)

    result = app.run()

    assert result == 0
    assert fakes["ui"].is_initialized is False  # shut down


def test_run_with_quit_event_exits_cleanly(tmp_path: Path) -> None:
    app, fakes = make_app(tmp_path, camera_frame_count=1000)
    pygame.init()  # idempotent; needed before touching pygame.event in this process
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.QUIT))

    result = app.run()

    assert result == 0


def test_run_starts_session_processes_frames_and_exits(tmp_path: Path) -> None:
    """SPACE then Q, spread across a short run of real frames, exercising
    the full loop: capture -> process -> render -> input handling."""
    app, fakes = make_app(tmp_path, camera_frame_count=3)
    pygame.init()  # idempotent; needed before touching pygame.event in this process
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))

    result = app.run()

    assert result == 0
    # camera ran dry after 3 frames -> clean stop; session was active at some point
    assert fakes["camera_capture"].release_calls == 1
