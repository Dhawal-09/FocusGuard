"""Tests for FaceAnalyzer (FOCUSGUARD_PRD.md section 10/11).

Uses a fake landmarker double shaped like the real MediaPipe Tasks API
(landmarker.detect(mp_image) -> result, where result.face_landmarks is a
list of per-face landmark lists, each landmark exposing .x/.y). No real
MediaPipe model, webcam, GPU, or network access is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.config_manager import EyesConfig, FaceConfig
from src.face.eye_metrics import LEFT_EYE_INDICES, RIGHT_EYE_INDICES, EyeState
from src.face.face_analyzer import FaceAnalysisError, FaceAnalyzer

IMAGE_SIZE = 100  # square image: normalized->pixel scaling preserves EAR ratios

CLOSED_THRESHOLD = 0.21
OPEN_THRESHOLD = 0.24

# Same synthetic (corner, top, top, corner, bottom, bottom) point patterns as
# test_eye_metrics.py, reused here as normalized landmark coordinates.
OPEN_POINTS = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.1), (0.3, 0.0), (0.2, -0.1), (0.1, -0.1)]  # EAR ~0.667
NARROW_OPEN_POINTS = [(0.0, 0.0), (0.1, 0.05), (0.2, 0.05), (0.3, 0.0), (0.2, -0.05), (0.1, -0.05)]  # EAR ~0.333
CLOSED_POINTS = [(0.0, 0.0), (0.1, 0.01), (0.2, 0.01), (0.3, 0.0), (0.2, -0.01), (0.1, -0.01)]  # EAR ~0.067
MID_ZONE_POINTS = [(0.0, 0.0), (0.1, 0.03375), (0.2, 0.03375), (0.3, 0.0), (0.2, -0.03375), (0.1, -0.03375)]  # EAR ~0.225
DEGENERATE_POINTS = [(0.15, 0.0), (0.1, 0.1), (0.2, 0.1), (0.15, 0.0), (0.2, -0.1), (0.1, -0.1)]  # EAR None


def make_config() -> FaceConfig:
    return FaceConfig(model="models/face_landmarker.task")


def make_eyes_config() -> EyesConfig:
    return EyesConfig(
        closed_threshold=CLOSED_THRESHOLD,
        open_threshold=OPEN_THRESHOLD,
        blink_max_duration_seconds=0.45,
        drowsiness_duration_seconds=1.20,
    )


def make_image() -> np.ndarray:
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


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


class FakeResult:
    def __init__(self, face_landmarks: list[list[FakeLandmark]]) -> None:
        self.face_landmarks = face_landmarks


class FakeLandmarker:
    """Consumes a list of results one per call; repeats the last one if
    only a single result was supplied."""

    def __init__(self, results=None, raise_error: Exception | None = None) -> None:
        if results is None:
            results = [FakeResult([])]
        self._results = list(results) if isinstance(results, list) else [results]
        self._raise_error = raise_error
        self.detect_calls = 0

    def detect(self, image):
        self.detect_calls += 1
        if self._raise_error is not None:
            raise self._raise_error
        if len(self._results) > 1:
            return self._results.pop(0)
        return self._results[0]


def factory_for(fake: FakeLandmarker):
    def _factory(model_path: str) -> FakeLandmarker:
        return fake

    return _factory


def make_analyzer(fake: FakeLandmarker) -> FaceAnalyzer:
    return FaceAnalyzer(make_config(), make_eyes_config(), landmarker_factory=factory_for(fake))


# --- load() -----------------------------------------------------------------


def test_load_succeeds() -> None:
    analyzer = make_analyzer(FakeLandmarker())

    analyzer.load()

    assert analyzer.is_loaded is True


def test_load_is_idempotent() -> None:
    fake = FakeLandmarker()
    factory_calls: list[str] = []

    def factory(model_path: str) -> FakeLandmarker:
        factory_calls.append(model_path)
        return fake

    analyzer = FaceAnalyzer(make_config(), make_eyes_config(), landmarker_factory=factory)

    analyzer.load()
    analyzer.load()

    assert len(factory_calls) == 1


def test_load_wraps_factory_exception_in_face_analysis_error() -> None:
    def failing_factory(model_path: str):
        raise OSError("model file not found")

    analyzer = FaceAnalyzer(make_config(), make_eyes_config(), landmarker_factory=failing_factory)

    with pytest.raises(FaceAnalysisError, match="Failed to load face landmark model"):
        analyzer.load()

    assert analyzer.is_loaded is False


# --- analyze(): face/landmark presence -----------------------------------------


def test_analyze_before_load_raises_face_analysis_error() -> None:
    analyzer = make_analyzer(FakeLandmarker())

    with pytest.raises(FaceAnalysisError, match="not loaded"):
        analyzer.analyze(make_image(), timestamp=1.0)


def test_analyze_no_face_returns_unknown() -> None:
    analyzer = make_analyzer(FakeLandmarker(results=[FakeResult([])]))
    analyzer.load()

    result = analyzer.analyze(make_image(), timestamp=1.0)

    assert result.face_detected is False
    assert result.eyes_state == EyeState.UNKNOWN
    assert result.eye_metric is None


def test_analyze_valid_face_open_eyes() -> None:
    landmarks = make_landmarks(left_points=OPEN_POINTS, right_points=OPEN_POINTS)
    analyzer = make_analyzer(FakeLandmarker(results=[FakeResult([landmarks])]))
    analyzer.load()

    result = analyzer.analyze(make_image(), timestamp=1.0)

    assert result.face_detected is True
    assert result.eyes_state == EyeState.OPEN
    assert result.eye_metric == pytest.approx(0.6667, abs=1e-3)
    assert result.landmarks is landmarks


def test_analyze_valid_face_closed_eyes() -> None:
    landmarks = make_landmarks(left_points=CLOSED_POINTS, right_points=CLOSED_POINTS)
    analyzer = make_analyzer(FakeLandmarker(results=[FakeResult([landmarks])]))
    analyzer.load()

    result = analyzer.analyze(make_image(), timestamp=1.0)

    assert result.face_detected is True
    assert result.eyes_state == EyeState.CLOSED
    assert result.eye_metric == pytest.approx(0.0667, abs=1e-3)


def test_analyze_both_eyes_averaged() -> None:
    landmarks = make_landmarks(left_points=OPEN_POINTS, right_points=NARROW_OPEN_POINTS)
    analyzer = make_analyzer(FakeLandmarker(results=[FakeResult([landmarks])]))
    analyzer.load()

    result = analyzer.analyze(make_image(), timestamp=1.0)

    # left ~0.6667, right ~0.3333 -> average ~0.5
    assert result.eye_metric == pytest.approx(0.5, abs=1e-3)


def test_analyze_one_eye_fallback_when_other_degenerate() -> None:
    landmarks = make_landmarks(left_points=OPEN_POINTS, right_points=DEGENERATE_POINTS)
    analyzer = make_analyzer(FakeLandmarker(results=[FakeResult([landmarks])]))
    analyzer.load()

    result = analyzer.analyze(make_image(), timestamp=1.0)

    assert result.face_detected is True
    assert result.eye_metric == pytest.approx(0.6667, abs=1e-3)
    assert result.eyes_state == EyeState.OPEN


def test_analyze_neither_eye_usable_returns_unknown_with_face_detected() -> None:
    landmarks = make_landmarks(left_points=DEGENERATE_POINTS, right_points=DEGENERATE_POINTS)
    analyzer = make_analyzer(FakeLandmarker(results=[FakeResult([landmarks])]))
    analyzer.load()

    result = analyzer.analyze(make_image(), timestamp=1.0)

    assert result.face_detected is True
    assert result.eyes_state == EyeState.UNKNOWN
    assert result.eye_metric is None


def test_analyze_multiple_faces_uses_first_only() -> None:
    open_landmarks = make_landmarks(left_points=OPEN_POINTS, right_points=OPEN_POINTS)
    closed_landmarks = make_landmarks(left_points=CLOSED_POINTS, right_points=CLOSED_POINTS)
    analyzer = make_analyzer(FakeLandmarker(results=[FakeResult([open_landmarks, closed_landmarks])]))
    analyzer.load()

    result = analyzer.analyze(make_image(), timestamp=1.0)

    assert result.eyes_state == EyeState.OPEN
    assert result.landmarks is open_landmarks


# --- analyze(): hysteresis persists across calls -------------------------------


def test_analyze_hysteresis_persists_across_consecutive_calls() -> None:
    open_landmarks = make_landmarks(left_points=OPEN_POINTS, right_points=OPEN_POINTS)
    mid_landmarks = make_landmarks(left_points=MID_ZONE_POINTS, right_points=MID_ZONE_POINTS)
    closed_landmarks = make_landmarks(left_points=CLOSED_POINTS, right_points=CLOSED_POINTS)

    fake = FakeLandmarker(
        results=[
            FakeResult([open_landmarks]),
            FakeResult([mid_landmarks]),  # dead zone -> retains OPEN
            FakeResult([closed_landmarks]),
            FakeResult([mid_landmarks]),  # dead zone -> retains CLOSED
        ]
    )
    analyzer = make_analyzer(fake)
    analyzer.load()

    states = [analyzer.analyze(make_image(), timestamp=float(i)).eyes_state for i in range(4)]

    assert states == [EyeState.OPEN, EyeState.OPEN, EyeState.CLOSED, EyeState.CLOSED]


def test_analyze_no_face_resets_hysteresis_state() -> None:
    open_landmarks = make_landmarks(left_points=OPEN_POINTS, right_points=OPEN_POINTS)
    mid_landmarks = make_landmarks(left_points=MID_ZONE_POINTS, right_points=MID_ZONE_POINTS)

    fake = FakeLandmarker(
        results=[
            FakeResult([open_landmarks]),
            FakeResult([]),  # face lost -> resets hysteresis memory to UNKNOWN
            FakeResult([mid_landmarks]),  # dead zone with no valid previous state -> OPEN
        ]
    )
    analyzer = make_analyzer(fake)
    analyzer.load()

    first = analyzer.analyze(make_image(), timestamp=1.0)
    second = analyzer.analyze(make_image(), timestamp=2.0)
    third = analyzer.analyze(make_image(), timestamp=3.0)

    assert first.eyes_state == EyeState.OPEN
    assert second.eyes_state == EyeState.UNKNOWN
    assert third.eyes_state == EyeState.OPEN


# --- analyze(): errors, latency, timestamps ------------------------------------


def test_analyze_inference_exception_raises_face_analysis_error() -> None:
    analyzer = make_analyzer(FakeLandmarker(raise_error=RuntimeError("backend crashed")))
    analyzer.load()

    with pytest.raises(FaceAnalysisError, match="Face landmark inference failed"):
        analyzer.analyze(make_image(), timestamp=1.0)

    assert analyzer.last_inference_ms is None


def test_analyze_malformed_image_raises_face_analysis_error() -> None:
    analyzer = make_analyzer(FakeLandmarker())
    analyzer.load()

    with pytest.raises(FaceAnalysisError, match="Face landmark inference failed"):
        analyzer.analyze(None, timestamp=1.0)  # type: ignore[arg-type]


def test_analyze_records_non_negative_inference_latency() -> None:
    analyzer = make_analyzer(FakeLandmarker(results=[FakeResult([])]))
    analyzer.load()

    assert analyzer.last_inference_ms is None

    analyzer.analyze(make_image(), timestamp=1.0)

    assert analyzer.last_inference_ms is not None
    assert analyzer.last_inference_ms >= 0.0


def test_analyze_uses_provided_timestamp_not_wall_clock() -> None:
    landmarks = make_landmarks(left_points=OPEN_POINTS, right_points=OPEN_POINTS)
    analyzer = make_analyzer(FakeLandmarker(results=[FakeResult([landmarks])]))
    analyzer.load()

    result = analyzer.analyze(make_image(), timestamp=12345.5)

    assert result.timestamp == 12345.5
