"""Tests for YOLODetector (FOCUSGUARD_PRD.md section 7).

All tests use fake model/result doubles shaped like Ultralytics' real API
(model(image, conf=..., device=..., verbose=...) -> list[Result], where
Result has .boxes.xyxy/.conf/.cls and .names). No real weights, inference,
or GPU are required.
"""

from __future__ import annotations

import pytest

from src.core.config_manager import YoloConfig
from src.detection.detection_types import Detection
from src.detection.yolo_detector import CELL_PHONE_CLASS_NAME, PERSON_CLASS_NAME, DetectionError, YOLODetector


def make_config(
    model: str = "models/yolo11n.pt",
    confidence: float = 0.45,
    phone_confidence: float = 0.55,
    device: str = "cpu",
) -> YoloConfig:
    return YoloConfig(model=model, confidence=confidence, phone_confidence=phone_confidence, device=device)


class FakeBoxes:
    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None, names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


DEFAULT_NAMES = {0: PERSON_CLASS_NAME, 1: CELL_PHONE_CLASS_NAME, 2: "chair"}


def make_result(entries: list[tuple[list[float], float, int]], names: dict[int, str] = DEFAULT_NAMES) -> FakeResult:
    if not entries:
        return FakeResult(FakeBoxes(xyxy=[], conf=[], cls=[]), names)
    xyxy = [e[0] for e in entries]
    conf = [e[1] for e in entries]
    cls = [e[2] for e in entries]
    return FakeResult(FakeBoxes(xyxy=xyxy, conf=conf, cls=cls), names)


class FakeModel:
    def __init__(self, results: list[FakeResult] | None = None, raise_error: Exception | None = None) -> None:
        self._results = results if results is not None else [make_result([])]
        self._raise_error = raise_error
        self.calls: list[dict] = []

    def __call__(self, source, conf, device, verbose=False):
        if self._raise_error is not None:
            raise self._raise_error
        self.calls.append({"source": source, "conf": conf, "device": device, "verbose": verbose})
        return self._results


def factory_for(model: FakeModel):
    def _factory(model_path: str) -> FakeModel:
        return model

    return _factory


# --- load() -----------------------------------------------------------------


def test_load_succeeds() -> None:
    detector = YOLODetector(make_config(), model_factory=factory_for(FakeModel()), cuda_available=lambda: False)

    detector.load()

    assert detector.is_loaded is True


def test_load_is_idempotent() -> None:
    fake = FakeModel()
    factory_calls: list[str] = []

    def factory(model_path: str) -> FakeModel:
        factory_calls.append(model_path)
        return fake

    detector = YOLODetector(make_config(), model_factory=factory, cuda_available=lambda: False)

    detector.load()
    detector.load()

    assert len(factory_calls) == 1


def test_load_wraps_factory_exception_in_detection_error() -> None:
    def failing_factory(model_path: str):
        raise OSError("model file not found")

    detector = YOLODetector(make_config(), model_factory=failing_factory, cuda_available=lambda: False)

    with pytest.raises(DetectionError, match="Failed to load YOLO model"):
        detector.load()

    assert detector.is_loaded is False


# --- device resolution --------------------------------------------------------


@pytest.mark.parametrize(
    "requested_device,cuda_available,expected",
    [
        ("cpu", True, "cpu"),
        ("cpu", False, "cpu"),
        ("cuda", True, "cuda"),
        ("cuda", False, "cpu"),  # graceful fallback
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
    ],
)
def test_device_resolution(requested_device: str, cuda_available: bool, expected: str) -> None:
    detector = YOLODetector(
        make_config(device=requested_device),
        model_factory=factory_for(FakeModel()),
        cuda_available=lambda: cuda_available,
    )

    assert detector.device == expected


# --- detect() -----------------------------------------------------------------


def test_detect_before_load_raises_detection_error() -> None:
    detector = YOLODetector(make_config(), model_factory=factory_for(FakeModel()), cuda_available=lambda: False)

    with pytest.raises(DetectionError, match="not loaded"):
        detector.detect(image="unused", timestamp=1.0)


def test_detect_filters_by_per_class_confidence_and_target_classes() -> None:
    result = make_result(
        [
            ([10.0, 20.0, 30.0, 40.0], 0.50, 0),  # person, passes (>= 0.45)
            ([50.0, 60.0, 70.0, 80.0], 0.50, 1),  # cell phone, fails (< 0.55)
            ([90.0, 100.0, 110.0, 120.0], 0.60, 1),  # cell phone, passes (>= 0.55)
            ([1.0, 2.0, 3.0, 4.0], 0.99, 2),  # chair, always discarded
        ]
    )
    fake_model = FakeModel(results=[result])
    detector = YOLODetector(make_config(), model_factory=factory_for(fake_model), cuda_available=lambda: False)
    detector.load()

    detections = detector.detect(image="frame", timestamp=42.0)

    assert len(detections) == 2
    assert detections[0] == Detection(
        class_name=PERSON_CLASS_NAME, confidence=0.50, x1=10.0, y1=20.0, x2=30.0, y2=40.0, timestamp=42.0
    )
    assert detections[1] == Detection(
        class_name=CELL_PHONE_CLASS_NAME, confidence=0.60, x1=90.0, y1=100.0, x2=110.0, y2=120.0, timestamp=42.0
    )


def test_detect_calls_model_with_min_of_both_thresholds() -> None:
    fake_model = FakeModel(results=[make_result([])])
    config = make_config(confidence=0.45, phone_confidence=0.55)
    detector = YOLODetector(config, model_factory=factory_for(fake_model), cuda_available=lambda: False)
    detector.load()

    detector.detect(image="frame", timestamp=1.0)

    assert fake_model.calls[0]["conf"] == pytest.approx(0.45)
    assert fake_model.calls[0]["device"] == "cpu"
    assert fake_model.calls[0]["verbose"] is False


def test_detect_returns_empty_list_when_no_detections() -> None:
    fake_model = FakeModel(results=[make_result([])])
    detector = YOLODetector(make_config(), model_factory=factory_for(fake_model), cuda_available=lambda: False)
    detector.load()

    detections = detector.detect(image="frame", timestamp=1.0)

    assert detections == []


def test_detect_raises_detection_error_on_inference_exception() -> None:
    fake_model = FakeModel(raise_error=RuntimeError("bad input shape"))
    detector = YOLODetector(make_config(), model_factory=factory_for(fake_model), cuda_available=lambda: False)
    detector.load()

    with pytest.raises(DetectionError, match="YOLO inference failed"):
        detector.detect(image="frame", timestamp=1.0)

    assert detector.last_inference_ms is None


def test_detect_records_non_negative_inference_latency() -> None:
    fake_model = FakeModel(results=[make_result([])])
    detector = YOLODetector(make_config(), model_factory=factory_for(fake_model), cuda_available=lambda: False)
    detector.load()

    assert detector.last_inference_ms is None

    detector.detect(image="frame", timestamp=1.0)

    assert detector.last_inference_ms is not None
    assert detector.last_inference_ms >= 0.0


def test_detect_uses_provided_timestamp_not_wall_clock() -> None:
    result = make_result([([0.0, 0.0, 1.0, 1.0], 0.9, 0)])
    fake_model = FakeModel(results=[result])
    detector = YOLODetector(make_config(), model_factory=factory_for(fake_model), cuda_available=lambda: False)
    detector.load()

    detections = detector.detect(image="frame", timestamp=999.5)

    assert all(d.timestamp == 999.5 for d in detections)
