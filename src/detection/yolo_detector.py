"""YOLO-based person/phone object detection (PRD section 7).

YOLODetector owns only object detection: loading the pretrained YOLO model,
resolving the CPU/CUDA inference device (with automatic CPU fallback), running
inference on a single supplied frame, and returning filtered person/cell-phone
detections with a measured inference latency.

It never decides *when* to run inference (that is later main-loop pacing
policy), tracks state across frames, or performs any downstream analysis such
as primary-person selection or temporal confirmation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np
import torch
from ultralytics import YOLO

from src.core.config_manager import YoloConfig
from src.detection.detection_types import Detection

PERSON_CLASS_NAME = "person"
CELL_PHONE_CLASS_NAME = "cell phone"


class DetectionError(Exception):
    """Raised for model load/inference failures with a human-readable message."""


class BoxesLike(Protocol):
    xyxy: Any
    conf: Any
    cls: Any


class ResultLike(Protocol):
    boxes: BoxesLike | None
    names: dict[int, str]


class ModelLike(Protocol):
    def __call__(
        self, source: np.ndarray, conf: float, device: str, verbose: bool
    ) -> list[ResultLike]: ...


ModelFactory = Callable[[str], ModelLike]


def _default_model_factory(model_path: str) -> ModelLike:
    return YOLO(model_path)


class YOLODetector:
    """Loads a YOLO model and runs single-frame inference for person/cell phone.

    The model backend and CUDA-availability check are both injectable so this
    class is fully unit-testable without real weights or a GPU.
    """

    def __init__(
        self,
        config: YoloConfig,
        model_factory: ModelFactory = _default_model_factory,
        cuda_available: Callable[[], bool] = torch.cuda.is_available,
    ) -> None:
        self._config = config
        self._model_factory = model_factory
        self._cuda_available = cuda_available
        self._model: ModelLike | None = None
        self._device = self._resolve_device(config.device)
        self._last_inference_ms: float | None = None

    @property
    def device(self) -> str:
        """The resolved inference device ("cpu" or "cuda"), never "auto"."""
        return self._device

    @property
    def last_inference_ms(self) -> float | None:
        """Wall-clock duration of the most recent successful detect() call."""
        return self._last_inference_ms

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _resolve_device(self, requested: str) -> str:
        if requested == "cpu":
            return "cpu"
        if requested in ("cuda", "auto"):
            return "cuda" if self._cuda_available() else "cpu"
        raise DetectionError(f"Unsupported yolo.device: {requested!r}")

    def load(self) -> None:
        """Load the configured model. No-op if already loaded."""
        if self._model is not None:
            return
        try:
            self._model = self._model_factory(self._config.model)
        except Exception as exc:
            raise DetectionError(
                f"Failed to load YOLO model from {self._config.model!r}: {exc}"
            ) from exc

    def detect(self, image: np.ndarray, timestamp: float) -> list[Detection]:
        """Run inference on a single frame and return filtered detections.

        ``timestamp`` should be the frame's own capture timestamp (e.g. from
        CameraManager's Frame), not the time inference happened to run.
        """
        if self._model is None:
            raise DetectionError("Model is not loaded. Call load() before detect().")

        min_confidence = min(self._config.confidence, self._config.phone_confidence)

        start = time.perf_counter()
        try:
            results = self._model(image, conf=min_confidence, device=self._device, verbose=False)
        except Exception as exc:
            raise DetectionError(f"YOLO inference failed: {exc}") from exc
        self._last_inference_ms = (time.perf_counter() - start) * 1000.0

        return self._filter_results(results, timestamp)

    def _filter_results(self, results: list[ResultLike], timestamp: float) -> list[Detection]:
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            names = result.names
            for xyxy, conf, cls_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
                class_name = names[int(cls_id)]
                confidence = float(conf)

                if class_name == PERSON_CLASS_NAME:
                    if confidence < self._config.confidence:
                        continue
                elif class_name == CELL_PHONE_CLASS_NAME:
                    if confidence < self._config.phone_confidence:
                        continue
                else:
                    continue

                x1, y1, x2, y2 = (float(v) for v in xyxy)
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=confidence,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        timestamp=timestamp,
                    )
                )
        return detections
