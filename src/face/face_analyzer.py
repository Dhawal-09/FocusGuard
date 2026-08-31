"""Facial landmark detection and face/eye analysis (PRD section 10/11).

FaceAnalyzer owns only face/eye analysis: loading the MediaPipe Face
Landmarker model, running single-frame inference, and deriving an
eye-openness classification. It is completely independent of CameraManager
and YOLODetector - it accepts raw (image, timestamp) primitives and never
imports either module, and never performs primary-person selection (that is
a later perception/aggregation-layer concern).

Head orientation is explicitly out of scope here; see src/face/head_pose.py
for a later phase.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import cv2
import numpy as np
from mediapipe import Image as MPImage
from mediapipe import ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from src.core.config_manager import EyesConfig, FaceConfig
from src.face.eye_metrics import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    EyeState,
    LandmarkPoint,
    classify_eye_state,
    combine_eye_metrics,
    compute_ear,
    extract_eye_pixel_points,
)


class FaceAnalysisError(Exception):
    """Raised for face model load/inference failures with a human-readable message."""


class FaceLandmarkerResultLike(Protocol):
    face_landmarks: list[list[LandmarkPoint]]


class FaceLandmarkerLike(Protocol):
    def detect(self, image: Any) -> FaceLandmarkerResultLike: ...


LandmarkerFactory = Callable[[str], FaceLandmarkerLike]


def _default_landmarker_factory(model_path: str) -> FaceLandmarkerLike:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


@dataclass(frozen=True)
class FaceAnalysisResult:
    """Result of analyzing a single frame. Landmarks are exposed (not just
    the derived eye metric) so a future head-pose phase can reuse this same
    inference pass instead of running Face Landmarker a second time."""

    face_detected: bool
    eyes_state: EyeState
    eye_metric: float | None
    timestamp: float
    landmarks: list[LandmarkPoint] | None = None


class FaceAnalyzer:
    """Loads a MediaPipe Face Landmarker model and runs single-frame face/eye analysis.

    The landmarker backend is injectable so this class is fully unit-testable
    without a real model, a webcam, or a GPU.
    """

    def __init__(
        self,
        face_config: FaceConfig,
        eyes_config: EyesConfig,
        landmarker_factory: LandmarkerFactory = _default_landmarker_factory,
    ) -> None:
        self._face_config = face_config
        self._eyes_config = eyes_config
        self._landmarker_factory = landmarker_factory
        self._landmarker: FaceLandmarkerLike | None = None
        self._previous_eye_state: EyeState = EyeState.UNKNOWN
        self._last_inference_ms: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self._landmarker is not None

    @property
    def last_inference_ms(self) -> float | None:
        """Wall-clock duration of the most recent successful analyze() call."""
        return self._last_inference_ms

    def load(self) -> None:
        """Load the configured face landmark model. No-op if already loaded."""
        if self._landmarker is not None:
            return
        try:
            self._landmarker = self._landmarker_factory(self._face_config.model)
        except Exception as exc:
            raise FaceAnalysisError(
                f"Failed to load face landmark model from {self._face_config.model!r}: {exc}"
            ) from exc

    def analyze(self, image: np.ndarray, timestamp: float) -> FaceAnalysisResult:
        """Run inference on a single BGR frame and return a FaceAnalysisResult.

        Never classifies a missing face/landmarks as CLOSED - both surface as
        UNKNOWN, per PRD section 10.
        """
        if self._landmarker is None:
            raise FaceAnalysisError("Model is not loaded. Call load() before analyze().")

        start = time.perf_counter()
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb_image)
            result = self._landmarker.detect(mp_image)
        except Exception as exc:
            raise FaceAnalysisError(f"Face landmark inference failed: {exc}") from exc
        self._last_inference_ms = (time.perf_counter() - start) * 1000.0

        if not result.face_landmarks:
            self._previous_eye_state = EyeState.UNKNOWN
            return FaceAnalysisResult(
                face_detected=False,
                eyes_state=EyeState.UNKNOWN,
                eye_metric=None,
                timestamp=timestamp,
            )

        landmarks = result.face_landmarks[0]  # num_faces=1: at most one face
        image_height, image_width = image.shape[0], image.shape[1]

        left_points = extract_eye_pixel_points(landmarks, LEFT_EYE_INDICES, image_width, image_height)
        right_points = extract_eye_pixel_points(landmarks, RIGHT_EYE_INDICES, image_width, image_height)
        left_ear = compute_ear(left_points)
        right_ear = compute_ear(right_points)
        eye_metric = combine_eye_metrics(left_ear, right_ear)

        if eye_metric is None:
            self._previous_eye_state = EyeState.UNKNOWN
            return FaceAnalysisResult(
                face_detected=True,
                eyes_state=EyeState.UNKNOWN,
                eye_metric=None,
                timestamp=timestamp,
                landmarks=landmarks,
            )

        eyes_state = classify_eye_state(
            eye_metric,
            self._previous_eye_state,
            self._eyes_config.closed_threshold,
            self._eyes_config.open_threshold,
        )
        self._previous_eye_state = eyes_state

        return FaceAnalysisResult(
            face_detected=True,
            eyes_state=eyes_state,
            eye_metric=eye_metric,
            timestamp=timestamp,
            landmarks=landmarks,
        )
