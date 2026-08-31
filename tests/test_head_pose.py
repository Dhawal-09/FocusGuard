"""Tests for head_pose.py (FOCUSGUARD_PRD.md section 13).

Sign-convention correctness is verified via a round-trip: synthesize 2D
landmarks from a KNOWN 3D rotation using cv2.projectPoints (the exact
inverse of solvePnP), then confirm estimate_head_pose recovers that same
signed rotation. No real webcam, MediaPipe model, GPU, or network required.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.face.head_pose import (
    LANDMARK_INDICES,
    MODEL_POINTS_3D,
    HeadOrientation,
    classify_orientation,
    estimate_head_pose,
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
YAW_THRESHOLD = 20.0
PITCH_THRESHOLD = 18.0


class FakeLandmark:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def _camera_matrix() -> np.ndarray:
    f = float(IMAGE_WIDTH)
    return np.array(
        [[f, 0.0, IMAGE_WIDTH / 2.0], [0.0, f, IMAGE_HEIGHT / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def make_landmarks_for_rotation(
    rotation_degrees_xyz: tuple[float, float, float],
    translation: tuple[float, float, float] = (0.0, 0.0, 800.0),
) -> list[FakeLandmark]:
    """Project MODEL_POINTS_3D under a known rotation/translation to 2D
    image points via cv2.projectPoints (the inverse of solvePnP), then wrap
    them as landmarks at the exact LANDMARK_INDICES positions
    estimate_head_pose reads from. This is how sign conventions are
    verified: the ground-truth rotation is controlled exactly, and the
    test asserts the pipeline recovers it (magnitude and sign)."""
    rvec = np.radians(np.array(rotation_degrees_xyz, dtype=np.float64)).reshape(3, 1)
    tvec = np.array(translation, dtype=np.float64).reshape(3, 1)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    image_points, _ = cv2.projectPoints(MODEL_POINTS_3D, rvec, tvec, _camera_matrix(), dist_coeffs)
    image_points = image_points.reshape(-1, 2)

    max_index = max(LANDMARK_INDICES)
    landmarks = [FakeLandmark(0.5, 0.5) for _ in range(max_index + 1)]
    for idx, (px, py) in zip(LANDMARK_INDICES, image_points):
        landmarks[idx] = FakeLandmark(px / IMAGE_WIDTH, py / IMAGE_HEIGHT)
    return landmarks


# --- Sign convention: verified via known-rotation round-trip -------------------


def test_frontal_pose_is_center_with_near_zero_yaw_pitch() -> None:
    landmarks = make_landmarks_for_rotation((0.0, 0.0, 0.0))

    result = estimate_head_pose(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, YAW_THRESHOLD, PITCH_THRESHOLD)

    assert result.orientation == HeadOrientation.CENTER
    assert result.yaw_degrees == pytest.approx(0.0, abs=1e-2)
    assert result.pitch_degrees == pytest.approx(0.0, abs=1e-2)


def test_positive_x_rotation_yields_positive_pitch_and_classifies_down() -> None:
    """Empirically verified: +X rotation tilts the model's forward vector
    toward -Y (down) -> positive pitch -> DOWN."""
    landmarks = make_landmarks_for_rotation((25.0, 0.0, 0.0))

    result = estimate_head_pose(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, YAW_THRESHOLD, PITCH_THRESHOLD)

    assert result.pitch_degrees == pytest.approx(25.0, abs=1e-1)
    assert result.orientation == HeadOrientation.DOWN


def test_negative_x_rotation_yields_negative_pitch_and_classifies_up() -> None:
    landmarks = make_landmarks_for_rotation((-25.0, 0.0, 0.0))

    result = estimate_head_pose(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, YAW_THRESHOLD, PITCH_THRESHOLD)

    assert result.pitch_degrees == pytest.approx(-25.0, abs=1e-1)
    assert result.orientation == HeadOrientation.UP


def test_positive_y_rotation_yields_positive_yaw_and_classifies_right() -> None:
    """Empirically verified: +Y rotation swings the forward vector toward
    the index-263 ("right eye outer corner") side -> positive yaw -> RIGHT."""
    landmarks = make_landmarks_for_rotation((0.0, 25.0, 0.0))

    result = estimate_head_pose(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, YAW_THRESHOLD, PITCH_THRESHOLD)

    assert result.yaw_degrees == pytest.approx(25.0, abs=1e-1)
    assert result.orientation == HeadOrientation.RIGHT


def test_negative_y_rotation_yields_negative_yaw_and_classifies_left() -> None:
    landmarks = make_landmarks_for_rotation((0.0, -25.0, 0.0))

    result = estimate_head_pose(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, YAW_THRESHOLD, PITCH_THRESHOLD)

    assert result.yaw_degrees == pytest.approx(-25.0, abs=1e-1)
    assert result.orientation == HeadOrientation.LEFT


def test_small_rotation_within_threshold_still_classifies_center() -> None:
    landmarks = make_landmarks_for_rotation((5.0, 5.0, 0.0))

    result = estimate_head_pose(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, YAW_THRESHOLD, PITCH_THRESHOLD)

    assert result.orientation == HeadOrientation.CENTER


# --- estimate_head_pose: degenerate/invalid input -------------------------------


def test_estimate_head_pose_returns_unknown_for_too_short_landmark_list() -> None:
    landmarks = [FakeLandmark(0.5, 0.5) for _ in range(10)]  # shorter than max index needed

    result = estimate_head_pose(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, YAW_THRESHOLD, PITCH_THRESHOLD)

    assert result.orientation == HeadOrientation.UNKNOWN
    assert result.yaw_degrees is None
    assert result.pitch_degrees is None


def test_estimate_head_pose_returns_unknown_for_all_identical_points() -> None:
    max_index = max(LANDMARK_INDICES)
    landmarks = [FakeLandmark(0.5, 0.5) for _ in range(max_index + 1)]  # degenerate: all coincident

    result = estimate_head_pose(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, YAW_THRESHOLD, PITCH_THRESHOLD)

    assert result.orientation == HeadOrientation.UNKNOWN
    assert result.yaw_degrees is None
    assert result.pitch_degrees is None


def test_estimate_head_pose_never_raises_on_malformed_landmarks() -> None:
    class BadLandmark:
        pass  # no .x/.y at all

    max_index = max(LANDMARK_INDICES)
    landmarks = [BadLandmark() for _ in range(max_index + 1)]

    result = estimate_head_pose(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, YAW_THRESHOLD, PITCH_THRESHOLD)

    assert result.orientation == HeadOrientation.UNKNOWN


def test_estimate_head_pose_never_raises_on_empty_landmarks() -> None:
    result = estimate_head_pose([], IMAGE_WIDTH, IMAGE_HEIGHT, YAW_THRESHOLD, PITCH_THRESHOLD)

    assert result.orientation == HeadOrientation.UNKNOWN
    assert result.yaw_degrees is None
    assert result.pitch_degrees is None


# --- classify_orientation: pure boundary tests ----------------------------------


def test_classify_center_when_both_within_threshold() -> None:
    assert classify_orientation(5.0, 5.0, YAW_THRESHOLD, PITCH_THRESHOLD) == HeadOrientation.CENTER


def test_classify_exactly_at_yaw_threshold_is_not_center() -> None:
    assert classify_orientation(YAW_THRESHOLD, 0.0, YAW_THRESHOLD, PITCH_THRESHOLD) == HeadOrientation.RIGHT


def test_classify_just_under_yaw_threshold_is_center() -> None:
    result = classify_orientation(YAW_THRESHOLD - 0.01, 0.0, YAW_THRESHOLD, PITCH_THRESHOLD)
    assert result == HeadOrientation.CENTER


def test_classify_exactly_at_pitch_threshold_is_not_center() -> None:
    assert classify_orientation(0.0, PITCH_THRESHOLD, YAW_THRESHOLD, PITCH_THRESHOLD) == HeadOrientation.DOWN


def test_classify_just_under_pitch_threshold_is_center() -> None:
    result = classify_orientation(0.0, PITCH_THRESHOLD - 0.01, YAW_THRESHOLD, PITCH_THRESHOLD)
    assert result == HeadOrientation.CENTER


def test_classify_negative_yaw_is_left() -> None:
    assert classify_orientation(-30.0, 0.0, YAW_THRESHOLD, PITCH_THRESHOLD) == HeadOrientation.LEFT


def test_classify_positive_yaw_is_right() -> None:
    assert classify_orientation(30.0, 0.0, YAW_THRESHOLD, PITCH_THRESHOLD) == HeadOrientation.RIGHT


def test_classify_negative_pitch_is_up() -> None:
    assert classify_orientation(0.0, -30.0, YAW_THRESHOLD, PITCH_THRESHOLD) == HeadOrientation.UP


def test_classify_positive_pitch_is_down() -> None:
    assert classify_orientation(0.0, 30.0, YAW_THRESHOLD, PITCH_THRESHOLD) == HeadOrientation.DOWN


def test_classify_both_exceeded_yaw_proportionally_larger_wins() -> None:
    result = classify_orientation(
        yaw_degrees=YAW_THRESHOLD * 2,
        pitch_degrees=PITCH_THRESHOLD * 1.1,
        yaw_threshold_degrees=YAW_THRESHOLD,
        pitch_threshold_degrees=PITCH_THRESHOLD,
    )
    assert result == HeadOrientation.RIGHT


def test_classify_both_exceeded_pitch_proportionally_larger_wins() -> None:
    result = classify_orientation(
        yaw_degrees=YAW_THRESHOLD * 1.1,
        pitch_degrees=PITCH_THRESHOLD * 2,
        yaw_threshold_degrees=YAW_THRESHOLD,
        pitch_threshold_degrees=PITCH_THRESHOLD,
    )
    assert result == HeadOrientation.DOWN
