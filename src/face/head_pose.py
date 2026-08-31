"""Approximate head orientation estimation (PRD section 13).

Pure, stateless helpers: yaw/pitch estimation from facial landmarks via
cv2.solvePnP, and threshold-based CENTER/LEFT/RIGHT/UP/DOWN classification.
This is an approximate attention-diversion signal, not gaze tracking, per
the PRD. No MediaPipe/camera/config dependency - callers only need objects
with .x/.y attributes (the same LandmarkPoint shape eye_metrics.py uses).

Landmark indices and 3D reference model
----------------------------------------
Six-point correspondence between MediaPipe's 468-point canonical Face Mesh
topology and a generic 3D face model, following the widely-documented
convention from OpenCV head-pose-estimation tutorials (originally
popularized for 68-point face models and commonly reused verbatim for
MediaPipe's topology, since both label the same anatomical points):

    MediaPipe index | Point                   | 3D model coordinate (mm)
    ----------------|-------------------------|---------------------------
    1               | Nose tip                | (   0.0,    0.0,    0.0)
    152             | Chin                    | (   0.0, -330.0,  -65.0)
    33              | Left eye, outer corner  | (-225.0,  170.0, -135.0)
    263             | Right eye, outer corner | ( 225.0,  170.0, -135.0)
    61              | Mouth, left corner      | (-150.0, -150.0, -125.0)
    291             | Mouth, right corner     | ( 150.0, -150.0, -125.0)

CAVEAT: this installed MediaPipe distribution does not ship a local
canonical-face-model reference file, so these indices could not be
mechanically cross-checked against Google's source in this environment -
they follow the standard, widely-documented mapping. A real-face visual
sanity check during an eventual hardware smoke test is recommended before
relying on this for real footage.

Camera intrinsics are approximated (no calibration exists anywhere in this
project): focal length = image width, principal point = image center, zero
lens distortion - a standard simplification for an uncalibrated webcam.

Sign convention (empirically verified, not assumed): using
cv2.projectPoints to synthesize 2D landmarks from a KNOWN 3D rotation of
the model above, then recovering that rotation via solvePnP + the Euler
extraction below, confirms:
    - pitch_degrees > 0  => head tilted DOWN (looking down)
    - pitch_degrees < 0  => head tilted UP (looking up)
    - yaw_degrees   > 0  => head turned toward the index-263 ("right eye
                            outer corner") side, classified RIGHT
    - yaw_degrees   < 0  => head turned toward the index-33 ("left eye
                            outer corner") side, classified LEFT
See tests/test_head_pose.py for the round-trip tests proving this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import cv2
import numpy as np

from src.face.eye_metrics import LandmarkPoint

NOSE_TIP_INDEX = 1
CHIN_INDEX = 152
LEFT_EYE_OUTER_INDEX = 33
RIGHT_EYE_OUTER_INDEX = 263
MOUTH_LEFT_INDEX = 61
MOUTH_RIGHT_INDEX = 291

LANDMARK_INDICES: tuple[int, int, int, int, int, int] = (
    NOSE_TIP_INDEX,
    CHIN_INDEX,
    LEFT_EYE_OUTER_INDEX,
    RIGHT_EYE_OUTER_INDEX,
    MOUTH_LEFT_INDEX,
    MOUTH_RIGHT_INDEX,
)

MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)

_SINGULAR_EPSILON = 1e-6


class HeadOrientation(Enum):
    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class HeadPoseResult:
    orientation: HeadOrientation
    yaw_degrees: float | None
    pitch_degrees: float | None


def _camera_matrix(image_width: int, image_height: int) -> np.ndarray:
    focal_length = float(image_width)
    center_x, center_y = image_width / 2.0, image_height / 2.0
    return np.array(
        [
            [focal_length, 0.0, center_x],
            [0.0, focal_length, center_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_to_yaw_pitch(rotation_matrix: np.ndarray) -> tuple[float, float]:
    """Extract yaw (about Y) and pitch (about X) in degrees from a 3x3
    rotation matrix. See module docstring for the verified sign convention."""
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    if sy < _SINGULAR_EPSILON:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
    else:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
    return math.degrees(yaw), math.degrees(pitch)


def classify_orientation(
    yaw_degrees: float,
    pitch_degrees: float,
    yaw_threshold_degrees: float,
    pitch_threshold_degrees: float,
) -> HeadOrientation:
    """Classify raw yaw/pitch into CENTER/LEFT/RIGHT/UP/DOWN.

    PRD section 13 defines only these five directional states (plus
    UNKNOWN) - no combined "upper-left" label exists - so when both axes
    simultaneously exceed their threshold, whichever axis is
    proportionally further past its own threshold wins. This is a
    recommended engineering tie-break, not something the PRD specifies.
    """
    yaw_ratio = abs(yaw_degrees) / yaw_threshold_degrees if yaw_threshold_degrees > 0 else math.inf
    pitch_ratio = abs(pitch_degrees) / pitch_threshold_degrees if pitch_threshold_degrees > 0 else math.inf

    yaw_exceeded = yaw_ratio >= 1.0
    pitch_exceeded = pitch_ratio >= 1.0

    if not yaw_exceeded and not pitch_exceeded:
        return HeadOrientation.CENTER

    if yaw_exceeded and (not pitch_exceeded or yaw_ratio >= pitch_ratio):
        return HeadOrientation.RIGHT if yaw_degrees > 0 else HeadOrientation.LEFT

    return HeadOrientation.DOWN if pitch_degrees > 0 else HeadOrientation.UP


def estimate_head_pose(
    landmarks: Sequence[LandmarkPoint],
    image_width: int,
    image_height: int,
    yaw_threshold_degrees: float,
    pitch_threshold_degrees: float,
) -> HeadPoseResult:
    """Estimate approximate yaw/pitch from facial landmarks via solvePnP and
    classify the result. Never raises for bad/degenerate input - returns
    UNKNOWN with yaw/pitch=None instead, mirroring
    eye_metrics.compute_ear's None-on-degenerate-input behavior.
    """
    try:
        image_points = np.array(
            [(landmarks[i].x * image_width, landmarks[i].y * image_height) for i in LANDMARK_INDICES],
            dtype=np.float64,
        )
    except (IndexError, AttributeError, TypeError):
        return HeadPoseResult(orientation=HeadOrientation.UNKNOWN, yaw_degrees=None, pitch_degrees=None)

    if not np.all(np.isfinite(image_points)):
        return HeadPoseResult(orientation=HeadOrientation.UNKNOWN, yaw_degrees=None, pitch_degrees=None)

    camera_matrix = _camera_matrix(image_width, image_height)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    try:
        success, rotation_vector, _translation_vector = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return HeadPoseResult(orientation=HeadOrientation.UNKNOWN, yaw_degrees=None, pitch_degrees=None)

    if not success:
        return HeadPoseResult(orientation=HeadOrientation.UNKNOWN, yaw_degrees=None, pitch_degrees=None)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    yaw_degrees, pitch_degrees = _rotation_matrix_to_yaw_pitch(rotation_matrix)

    if not (math.isfinite(yaw_degrees) and math.isfinite(pitch_degrees)):
        return HeadPoseResult(orientation=HeadOrientation.UNKNOWN, yaw_degrees=None, pitch_degrees=None)

    orientation = classify_orientation(yaw_degrees, pitch_degrees, yaw_threshold_degrees, pitch_threshold_degrees)
    return HeadPoseResult(orientation=orientation, yaw_degrees=yaw_degrees, pitch_degrees=pitch_degrees)
