"""Eye-openness metric calculation (PRD section 11).

Pure, stateless helpers: Eye Aspect Ratio (EAR) computation from facial
landmarks, combining both eyes into one metric, and hysteresis-based
OPEN/CLOSED/UNKNOWN classification. None of this depends on MediaPipe
itself - callers only need to supply objects with .x/.y attributes.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Protocol, Sequence

# Standard MediaPipe Face Mesh (468-point topology) 6-point eye subsets, each
# ordered (corner, top, top, corner, bottom, bottom) to match the classic EAR
# formula: EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|).
LEFT_EYE_INDICES: tuple[int, int, int, int, int, int] = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_INDICES: tuple[int, int, int, int, int, int] = (33, 160, 158, 133, 153, 144)

_DEGENERATE_DISTANCE_EPSILON = 1e-6


class EyeState(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


class LandmarkPoint(Protocol):
    x: float
    y: float


def extract_eye_pixel_points(
    landmarks: Sequence[LandmarkPoint],
    indices: Sequence[int],
    image_width: int,
    image_height: int,
) -> list[tuple[float, float]]:
    """Convert normalized landmark coordinates to pixel space for the given indices.

    Converting to pixel space (rather than computing distances directly on
    normalized [0,1] coordinates) avoids aspect-ratio distortion when image
    width and height differ.
    """
    return [(landmarks[i].x * image_width, landmarks[i].y * image_height) for i in indices]


def compute_ear(points: Sequence[tuple[float, float]]) -> float | None:
    """Compute the Eye Aspect Ratio from 6 (x, y) pixel-space points.

    Returns None if the eye-corner distance is degenerate (~0), which would
    otherwise divide by zero - this marks the eye as unusable rather than
    raising or returning an invalid value.
    """
    if len(points) != 6:
        raise ValueError(f"compute_ear requires exactly 6 points, got {len(points)}")

    p1, p2, p3, p4, p5, p6 = points
    horizontal = math.hypot(p4[0] - p1[0], p4[1] - p1[1])
    if horizontal < _DEGENERATE_DISTANCE_EPSILON:
        return None

    vertical = math.hypot(p2[0] - p6[0], p2[1] - p6[1]) + math.hypot(p3[0] - p5[0], p3[1] - p5[1])
    return vertical / (2.0 * horizontal)


def combine_eye_metrics(left_ear: float | None, right_ear: float | None) -> float | None:
    """Combine both eyes into a single metric: average if both usable, else
    fall back to whichever single eye is usable, else None."""
    if left_ear is not None and right_ear is not None:
        return (left_ear + right_ear) / 2.0
    if left_ear is not None:
        return left_ear
    if right_ear is not None:
        return right_ear
    return None


def classify_eye_state(
    metric: float,
    previous_state: EyeState,
    closed_threshold: float,
    open_threshold: float,
) -> EyeState:
    """Hysteresis classification per PRD section 11.

    Values between the thresholds retain the previous state; if there is no
    valid previous state yet (UNKNOWN), the dead zone defaults to OPEN.
    """
    if metric < closed_threshold:
        return EyeState.CLOSED
    if metric > open_threshold:
        return EyeState.OPEN
    if previous_state == EyeState.UNKNOWN:
        return EyeState.OPEN
    return previous_state
