"""Tests for PerceptionSnapshot assembly and VisionQuality classification
(FOCUSGUARD_PRD.md section 16).

Fully deterministic: synthetic Detection lists and face/head keyword
arguments only. No webcam, GPU, YOLO, or MediaPipe model.
"""

from __future__ import annotations

from src.core.types import VisionQuality, build_perception_snapshot
from src.detection.detection_types import Detection
from src.face.eye_metrics import EyeState
from src.face.head_pose import HeadOrientation

PERSON = "person"
PHONE = "cell phone"


def person(x1: float, y1: float, x2: float, y2: float, confidence: float = 0.9) -> Detection:
    return Detection(class_name=PERSON, confidence=confidence, x1=x1, y1=y1, x2=x2, y2=y2, timestamp=0.0)


def phone(x1: float, y1: float, x2: float, y2: float, confidence: float = 0.9) -> Detection:
    return Detection(class_name=PHONE, confidence=confidence, x1=x1, y1=y1, x2=x2, y2=y2, timestamp=0.0)


# --- Defaults / no detections -----------------------------------------------


def test_no_detections_yields_no_person_and_no_phone() -> None:
    snapshot = build_perception_snapshot(1.0, [])

    assert snapshot.person_present is False
    assert snapshot.primary_person is None
    assert snapshot.phone_detected is False
    assert snapshot.phone_confidence is None


def test_no_face_kwargs_default_to_unknown_not_closed() -> None:
    """Missing face analysis must never be interpreted as eyes closed
    (PRD section 10)."""
    snapshot = build_perception_snapshot(1.0, [])

    assert snapshot.face_present is False
    assert snapshot.eyes_state == EyeState.UNKNOWN
    assert snapshot.eye_metric is None
    assert snapshot.head_orientation == HeadOrientation.UNKNOWN
    assert snapshot.head_yaw is None
    assert snapshot.head_pitch is None


# --- Person / phone field population ----------------------------------------


def test_person_present_and_primary_person_populated() -> None:
    only = person(0, 0, 100, 100)

    snapshot = build_perception_snapshot(1.0, [only])

    assert snapshot.person_present is True
    assert snapshot.primary_person is only


def test_phone_detected_and_confidence_populated() -> None:
    primary = person(0, 0, 100, 100)
    the_phone = phone(40, 40, 60, 60, confidence=0.77)

    snapshot = build_perception_snapshot(1.0, [primary, the_phone])

    assert snapshot.phone_detected is True
    assert snapshot.phone_confidence == 0.77


def test_timestamp_propagates() -> None:
    snapshot = build_perception_snapshot(42.5, [])

    assert snapshot.timestamp == 42.5


def test_face_eye_head_kwargs_propagate_verbatim() -> None:
    snapshot = build_perception_snapshot(
        1.0,
        [person(0, 0, 100, 100)],
        face_present=True,
        eyes_state=EyeState.CLOSED,
        eye_metric=0.15,
        head_orientation=HeadOrientation.LEFT,
        head_yaw=-25.0,
        head_pitch=3.0,
    )

    assert snapshot.face_present is True
    assert snapshot.eyes_state == EyeState.CLOSED
    assert snapshot.eye_metric == 0.15
    assert snapshot.head_orientation == HeadOrientation.LEFT
    assert snapshot.head_yaw == -25.0
    assert snapshot.head_pitch == 3.0


# --- VisionQuality: NO_PERSON dominates -------------------------------------


def test_no_person_present_is_no_person_regardless_of_face_data() -> None:
    """Person presence (from YOLO) is authoritative for NO_PERSON, even in
    the edge case where face data was somehow supplied without a
    corresponding person detection."""
    snapshot = build_perception_snapshot(
        1.0,
        [],  # no person detections at all
        face_present=True,
        eyes_state=EyeState.OPEN,
        head_orientation=HeadOrientation.CENTER,
    )

    assert snapshot.vision_quality == VisionQuality.NO_PERSON


# --- VisionQuality: DEGRADED (person present, face-derived data missing) ---


def test_person_present_but_no_face_is_degraded() -> None:
    snapshot = build_perception_snapshot(1.0, [person(0, 0, 100, 100)], face_present=False)

    assert snapshot.vision_quality == VisionQuality.DEGRADED


def test_person_present_face_present_but_eyes_unknown_is_degraded() -> None:
    snapshot = build_perception_snapshot(
        1.0,
        [person(0, 0, 100, 100)],
        face_present=True,
        eyes_state=EyeState.UNKNOWN,
        head_orientation=HeadOrientation.CENTER,
    )

    assert snapshot.vision_quality == VisionQuality.DEGRADED


def test_person_present_face_present_but_head_unknown_is_degraded() -> None:
    snapshot = build_perception_snapshot(
        1.0,
        [person(0, 0, 100, 100)],
        face_present=True,
        eyes_state=EyeState.OPEN,
        head_orientation=HeadOrientation.UNKNOWN,
    )

    assert snapshot.vision_quality == VisionQuality.DEGRADED


# --- VisionQuality: GOOD -----------------------------------------------------


def test_person_present_face_present_eyes_and_head_known_is_good() -> None:
    snapshot = build_perception_snapshot(
        1.0,
        [person(0, 0, 100, 100)],
        face_present=True,
        eyes_state=EyeState.OPEN,
        head_orientation=HeadOrientation.CENTER,
    )

    assert snapshot.vision_quality == VisionQuality.GOOD


def test_good_with_eyes_closed_is_still_good() -> None:
    """A known CLOSED eye state is fully reliable data, not degraded -
    degraded means *unknown*, not merely 'not open'."""
    snapshot = build_perception_snapshot(
        1.0,
        [person(0, 0, 100, 100)],
        face_present=True,
        eyes_state=EyeState.CLOSED,
        head_orientation=HeadOrientation.CENTER,
    )

    assert snapshot.vision_quality == VisionQuality.GOOD


def test_good_with_off_center_head_is_still_good() -> None:
    """A known off-center orientation is fully reliable data, not
    degraded."""
    snapshot = build_perception_snapshot(
        1.0,
        [person(0, 0, 100, 100)],
        face_present=True,
        eyes_state=EyeState.OPEN,
        head_orientation=HeadOrientation.LEFT,
    )

    assert snapshot.vision_quality == VisionQuality.GOOD
