"""Shared core data structures: the central per-frame PerceptionSnapshot
(PRD section 16).

PerceptionSnapshot aggregates the current frame's raw detector/analyzer
output (YOLO detections, FaceAnalyzer result, head-pose result) into one
immutable structure. It performs no temporal filtering itself - the state
machine consumes stable/filtered signals derived FROM a sequence of
snapshots (see src/state/), not the snapshot fields directly (PRD section
16: "the state machine consumes stable/filtered signals rather than raw
detector output").

build_perception_snapshot() is the single assembly point: it applies
primary-person selection and phone-person association (PRD section 8, via
src.detection.primary_person) and computes VisionQuality deterministically
from the assembled fields, so no call site has to re-derive that logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.detection.detection_types import Detection
from src.detection.primary_person import select_associated_phone, select_primary_person
from src.face.eye_metrics import EyeState
from src.face.head_pose import HeadOrientation


class VisionQuality(Enum):
    """Deterministic classification of how much a snapshot's perception can
    be trusted for state evaluation.

    NO_PERSON: no primary person detected this frame - face/eye/head
    signals are meaningless without a person, and this is the case that
    should lead toward AWAY (via a person-away temporal filter), never
    UNKNOWN.

    DEGRADED: a person is present but face-derived signals could not be
    reliably obtained this frame (no face detected, or eyes/head could not
    be classified). This - and only this - is the case that should lead
    toward UNKNOWN: a person is present but required face-derived
    perception is unavailable.

    GOOD: a person is present and face, eyes, and head were all reliably
    read this frame.
    """

    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    NO_PERSON = "NO_PERSON"


@dataclass(frozen=True)
class PerceptionSnapshot:
    """Immutable snapshot of one frame's perception (PRD section 16)."""

    timestamp: float
    person_present: bool
    primary_person: Detection | None
    phone_detected: bool
    phone_confidence: float | None
    face_present: bool
    eyes_state: EyeState
    eye_metric: float | None
    head_orientation: HeadOrientation
    head_yaw: float | None
    head_pitch: float | None
    vision_quality: VisionQuality


def _compute_vision_quality(
    person_present: bool,
    face_present: bool,
    eyes_state: EyeState,
    head_orientation: HeadOrientation,
) -> VisionQuality:
    if not person_present:
        return VisionQuality.NO_PERSON
    if not face_present or eyes_state == EyeState.UNKNOWN or head_orientation == HeadOrientation.UNKNOWN:
        return VisionQuality.DEGRADED
    return VisionQuality.GOOD


def build_perception_snapshot(
    timestamp: float,
    detections: list[Detection],
    *,
    face_present: bool = False,
    eyes_state: EyeState = EyeState.UNKNOWN,
    eye_metric: float | None = None,
    head_orientation: HeadOrientation = HeadOrientation.UNKNOWN,
    head_yaw: float | None = None,
    head_pitch: float | None = None,
) -> PerceptionSnapshot:
    """Assemble a PerceptionSnapshot from one frame's raw detector/analyzer
    output.

    `detections` is the full YOLO detection list for the frame (person and
    cell-phone boxes); primary-person selection and phone association (PRD
    section 8) are applied internally. The face/eye/head keyword arguments
    are the corresponding fields already produced by FaceAnalyzer /
    head_pose.estimate_head_pose for this same frame - callers that skip
    face analysis for a frame simply omit them, which defaults to the same
    "unknown" values FaceAnalyzer itself returns when a face cannot be
    reliably analyzed (PRD section 10).
    """
    primary_person = select_primary_person(detections)
    person_present = primary_person is not None

    associated_phone = select_associated_phone(detections, primary_person)
    phone_detected = associated_phone is not None
    phone_confidence = associated_phone.confidence if associated_phone is not None else None

    vision_quality = _compute_vision_quality(person_present, face_present, eyes_state, head_orientation)

    return PerceptionSnapshot(
        timestamp=timestamp,
        person_present=person_present,
        primary_person=primary_person,
        phone_detected=phone_detected,
        phone_confidence=phone_confidence,
        face_present=face_present,
        eyes_state=eyes_state,
        eye_metric=eye_metric,
        head_orientation=head_orientation,
        head_yaw=head_yaw,
        head_pitch=head_pitch,
        vision_quality=vision_quality,
    )
