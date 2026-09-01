"""Primary-person selection and phone-person association (PRD section 8).

Pure functions only - no model inference, no state, no config dependency
beyond simple geometry on Detection boxes already produced by YOLODetector.
Multi-person tracking is explicitly out of scope: "largest box wins" is
the entire primary-person policy (PRD section 8), and phone association is
a best-effort spatial heuristic, never a hard gate on whether a phone
counts as detected - a missed spatial association must never suppress a
real phone-distraction signal.
"""

from __future__ import annotations

from src.detection.detection_types import Detection
from src.detection.yolo_detector import CELL_PHONE_CLASS_NAME, PERSON_CLASS_NAME


def _area(detection: Detection) -> float:
    return max(0.0, detection.x2 - detection.x1) * max(0.0, detection.y2 - detection.y1)


def select_primary_person(detections: list[Detection]) -> Detection | None:
    """Return the largest-area person detection, or None if no person is present.

    Ties (equal area) resolve to whichever detection appears first, for
    determinism.
    """
    largest: Detection | None = None
    largest_area = -1.0
    for detection in detections:
        if detection.class_name != PERSON_CLASS_NAME:
            continue
        area = _area(detection)
        if area > largest_area:
            largest = detection
            largest_area = area
    return largest


def _is_associated(phone: Detection, person: Detection) -> bool:
    """True if the phone detection's center point falls within the
    person's bounding box. A simple, cheap spatial heuristic per PRD
    section 8 - not full IoU/tracking."""
    phone_center_x = (phone.x1 + phone.x2) / 2.0
    phone_center_y = (phone.y1 + phone.y2) / 2.0
    return person.x1 <= phone_center_x <= person.x2 and person.y1 <= phone_center_y <= person.y2


def select_associated_phone(detections: list[Detection], primary_person: Detection | None) -> Detection | None:
    """Return the phone detection most relevant to the primary person.

    Preference order: (1) among phones spatially associated with the
    primary person (per `_is_associated`), the highest-confidence one;
    (2) if none is associated with the primary person (or there is no
    primary person at all), the highest-confidence phone detection in the
    frame. A missed spatial association must never make phone detection
    disappear - it only affects which phone's confidence is reported.
    """
    phones = [d for d in detections if d.class_name == CELL_PHONE_CLASS_NAME]
    if not phones:
        return None

    if primary_person is not None:
        associated = [phone for phone in phones if _is_associated(phone, primary_person)]
        if associated:
            return max(associated, key=lambda d: d.confidence)

    return max(phones, key=lambda d: d.confidence)
