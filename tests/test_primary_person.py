"""Tests for primary-person selection and phone-person association
(FOCUSGUARD_PRD.md section 8).

Fully deterministic: synthetic Detection lists only. No webcam, GPU, or
YOLO model.
"""

from __future__ import annotations

from src.detection.detection_types import Detection
from src.detection.primary_person import select_associated_phone, select_primary_person

PERSON = "person"
PHONE = "cell phone"


def person(x1: float, y1: float, x2: float, y2: float, confidence: float = 0.9, timestamp: float = 0.0) -> Detection:
    return Detection(class_name=PERSON, confidence=confidence, x1=x1, y1=y1, x2=x2, y2=y2, timestamp=timestamp)


def phone(x1: float, y1: float, x2: float, y2: float, confidence: float = 0.9, timestamp: float = 0.0) -> Detection:
    return Detection(class_name=PHONE, confidence=confidence, x1=x1, y1=y1, x2=x2, y2=y2, timestamp=timestamp)


# --- select_primary_person -------------------------------------------------


def test_no_detections_returns_none() -> None:
    assert select_primary_person([]) is None


def test_no_person_detections_returns_none() -> None:
    detections = [phone(10, 10, 20, 20)]

    assert select_primary_person(detections) is None


def test_single_person_is_primary() -> None:
    only = person(0, 0, 100, 100)

    assert select_primary_person([only]) is only


def test_largest_person_wins_over_smaller() -> None:
    small = person(0, 0, 10, 10)
    large = person(0, 0, 200, 200)

    assert select_primary_person([small, large]) is large
    assert select_primary_person([large, small]) is large


def test_ties_resolve_to_first_encountered() -> None:
    first = person(0, 0, 100, 100)
    second = person(200, 200, 300, 300)  # identical area, different position

    assert select_primary_person([first, second]) is first


def test_non_person_detections_are_ignored() -> None:
    the_person = person(0, 0, 50, 50)
    detections = [phone(0, 0, 200, 200), the_person]  # phone box is larger but not a person

    assert select_primary_person(detections) is the_person


def test_degenerate_zero_area_box_is_still_selectable_when_only_option() -> None:
    zero_area = person(50, 50, 50, 50)

    assert select_primary_person([zero_area]) is zero_area


# --- select_associated_phone ------------------------------------------------


def test_no_phone_detections_returns_none() -> None:
    primary = person(0, 0, 100, 100)

    assert select_associated_phone([primary], primary) is None


def test_no_primary_person_still_returns_highest_confidence_phone() -> None:
    low = phone(0, 0, 10, 10, confidence=0.55)
    high = phone(100, 100, 110, 110, confidence=0.90)

    result = select_associated_phone([low, high], None)

    assert result is high


def test_phone_inside_primary_person_box_is_associated() -> None:
    primary = person(0, 0, 100, 100)
    inside_phone = phone(40, 40, 60, 60)  # center (50, 50) is inside the person box

    result = select_associated_phone([primary, inside_phone], primary)

    assert result is inside_phone


def test_phone_outside_primary_person_box_still_returned_as_fallback() -> None:
    """A missed spatial association must never suppress a real phone
    signal - PRD section 8 says association is a 'preferable' heuristic,
    not a hard gate."""
    primary = person(0, 0, 100, 100)
    outside_phone = phone(500, 500, 520, 520)  # nowhere near the person

    result = select_associated_phone([primary, outside_phone], primary)

    assert result is outside_phone


def test_associated_phone_preferred_over_higher_confidence_unassociated_phone() -> None:
    primary = person(0, 0, 100, 100)
    associated = phone(40, 40, 60, 60, confidence=0.60)
    unassociated_but_higher_confidence = phone(500, 500, 520, 520, confidence=0.95)

    result = select_associated_phone([primary, associated, unassociated_but_higher_confidence], primary)

    assert result is associated


def test_highest_confidence_among_multiple_associated_phones_wins() -> None:
    primary = person(0, 0, 100, 100)
    low = phone(10, 10, 20, 20, confidence=0.55)
    high = phone(60, 60, 70, 70, confidence=0.80)

    result = select_associated_phone([primary, low, high], primary)

    assert result is high


def test_phone_center_exactly_on_person_boundary_counts_as_associated() -> None:
    primary = person(0, 0, 100, 100)
    boundary_phone = phone(90, 90, 110, 110)  # center exactly (100, 100), the corner

    result = select_associated_phone([primary, boundary_phone], primary)

    assert result is boundary_phone
