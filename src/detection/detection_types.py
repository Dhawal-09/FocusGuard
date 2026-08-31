"""Shared data structures for object detection results (PRD section 7)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """A single filtered YOLO detection for a target class (person/cell phone)."""

    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    timestamp: float
