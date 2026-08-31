"""Shared pytest fixtures for FocusGuard tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

VALID_CONFIG: dict[str, Any] = {
    "camera": {"index": 0, "width": 1280, "height": 720, "target_fps": 30},
    "yolo": {
        "model": "models/yolo11n.pt",
        "confidence": 0.45,
        "phone_confidence": 0.55,
        "device": "auto",
    },
    "phone": {
        "confirm_duration_seconds": 0.35,
        "clear_duration_seconds": 0.60,
        "warning_cooldown_seconds": 10,
    },
    "face": {"model": "models/face_landmarker.task"},
    "eyes": {
        "closed_threshold": 0.21,
        "open_threshold": 0.24,
        "blink_max_duration_seconds": 0.45,
        "drowsiness_duration_seconds": 1.20,
    },
    "head": {
        "yaw_threshold_degrees": 20,
        "pitch_threshold_degrees": 18,
        "confirmation_seconds": 0.80,
    },
    "person": {"away_duration_seconds": 3.0},
    "audio": {
        "enabled": True,
        "volume": 0.70,
        "music_enabled": False,
        "music_volume": 0.25,
    },
    "ui": {"debug": False},
    "score": {
        "starting_score": 100,
        "phone_event_penalty": 10,
        "drowsiness_event_penalty": 5,
        "attention_event_penalty": 3,
        "away_event_penalty": 5,
    },
    "session": {"max_event_log_entries": 100},
}


def write_config(path: Path, overrides: dict[str, Any] | None = None) -> Path:
    """Write a valid config (deep-merged with overrides) to path and return it."""
    data: dict[str, Any] = {section: dict(values) for section, values in VALID_CONFIG.items()}
    if overrides:
        for section, values in overrides.items():
            data.setdefault(section, {}).update(values)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def valid_config_path(tmp_path: Path) -> Path:
    return write_config(tmp_path / "config.yaml")
