"""Verify the Phase 0 package structure imports cleanly."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "src.core.config_manager",
    "src.core.types",
    "src.camera.camera_manager",
    "src.detection.yolo_detector",
    "src.detection.detection_types",
    "src.face.face_analyzer",
    "src.face.eye_metrics",
    "src.face.head_pose",
    "src.state.temporal_filter",
    "src.state.state_manager",
    "src.events.event_manager",
    "src.audio.audio_manager",
    "src.session.session_manager",
    "src.ui.ui_manager",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None


def test_config_manager_exports_expected_symbols() -> None:
    from src.core import config_manager

    assert hasattr(config_manager, "ConfigManager")
    assert hasattr(config_manager, "ConfigError")
    assert hasattr(config_manager, "AppConfig")
