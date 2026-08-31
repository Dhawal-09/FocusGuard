"""Tests for configuration loading and validation (FOCUSGUARD_PRD.md section 30)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config_manager import ConfigError, ConfigManager
from tests.conftest import write_config


def test_loads_valid_config(valid_config_path: Path) -> None:
    config = ConfigManager(valid_config_path).load()

    assert config.camera.index == 0
    assert config.camera.width == 1280
    assert config.yolo.model == "models/yolo11n.pt"
    assert config.yolo.device == "auto"
    assert config.phone.confirm_duration_seconds == pytest.approx(0.35)
    assert config.eyes.closed_threshold == pytest.approx(0.21)
    assert config.eyes.open_threshold == pytest.approx(0.24)
    assert config.head.yaw_threshold_degrees == pytest.approx(20)
    assert config.person.away_duration_seconds == pytest.approx(3.0)
    assert config.audio.enabled is True
    assert config.ui.debug is False
    assert config.score.starting_score == 100
    assert config.session.max_event_log_entries == 100
    assert config.source_path == valid_config_path


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(ConfigError, match="not found"):
        ConfigManager(missing_path).load()


def test_malformed_yaml_raises_config_error(tmp_path: Path) -> None:
    bad_file = tmp_path / "config.yaml"
    bad_file.write_text("camera: [unclosed", encoding="utf-8")

    with pytest.raises(ConfigError, match="not valid YAML"):
        ConfigManager(bad_file).load()


def test_non_mapping_yaml_raises_config_error(tmp_path: Path) -> None:
    bad_file = tmp_path / "config.yaml"
    bad_file.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="mapping"):
        ConfigManager(bad_file).load()


def test_missing_section_raises_config_error(tmp_path: Path) -> None:
    path = write_config(tmp_path / "config.yaml")
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data["yolo"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="yolo"):
        ConfigManager(path).load()


def test_invalid_yolo_device_raises_config_error(tmp_path: Path) -> None:
    path = write_config(tmp_path / "config.yaml", {"yolo": {"device": "tpu"}})

    with pytest.raises(ConfigError, match="yolo.device"):
        ConfigManager(path).load()


@pytest.mark.parametrize(
    "overrides",
    [
        {"yolo": {"confidence": 1.5}},
        {"yolo": {"confidence": -0.1}},
        {"phone": {"confirm_duration_seconds": -1.0}},
        {"audio": {"volume": 1.1}},
        {"score": {"starting_score": -5}},
        {"session": {"max_event_log_entries": 0}},
    ],
)
def test_out_of_range_values_raise_config_error(tmp_path: Path, overrides: dict) -> None:
    path = write_config(tmp_path / "config.yaml", overrides)

    with pytest.raises(ConfigError):
        ConfigManager(path).load()


def test_eyes_open_threshold_must_exceed_closed_threshold(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "config.yaml",
        {"eyes": {"closed_threshold": 0.25, "open_threshold": 0.20}},
    )

    with pytest.raises(ConfigError, match="open_threshold"):
        ConfigManager(path).load()


@pytest.mark.parametrize(
    "overrides",
    [
        {"camera": {"index": "zero"}},
        {"audio": {"enabled": "yes"}},
        {"yolo": {"model": ""}},
    ],
)
def test_wrong_type_values_raise_config_error(tmp_path: Path, overrides: dict) -> None:
    path = write_config(tmp_path / "config.yaml", overrides)

    with pytest.raises(ConfigError):
        ConfigManager(path).load()


def test_default_config_path_points_at_repo_config(tmp_path: Path) -> None:
    from src.core.config_manager import DEFAULT_CONFIG_PATH

    assert DEFAULT_CONFIG_PATH.name == "config.yaml"
    assert DEFAULT_CONFIG_PATH.parent.name == "config"


def test_repo_config_yaml_is_valid() -> None:
    """The actual config/config.yaml shipped in the repo must itself validate."""
    config = ConfigManager().load()

    assert config.camera.target_fps == 30
    assert config.yolo.confidence == pytest.approx(0.45)
