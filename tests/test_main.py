"""Tests for the application entry point (Phase 0: config load only)."""

from __future__ import annotations

from pathlib import Path

import main
from src.core.config_manager import ConfigManager


def test_main_returns_zero_with_valid_config(monkeypatch: "pytest.MonkeyPatch", valid_config_path: Path) -> None:
    monkeypatch.setattr(main, "ConfigManager", lambda: ConfigManager(valid_config_path))

    assert main.main() == 0


def test_main_returns_one_with_missing_config(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.yaml"
    monkeypatch.setattr(main, "ConfigManager", lambda: ConfigManager(missing_path))

    assert main.main() == 1


def test_main_prints_configuration_source(
    monkeypatch: "pytest.MonkeyPatch", valid_config_path: Path, capsys: "pytest.CaptureFixture[str]"
) -> None:
    monkeypatch.setattr(main, "ConfigManager", lambda: ConfigManager(valid_config_path))

    main.main()
    captured = capsys.readouterr()

    assert str(valid_config_path) in captured.out
    assert "Phase 0" in captured.out
