"""Tests for the application entry point (PRD section 34 integration).

main() only loads configuration and delegates to FocusGuardApp - tested
here by substituting a fake FocusGuardApp, so these tests never touch a
real camera, model, or Pygame window (that is FocusGuardApp's own test
suite's job, in tests/test_app.py).
"""

from __future__ import annotations

from pathlib import Path

import main
from src.core.config_manager import ConfigManager


class FakeApp:
    """Records the config it was constructed with and returns a
    configurable exit code from run() - substituted for the real
    FocusGuardApp so these tests exercise only main()'s own logic."""

    last_config = None

    def __init__(self, config, *, run_result: int = 0) -> None:
        FakeApp.last_config = config
        self._run_result = run_result

    def run(self) -> int:
        return self._run_result


def test_main_returns_one_with_missing_config(monkeypatch, tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.yaml"
    monkeypatch.setattr(main, "ConfigManager", lambda: ConfigManager(missing_path))

    assert main.main() == 1


def test_main_prints_configuration_error_message(monkeypatch, tmp_path: Path, capsys) -> None:
    missing_path = tmp_path / "does_not_exist.yaml"
    monkeypatch.setattr(main, "ConfigManager", lambda: ConfigManager(missing_path))

    main.main()
    captured = capsys.readouterr()

    assert "Configuration error" in captured.out


def test_main_does_not_construct_app_on_config_error(monkeypatch, tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.yaml"
    monkeypatch.setattr(main, "ConfigManager", lambda: ConfigManager(missing_path))
    FakeApp.last_config = None
    monkeypatch.setattr(main, "FocusGuardApp", FakeApp)

    main.main()

    assert FakeApp.last_config is None


def test_main_constructs_focus_guard_app_with_loaded_config(monkeypatch, valid_config_path: Path) -> None:
    monkeypatch.setattr(main, "ConfigManager", lambda: ConfigManager(valid_config_path))
    FakeApp.last_config = None
    monkeypatch.setattr(main, "FocusGuardApp", FakeApp)

    main.main()

    assert FakeApp.last_config is not None
    assert FakeApp.last_config.source_path == valid_config_path


def test_main_returns_focus_guard_app_run_result_zero(monkeypatch, valid_config_path: Path) -> None:
    monkeypatch.setattr(main, "ConfigManager", lambda: ConfigManager(valid_config_path))
    monkeypatch.setattr(main, "FocusGuardApp", lambda config: FakeApp(config, run_result=0))

    assert main.main() == 0


def test_main_returns_focus_guard_app_run_result_nonzero(monkeypatch, valid_config_path: Path) -> None:
    monkeypatch.setattr(main, "ConfigManager", lambda: ConfigManager(valid_config_path))
    monkeypatch.setattr(main, "FocusGuardApp", lambda config: FakeApp(config, run_result=1))

    assert main.main() == 1
