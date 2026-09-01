"""FocusGuard entry point.

Loads and validates configuration, then runs FocusGuardApp - the full,
integrated real-time monitoring application (see FOCUSGUARD_PRD.md
sections 5 and 34). This file stays a thin entry point deliberately: all
orchestration logic lives in src/core/app.py, where it is unit-testable.
"""

from __future__ import annotations

from src.core.app import FocusGuardApp
from src.core.config_manager import ConfigError, ConfigManager


def main() -> int:
    try:
        config = ConfigManager().load()
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1

    return FocusGuardApp(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
