"""FocusGuard entry point.

Phase 0: loads and validates configuration only. Camera, detection, face
analysis, state machine, UI, and audio are implemented in later phases
(see FOCUSGUARD_PRD.md).
"""

from __future__ import annotations

from src.core.config_manager import ConfigError, ConfigManager


def main() -> int:
    try:
        config = ConfigManager().load()
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1

    print("FocusGuard - Phase 0 (project foundation)")
    print(f"Configuration loaded from: {config.source_path}")
    print(
        "Camera capture, YOLO detection, face analysis, state machine, "
        "UI, and audio are not implemented yet."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
