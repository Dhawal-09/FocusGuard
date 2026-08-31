# FocusGuard

FocusGuard is a local, real-time computer-vision desktop application that watches a
webcam feed and gives **estimated** focus feedback while you study or work: person
presence, phone distraction, eye-closure/drowsiness signals, and approximate head
orientation.

FocusGuard is a portfolio/technical demonstration. It is **not** a medical,
psychological, biometric, or scientifically validated attention-monitoring system.

**Privacy: FocusGuard processes webcam input locally and does not upload webcam
frames.** No cloud vision APIs, no face recognition, no biometric database, and no
automatic webcam recording are used.

Full product/technical requirements: [`FOCUSGUARD_PRD.md`](FOCUSGUARD_PRD.md).
Repository engineering rules: [`AGENTS.md`](AGENTS.md).

## Current Implementation Status

**Phase 0 — Project Foundation only.**

This repository currently contains the project skeleton, dependency setup,
configuration loading/validation, and initial tests. It does **not** yet contain
any computer-vision functionality. Specifically, none of the following are
implemented yet:

- webcam capture
- YOLO person/phone detection
- face/eye/head-pose analysis
- temporal filtering
- the focus state machine
- event processing
- audio alerts
- the Pygame dashboard UI
- session analytics / focus score

These are implemented incrementally in later phases (see `FOCUSGUARD_PRD.md`,
section 39, "Development Phases"). Running `python main.py` today only loads and
validates `config/config.yaml` and prints a status message.

## Prerequisites

- Windows (primary target for this MVP; CPU inference must work without a GPU)
- Python 3.11 (see "Why Python 3.11" below)
- A webcam (needed starting Phase 2, not required for Phase 0)

## Why Python 3.11 and These Dependency Versions

Dependency choice was driven by compatibility across the full stack the PRD
requires (OpenCV, Ultralytics/YOLO, MediaPipe, NumPy, Pygame) rather than by
picking the newest release of each:

| Package | Constraint | Reason |
|---|---|---|
| Python | 3.11 | Broadest, most stable wheel coverage across OpenCV, Ultralytics (PyTorch), MediaPipe, and Pygame on Windows. Python 3.12/3.13 have historically had lagging or inconsistent wheel availability for MediaPipe and PyTorch. |
| numpy | `>=1.26,<2.0` | MediaPipe and some PyTorch/Ultralytics releases have had compatibility issues with NumPy 2.x; pinning below 2.0 avoids ABI surprises. |
| opencv-python | `>=4.9,<4.11` | Stable release line with prebuilt Windows wheels for Python 3.11. |
| ultralytics | `>=8.3,<8.4` | Current stable line supporting the YOLO11n model specified in the PRD; pulls in PyTorch as its inference backend. |
| mediapipe | `>=0.10.14,<0.11` | First MediaPipe line with solid Python 3.11 support for the Face Landmarker task used for eye/head analysis in later phases. |
| pygame | `>=2.5,<2.7` | Stable release line used for the dashboard UI and audio (`pygame.mixer`) in later phases. |
| PyYAML | `>=6.0,<7.0` | Configuration file parsing. |
| pytest | `>=8.0,<9.0` | Test runner. |

Exact resolved versions actually installed and verified in this environment are
pinned in `requirements.txt` (numpy 1.26.4, opencv-python 4.10.0.84, ultralytics
8.3.253, mediapipe 0.10.35, pygame 2.6.1, PyYAML 6.0.3, pytest 8.4.2). Ultralytics
transitively installs PyTorch/torchvision (CPU build, ~1.2 GB) as its inference
backend, and MediaPipe transitively installs `opencv-contrib-python`; these are
not listed directly in `requirements.txt` since they are not imported by
FocusGuard's own code.

No ML model weights are downloaded in Phase 0; `models/yolo11n.pt` is fetched by
Ultralytics automatically the first time detection actually runs (Phase 3+).

## Virtual Environment Setup

From the project root, using Python 3.11:

```bash
python -m venv .venv
```

Activate it:

```powershell
# PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Git Bash
source .venv/Scripts/activate
```

## Dependency Installation

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python main.py
```

In Phase 0 this only loads and validates `config/config.yaml` and prints a
confirmation message — it does not open the camera or any UI window yet.

## Running Tests

```bash
pytest -q
```

Phase 0 tests cover:

- configuration loading (`tests/test_config_manager.py`)
- configuration validation, including invalid/out-of-range/malformed values
- basic import health of every module in `src/` (`tests/test_imports.py`)
- application entry-point behavior for valid and missing configuration
  (`tests/test_main.py`)

## Project Structure

```text
focusguard/
├── main.py
├── config/
│   └── config.yaml
├── models/            # YOLO weights land here (Phase 3+); not tracked by Git
├── assets/
│   ├── sounds/        # user-provided audio warnings (Phase 10+)
│   └── music/         # user-provided background focus music (Phase 10+)
├── logs/              # session JSON summaries (Phase 11+); not tracked by Git
├── src/
│   ├── core/          # config_manager.py, types.py
│   ├── camera/        # camera_manager.py (Phase 2)
│   ├── detection/      # yolo_detector.py, detection_types.py (Phase 3)
│   ├── face/           # face_analyzer.py, eye_metrics.py, head_pose.py (Phase 5-6)
│   ├── state/          # temporal_filter.py, state_manager.py (Phase 7-8)
│   ├── events/         # event_manager.py (Phase 8)
│   ├── audio/          # audio_manager.py (Phase 10)
│   ├── session/        # session_manager.py (Phase 11)
│   └── ui/             # ui_manager.py (Phase 9)
└── tests/
```

## Phase 0 Scope Statement

**Phase 0 is project foundation only**: environment, dependency management,
project/config structure, and initial tests. It intentionally implements no
computer-vision, UI, audio, or session logic. Subsequent phases are implemented
one at a time per `FOCUSGUARD_PRD.md`, section 39.
