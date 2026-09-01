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

**Fully integrated — `python main.py` runs the real, end-to-end application.**

Every subsystem the PRD describes is implemented and wired together by
`src/core/app.py`, following the main-loop order in `FOCUSGUARD_PRD.md`
section 34:

- webcam capture (`CameraManager`)
- YOLO person/phone detection (`YOLODetector`)
- face/eye/head-pose analysis (`FaceAnalyzer`, `head_pose.py`)
- primary-person selection and per-frame perception snapshots
- temporal filtering (phone, drowsiness, attention, person-away)
- the focus state machine (`StateManager`)
- event generation and a bounded event log (`EventManager`)
- audio warnings and background focus music (`AudioManager`)
- the Pygame dashboard UI, including a debug overlay (`UIManager`)
- session analytics, focus score, and JSON summary persistence (`SessionManager`)
- startup/per-frame error handling per section 35 (readable, non-silent failures;
  a single bad detection/face-analysis frame degrades gracefully rather than
  crashing the session)

Controls: `SPACE` start/pause/resume, `Q`/`ESC` exit (ends and saves the
current session first, if one is active), `M` toggle mute, `D` toggle debug
overlay, `R` reset (only while paused or idle, never mid-session).

Development proceeded phase-by-phase per `FOCUSGUARD_PRD.md` section 39 and
is documented commit-by-commit and PR-by-PR in this repository's history.

## Prerequisites

- Windows (primary target for this MVP; CPU inference must work without a GPU)
- Python 3.11 (see "Why Python 3.11" below)
- A webcam
- A MediaPipe Face Landmarker model file at `models/face_landmarker.task` (not
  auto-downloaded; download it from MediaPipe's model index and place it there)

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

`models/yolo11n.pt` is fetched automatically by Ultralytics the first time
detection runs. The MediaPipe Face Landmarker model (`models/face_landmarker.task`)
is **not** auto-downloaded and must be placed there manually before running the
application (see Prerequisites above).

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

This loads and validates `config/config.yaml`, opens the webcam, loads the YOLO
and face-landmark models, opens the dashboard window, and runs the full
real-time monitoring loop. Press `SPACE` to start a session. A failure to open
the camera, load a model, or initialize Pygame is reported with a readable
message and a non-zero exit code; a failed audio device is reported but does
not stop the app (it just runs without sound).

## Running Tests

```bash
pytest -q
```

The suite is fully deterministic and requires no webcam, GPU, real model
weights, real audio device, or real display — every hardware-facing module
(`CameraManager`, `YOLODetector`, `FaceAnalyzer`, `UIManager`, `AudioManager`)
is unit-tested against injected fake backends, and `src/core/app.py`'s
integration/orchestration logic (`tests/test_app.py`) is tested the same way,
composing real manager instances wired to those same fakes rather than a
separately-mocked orchestrator.

## Development Workflow

FocusGuard follows a lightweight GitHub Flow-based development workflow.

### Branching

All feature development happens on dedicated branches:

- `feature/*` — new functionality
- `fix/*` — bug fixes
- `refactor/*` — code restructuring
- `test/*` — test improvements
- `docs/*` — documentation
- `chore/*` — tooling, dependencies, and configuration
- `hotfix/*` — urgent fixes

The `main` branch contains only reviewed and merged work.

### Pull Requests

Every feature is developed on a dedicated branch and merged into `main`
through a Pull Request after testing and review.

### Commit Convention

FocusGuard follows Conventional Commits.

Example:

```text
feat(camera): implement camera manager
```

### Release Tags

Major development milestones are tagged:

- `v0.1.0` — Phase 0: Project Foundation
- `v0.2.0` — Phase 1: Camera Manager
- `v1.0.0` — FocusGuard V1 MVP

Detailed repository rules and agent instructions are documented in
[`GIT_WORKFLOW.md`](GIT_WORKFLOW.md).

## Project Structure

```text
focusguard/
├── main.py             # thin entry point: load config, run FocusGuardApp
├── config/
│   └── config.yaml
├── models/             # YOLO weights (auto-downloaded) + face_landmarker.task
│                       # (user-provided); not tracked by Git
├── assets/
│   ├── sounds/         # user-provided audio warnings
│   └── music/          # user-provided background focus music
├── logs/               # session JSON summaries; not tracked by Git
├── src/
│   ├── core/           # config_manager.py, types.py, app.py (FocusGuardApp - the
│   │                   # full integration/orchestration layer)
│   ├── camera/         # camera_manager.py
│   ├── detection/      # yolo_detector.py, detection_types.py, primary_person.py
│   ├── face/           # face_analyzer.py, eye_metrics.py, head_pose.py
│   ├── state/          # temporal_filter.py, state_manager.py, phone/head/
│   │                   # drowsiness/person-away temporal filters
│   ├── events/         # event_manager.py
│   ├── audio/          # audio_manager.py
│   ├── session/        # session_manager.py
│   └── ui/             # ui_manager.py, dashboard_view.py
└── tests/
```

## Scope Statement

FocusGuard V1 is implemented as specified in `FOCUSGUARD_PRD.md`: a local,
real-time, portfolio-grade computer-vision desktop application, built
incrementally phase-by-phase per section 39. Out of scope by design (PRD
section 43): a custom-trained detection model, personalized calibration,
browser integration, and application packaging/distribution.
