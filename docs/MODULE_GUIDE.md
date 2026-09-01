# Module Guide

A per-file reference. For the big picture, see [`ARCHITECTURE.md`](ARCHITECTURE.md);
for the runtime call order, see [`SYSTEM_FLOW.md`](SYSTEM_FLOW.md).

## `src/core/`

| File | Key exports | Notes |
|---|---|---|
| `config_manager.py` | `ConfigManager`, `AppConfig`, `ConfigError`, and one frozen dataclass per config section (`CameraConfig`, `YoloConfig`, `PhoneConfig`, `FaceConfig`, `EyesConfig`, `HeadConfig`, `PersonConfig`, `AudioConfig`, `UIConfig`, `ScoreConfig`, `SessionConfig`) | The only module every other config-consuming class imports from. See [`CONFIGURATION.md`](CONFIGURATION.md). |
| `types.py` | `PerceptionSnapshot`, `VisionQuality`, `build_perception_snapshot()` | The one place raw detector/analyzer output becomes the single immutable per-frame contract everything downstream consumes. |
| `app.py` | `FocusGuardApp` | The orchestrator. The only module that imports every other manager. See [`SYSTEM_FLOW.md`](SYSTEM_FLOW.md). |

## `src/camera/`

| File | Key exports | Notes |
|---|---|---|
| `camera_manager.py` | `CameraManager`, `Frame`, `CameraError` | `Frame.timestamp` is `time.monotonic()` at capture, never wall-clock. Injectable via `capture_factory`. |

## `src/detection/`

| File | Key exports | Notes |
|---|---|---|
| `detection_types.py` | `Detection` | One frozen dataclass: `class_name, confidence, x1, y1, x2, y2, timestamp`. |
| `yolo_detector.py` | `YOLODetector`, `DetectionError`, `PERSON_CLASS_NAME`, `CELL_PHONE_CLASS_NAME` | Loads YOLO, resolves `auto`/`cpu`/`cuda` device once, filters output to person/cell-phone above configured confidence. Injectable via `model_factory`. |
| `primary_person.py` | `select_primary_person()`, `select_associated_phone()` | Pure functions, no model, no state. |

## `src/face/`

| File | Key exports | Notes |
|---|---|---|
| `eye_metrics.py` | `EyeState`, `compute_ear()`, `classify_eye_state()`, `combine_eye_metrics()` | Eye Aspect Ratio math + hysteresis classification. Pure functions. |
| `face_analyzer.py` | `FaceAnalyzer`, `FaceAnalysisResult`, `FaceAnalysisError` | Loads MediaPipe Face Landmarker, runs inference, derives eye state. Injectable via `landmarker_factory`. Never classifies a missing face as `CLOSED`. |
| `head_pose.py` | `HeadOrientation`, `HeadPoseResult`, `estimate_head_pose()`, `classify_orientation()` | Pure function — `cv2.solvePnP` against a generic 3D reference face using six of the landmarks `FaceAnalyzer` already produced. Not a model. |

## `src/state/`

| File | Key exports | Notes |
|---|---|---|
| `temporal_filter.py` | `DurationConfirmer`, `hysteresis()`, `Debouncer`, `Cooldown` | The generic, reusable timing primitives everything else in this directory (and `AudioManager`, `EventManager`) is built from. See [`TEMPORAL_FILTERING.md`](TEMPORAL_FILTERING.md). |
| `phone_temporal_filter.py` | `PhoneTemporalFilter`, `PhoneFilterResult` | Confirm/clear with a grace period. |
| `head_orientation_filter.py` | `HeadOrientationFilter`, `HeadOrientationFilterResult` | Confirm, immediate clear. `UNKNOWN` counts as centered. |
| `drowsiness_filter.py` | `DrowsinessFilter`, `DrowsinessFilterResult` | Thin `DurationConfirmer` wrapper keyed on `EyeState.CLOSED`. |
| `person_away_filter.py` | `PersonAwayFilter`, `PersonAwayFilterResult` | Thin `DurationConfirmer` wrapper keyed on `not person_present`. |
| `state_manager.py` | `StateManager`, `FocusState`, `StateTransition` | The priority-based decision engine. See [`STATE_MACHINE.md`](STATE_MACHINE.md). |

## `src/events/`

| File | Key exports | Notes |
|---|---|---|
| `event_manager.py` | `EventManager`, `Event`, `EventType`, `Severity` | Bounded log, one cooldown (phone only). See [`EVENT_SYSTEM.md`](EVENT_SYSTEM.md). |

## `src/audio/`

| File | Key exports | Notes |
|---|---|---|
| `audio_manager.py` | `AudioManager`, `AudioError`, `DEFAULT_SOUND_PATHS`, `DEFAULT_MUSIC_PATH` | Owns *only* audio — never imports `EventType` or `FocusState`. See [`AUDIO_SYSTEM.md`](AUDIO_SYSTEM.md). |

## `src/session/`

| File | Key exports | Notes |
|---|---|---|
| `session_manager.py` | `SessionManager`, `SessionSummary`, `SessionError` | Incremental push model (`record_transition`/`record_event`). See [`SESSION_ANALYTICS.md`](SESSION_ANALYTICS.md). |

## `src/ui/`

| File | Key exports | Notes |
|---|---|---|
| `dashboard_view.py` | `DashboardView`, `DebugInfo`, `UIAction`, and formatting helpers (`format_status()`, `format_duration()`, ...) | Plain data + pure functions — importable without `pygame` at all. |
| `ui_manager.py` | `UIManager`, `UIError` | The only module that touches `pygame`. Never contains CV logic (PRD explicit rule). Injectable in tests via `SDL_VIDEODRIVER=dummy`, not constructor injection. |

## `main.py`

Deliberately thin: load config, construct `FocusGuardApp`, call `.run()`, return its
exit code. All orchestration logic lives in `src/core/app.py` specifically so it's
unit-testable — `main.py` itself has only 6 tests, all about its own config-load/delegate
logic (via a substituted fake `FocusGuardApp`), never touching real hardware.

## Cross-cutting: what's a *class* vs. a *pure function* in this codebase

A useful lens when reading the source for the first time — modules split cleanly into
two categories:

- **Stateful classes with a lifecycle** (need `init()`/`load()`, hold internal state
  across calls): `CameraManager`, `YOLODetector`, `FaceAnalyzer`, every temporal
  filter, `StateManager`, `EventManager`, `AudioManager`, `SessionManager`,
  `UIManager`, `FocusGuardApp`.
- **Pure, stateless functions** (same input always produces the same output, nothing
  to construct or tear down): everything in `eye_metrics.py`, `head_pose.py`,
  `primary_person.py`, `build_perception_snapshot()`, `hysteresis()`, and every
  `format_*()` helper in `dashboard_view.py`.

This split is not incidental — it's exactly why the pure-function modules have no
dedicated `init()`/fixture ceremony in their tests at all, just direct calls with
synthetic inputs.
