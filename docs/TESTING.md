# Testing Architecture

## Commands actually supported by this repository

```bash
pytest -q            # run the full deterministic suite
git diff --check      # verify no whitespace errors in a change (used after every phase)
```

There is no separate `make test`, `tox`, or CI config file in this repository — `pytest
-q` is the one command every phase of this project has been verified against, and it's
what `README.md`'s "Running Tests" section documents.

## Test inventory (from the repository, as of this documentation pass)

**664 tests total**, collected across 21 test files, all passing, running in roughly
10–15 seconds with **no webcam, GPU, real model weights, real audio device, or real
display required for any of them**.

| Test file | Count | Covers |
|---|---|---|
| `test_audio_manager.py` | 89 | `AudioManager` — one-shot warnings, persistent reminders, mute, cooldown, missing files, lifecycle |
| `test_temporal_filter.py` | 67 | `DurationConfirmer`, `hysteresis()`, `Debouncer`, `Cooldown` (generic primitives) |
| `test_session_manager.py` | 62 | `SessionManager` — lifecycle, duration accounting, counts, score, JSON |
| `test_app.py` | 53 | `FocusGuardApp` — full orchestration, wiring, error paths, controls |
| `test_dashboard_view.py` | 40 | `DashboardView` formatting helpers |
| `test_ui_manager.py` | 39 | `UIManager` rendering and input handling |
| `test_phone_temporal_filter.py` | 35 | `PhoneTemporalFilter` |
| `test_state_manager.py` | 31 | `StateManager` priority/transitions |
| `test_head_orientation_filter.py` | 29 | `HeadOrientationFilter` |
| `test_event_manager.py` | 27 | `EventManager` — event types, cooldown, bounded log |
| `test_imports.py` | 22 | Every module in `src/` imports cleanly |
| `test_head_pose.py` | 21 | `head_pose.py` geometry (round-trip via `cv2.projectPoints`) |
| `test_config_manager.py` | 21 | `ConfigManager` validation |
| `test_face_analyzer.py` | 17 | `FaceAnalyzer` |
| `test_yolo_detector.py` | 16 | `YOLODetector` |
| `test_eye_metrics.py` | 16 | EAR computation, hysteresis classification |
| `test_drowsiness_filter.py` | 16 | `DrowsinessFilter` |
| `test_camera_manager.py` | 16 | `CameraManager` |
| `test_primary_person.py` | 14 | Primary-person selection, phone association |
| `test_person_away_filter.py` | 14 | `PersonAwayFilter` |
| `test_perception_snapshot.py` | 13 | `build_perception_snapshot()`, `VisionQuality` |
| `test_main.py` | 6 | `main.py` entry point |

## How hardware-dependent components are tested without hardware

Every module that touches something slow, flaky, or physical takes its dependency as
an **injectable factory or backend** (see [`ARCHITECTURE.md`](ARCHITECTURE.md#dependency-injection--testing-seams)
for the full table). This is the single idea that makes the whole suite possible:

- **Camera** — `CameraManager(config, capture_factory=lambda index: FakeVideoCapture(...))`.
  Tests supply a fake object shaped like `cv2.VideoCapture` (`isOpened()`, `set()`,
  `read()`, `release()`).
- **YOLO** — `YOLODetector(config, model_factory=lambda path: FakeYoloModel(...))`. The
  fake mimics Ultralytics' callable-model shape (`model(image, conf=, device=,
  verbose=)` → objects with `.boxes.xyxy/.conf/.cls` and `.names`).
- **MediaPipe** — `FaceAnalyzer(config, config, landmarker_factory=lambda path:
  FakeLandmarker(...))`. The fake mimics `.detect(image) -> result.face_landmarks`.
- **Audio** — `AudioManager(config, config, mixer_backend=FakeMixerBackend(),
  music_backend=FakeMusicBackend(), sound_factory=fake_factory)`.
- **Pygame display** — no injection needed; tests set `SDL_VIDEODRIVER=dummy` before
  touching `pygame`, which is SDL's own standard headless driver.

### `FocusGuardApp` is tested the same way — not with a separately-mocked orchestrator

`tests/test_app.py` constructs **real** `CameraManager`/`YOLODetector`/`FaceAnalyzer`/
`UIManager`/`AudioManager` instances, each wired to its own fakes exactly as above, and
passes them into a **real** `FocusGuardApp`. This means the orchestration logic under
test is the actual production code path in `src/core/app.py` — not a parallel mock of
it. Two distinct testing strategies are used within that file:

1. **Direct calls to `_process_frame()`** with hand-constructed `Frame` objects at
   exact chosen timestamps — used for anything requiring precise duration-threshold
   testing, since `CameraManager.read_frame()` uses real `time.monotonic()` internally
   (not controllable from a test).
2. **A handful of full `run()`/`_main_loop()` smoke tests** using a fake camera that
   yields a fixed number of frames (so the loop terminates deterministically once it
   runs dry) or a posted `pygame.QUIT` event — for genuine startup/shutdown/plumbing
   coverage.

## Deterministic timestamps

Every temporal test in this codebase uses synthetic, hand-chosen `float` timestamps —
never real wall-clock delays or `time.sleep()` — so a test asserting "confirms after
0.35 seconds" takes microseconds to run, not 350 milliseconds. A recurring, explicitly
documented lesson across the test suite: chained float addition (`0.1 + 0.1 + 0.1`) can
land a hair off an exact boundary due to IEEE754 rounding, so tests use precise
summation helpers (`math.fsum`) and the production code itself tolerates a `1e-9`
epsilon at every duration-boundary comparison.

## Error-path testing

Every failure mode documented in [`SYSTEM_FLOW.md`](SYSTEM_FLOW.md#error-handling-in-the-flow-prd-35)
has a corresponding deterministic test: camera open/read failure, YOLO/model load
failure, YOLO inference exception mid-frame, face-analysis exception mid-frame, Pygame
init failure, audio mixer init failure (non-fatal), missing/corrupt sound files,
malformed/invalid config values, out-of-order timestamps. None of these require
actually breaking real hardware — each is triggered by configuring a fake backend to
raise on cue.

## Regression testing in practice

Every new feature added after the fact (YOLO detection-interval throttling,
persistent audio reminders) was implemented alongside an explicit regression test
proving the **pre-existing** behavior was unaffected — e.g.
`test_existing_one_shot_play_phone_warning_unaffected_by_persistent_reminder_state` in
`test_audio_manager.py`, and `test_zero_interval_calls_yolo_every_frame` in
`test_app.py` (proving `detection_interval_seconds=0.0` reproduces the exact pre-throttling
behavior every earlier test already assumed).

## Real-hardware smoke tests (outside the automated suite)

Several phases were additionally verified against real hardware — a real webcam, real
YOLO11n weights, real MediaPipe Face Landmarker model, a real Pygame window, and (when
available) a real audio device — using small scripts run manually, **not** part of
`pytest -q` and not committed to the repository (they lived in a scratch directory
outside version control and were deleted after use). This is documented here because
it materially informs [`PERFORMANCE.md`](PERFORMANCE.md)'s numbers and because it
caught at least one real implementation bug during development (a debug-overlay text
legibility issue, and a spurious double-fire in the `FOCUS_RESTORED` event logic — both
fixed at the time, with regression tests added). These smoke tests are not
reproducible as-is from the repository since the scripts themselves were not kept —
this documentation reports what was verified and how, not a runnable artifact.
