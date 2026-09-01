# Interview Guide

Practical, ready-to-use explanations of FocusGuard at increasing depth, plus prepared
answers to the questions an interviewer is likely to ask. Every answer is grounded in
what's actually in the repository — no feature described here is aspirational.

## 30-second explanation

> "FocusGuard is a local, real-time computer-vision app — it watches your webcam
> through YOLO and MediaPipe, figures out if you're focused, on your phone, drowsy, or
> looking away, and shows you a live status with a focus score. Everything runs
> on-device, nothing is uploaded."

## 60-second explanation

> "FocusGuard watches your webcam and gives you real-time feedback on your focus while
> you study or work. Under the hood it's a pipeline: YOLO finds you and your phone in
> each frame, MediaPipe finds 468 points on your face for eye-openness and head angle,
> and then — this is the important part — none of those raw per-frame readings are
> trusted directly. They go through temporal filters that require a condition to hold
> continuously for a configured duration before it counts, so a single blink or one
> blurry frame never triggers a false alarm. A priority-based state machine turns the
> filtered signals into one clear status, which drives a live dashboard, optional audio
> warnings, and a running session score. It's built as fifteen independently-testable
> modules with a single orchestrator wiring them together, which is also why it has
> 660+ deterministic tests that run in about ten seconds with no camera or GPU
> attached."

## 2-minute technical explanation

> "The architecture is a strict pipeline, one class per responsibility, none of them
> aware of each other. `CameraManager` reads frames. `YOLODetector` finds person and
> cell-phone boxes. `FaceAnalyzer` runs MediaPipe and computes an Eye Aspect Ratio for
> openness. A pure function does head-pose estimation via `solvePnP` on six of those
> same landmarks — no second model call. All of that gets assembled into one immutable
> `PerceptionSnapshot` per frame, including a derived `VisionQuality` that distinguishes
> 'no person visible' from 'person's there but I can't read their face right now' — that
> distinction is what drives whether the state machine reports `AWAY` or `UNKNOWN`.
>
> Four independent temporal filters — phone, drowsiness, attention, and person-away —
> each require their raw signal to hold continuously for a configured duration before
> reporting 'confirmed.' They're built from a small set of generic, timestamp-based
> primitives — a duration-confirmer state machine, a cooldown, hysteresis — so the same
> mechanism handles four completely different conditions.
>
> `StateManager` takes those four confirmed booleans plus vision quality and runs them
> through a strict priority chain — away beats phone beats drowsiness beats attention
> beats focused — deterministically, every frame. State transitions generate events,
> which drive a bounded on-screen log, a running focus score, session-duration
> accounting, and audio warnings — including, as a feature I added after the original
> spec was done, *persistent* reminders that repeat every N seconds while a distraction
> continues, not just once.
>
> The whole thing is glued together by one orchestrator class that's the only module
> that imports everything else — every other manager is independently constructible
> with an injectable fake for its hardware dependency, which is why the test suite
> doesn't need a camera, a GPU, or an audio device at all."

## 5-minute deep-dive

Use the 2-minute explanation as the spine, then draw from these on request:

- **Why temporal filtering, concretely**: walk through the drowsiness example — a
  blink is a closure under 1.2 seconds and never reaches `CONFIRMED`; sustained closure
  past 1.2 seconds does. No blink-specific code exists — it falls out of the duration
  threshold for free. (See [`TEMPORAL_FILTERING.md`](TEMPORAL_FILTERING.md).)
- **The priority example**: phone-in-hand *and* looking away simultaneously →
  `PHONE_DISTRACTION` wins, because it's earlier in the `elif` chain — walk through why
  that ordering was chosen. (See [`STATE_MACHINE.md`](STATE_MACHINE.md).)
- **The measured performance story**: real hardware showed YOLO costing 50–70ms/call,
  capping FPS at ~11 against a 20–30 target; throttling YOLO to once per 0.1s (while
  face analysis stayed unthrottled) measured ~26.5 fps — a real 2.4× improvement, not
  an estimate. (See [`PERFORMANCE.md`](PERFORMANCE.md).)
- **The one-shot vs. persistent audio distinction** — walk through the worked example
  in [`AUDIO_SYSTEM.md`](AUDIO_SYSTEM.md).
- **How 660+ tests run without hardware** — dependency injection at every hardware
  boundary; `FocusGuardApp` itself tested with real managers wired to fakes, not a
  mocked orchestrator. (See [`TESTING.md`](TESTING.md).)

---

## Prepared Q&A

Format for each: **short answer**, **technical explanation**, **code pointer**.

### Why YOLO?

**Short:** A lightweight, well-supported, pretrained object detector that already
knows "person" and "cell phone" — no custom training needed, which the project
explicitly scoped out.

**Technical:** YOLO11n is the PRD's specified baseline — small enough for real-time CPU
inference, with a stock COCO-trained model already covering the two classes needed.
Device selection (`auto`/`cpu`/`cuda`) resolves once at construction with automatic
CPU fallback.

**Code:** `src/detection/yolo_detector.py` (`YOLODetector._resolve_device`).

### Why MediaPipe?

**Short:** A pretrained, fast, CPU-friendly face-landmark model — gives 468 points per
face without training anything.

**Technical:** `FaceAnalyzer` uses MediaPipe's Face Landmarker task; eye-openness (EAR)
and head-pose geometry are both derived from the *same* landmark set in one inference
call, not two separate models.

**Code:** `src/face/face_analyzer.py`, `src/face/head_pose.py`.

### Why temporal filtering? Why not trust a single frame?

**Short:** A single frame can be wrong for entirely mundane reasons — motion blur, a
blink, one bad-lighting frame — and reacting to every one of those would make the app
useless (constant false alarms).

**Technical:** Every condition in FocusGuard is a **confirm-over-duration** state
machine, not an instant reaction. A closure under `drowsiness_duration_seconds`
(1.2s default) never reaches `CONFIRMED`. This is enforced structurally —
`StateManager` never even receives raw per-frame booleans, only already-filtered ones.

**Code:** `src/state/temporal_filter.py` (`DurationConfirmer`), `src/state/*_filter.py`.

### Why a state machine?

**Short:** Multiple conditions can be true in the same frame (phone AND looking away)
— something has to deterministically pick one status to show.

**Technical:** `StateManager.evaluate()` is a strict `elif` priority chain
(`AWAY > PHONE_DISTRACTION > DROWSINESS_SIGNAL > ATTENTION_DIVERTED > FOCUSED >
UNKNOWN`), re-evaluated fresh every frame from already-filtered inputs — deterministic
by construction, no ambiguity possible.

**Code:** `src/state/state_manager.py`.

### How do you handle conflicting states?

**Short:** Priority order — the more severe/actionable condition wins. Away beats
everything (if they're not there, nothing else matters); phone beats drowsiness and
attention (a phone in hand is judged more actionable).

**Technical:** See the worked phone+looking-away example.

**Code:** `src/state/state_manager.py::evaluate()`; see
[`STATE_MACHINE.md`](STATE_MACHINE.md#concrete-example-phone--looking-away-simultaneously).

### How did you improve FPS?

**Short:** Measured first (real hardware, real models), found YOLO was the bottleneck
at ~11fps, throttled YOLO calls to once per 0.1s while keeping face analysis
unthrottled, re-measured at ~26.5fps.

**Technical:** `yolo.detection_interval_seconds` gates real YOLO calls in
`FocusGuardApp._process_frame()`; frames in between reuse the last `Detection` list.
Not a guess — a measured 2.4× improvement.

**Code:** `src/core/app.py::_should_run_detection()`; full numbers in
[`PERFORMANCE.md`](PERFORMANCE.md).

### Why throttle YOLO but not face analysis?

**Short:** YOLO is 3–4× more expensive, and eye/drowsiness timing needs frame-accurate
face data — a person's position doesn't change meaningfully in 0.1s, but a blink does.

**Technical:** See [`CV_PIPELINE.md`](CV_PIPELINE.md#cpu-performance-implications).

**Code:** `src/core/app.py::_process_frame()`.

### How does the audio system avoid spam?

**Short:** Two layers — the underlying condition only reports "confirmed" once per
genuine confirmation (not every frame it stays true), and a separate persistent-reminder
timer only repeats a warning every N configured seconds while the condition remains
continuously active, resetting completely the moment it clears.

**Technical:** `_PersistentReminder` tracks a baseline timestamp per condition;
`notify_*()` methods are called every frame with the *current* boolean, but only return
"due" once the interval has genuinely elapsed since the last reminder. Mute doesn't
pause this timer — only the actual sound playback — so unmuting never causes a
backlog burst.

**Code:** `src/audio/audio_manager.py` (`_PersistentReminder`,
`notify_drowsiness()` etc.); full explanation in [`AUDIO_SYSTEM.md`](AUDIO_SYSTEM.md).

### How is session duration calculated?

**Short:** Every state-transition call credits the *elapsed interval since the last
call* to whichever state was active *during* that interval — not the new one just
reported.

**Technical:** `SessionManager.record_transition()` is called every evaluated frame
(not just on changes); duration attribution is `[last_call_timestamp,
this_call_timestamp)` credited to `_last_state`. Pausing resets the accounting clock's
origin on resume, so a paused gap is never silently counted.

**Code:** `src/session/session_manager.py::_accumulate()`; worked example in
[`SESSION_ANALYTICS.md`](SESSION_ANALYTICS.md).

### How did you test hardware-dependent components?

**Short:** Every hardware boundary (camera, YOLO, MediaPipe, audio, display) is
injectable — tests supply a fake shaped like the real thing, and even the full
application orchestrator is tested with real managers wired to those same fakes, not a
separately-mocked orchestrator.

**Technical:** See [`TESTING.md`](TESTING.md) for the exact injection points and how
`FocusGuardApp` itself gets tested this way.

**Code:** Constructor keyword arguments across every manager (`capture_factory`,
`model_factory`, `landmarker_factory`, `mixer_backend`/`music_backend`/`sound_factory`).

### How does error handling work?

**Short:** Startup failures (camera/model/display) are fatal with a readable message;
per-frame inference failures degrade that one frame gracefully and keep the session
running; audio failures never take down the app at all.

**Technical:** See the full table in
[`SYSTEM_FLOW.md`](SYSTEM_FLOW.md#error-handling-in-the-flow-prd-35).

**Code:** `src/core/app.py::run()`, `_startup()`, `_process_frame()`.

### What happens if the camera fails?

**Short:** At startup, the app exits with a readable error before ever opening a
window's worth of broken state. Mid-session, a read failure stops the loop cleanly
(logs a `CAMERA_ERROR` event if a session was active) rather than retry-looping
forever or crashing.

**Code:** `src/core/app.py::_handle_camera_error()`.

### What happens if YOLO fails?

**Short:** Loading failure at startup is fatal (can't run without a model). An
inference exception mid-session degrades just that one frame to "no detections" and
logs a `MODEL_ERROR` — the session keeps running.

**Code:** `src/core/app.py::_process_frame()`.

### What happens if audio fails?

**Short:** Never fatal, anywhere. A missing sound *file* is caught once at load time
and that sound simply never plays again. A missing audio *device* is caught at startup
and the whole app runs without sound, logged but not blocking.

**Code:** `src/audio/audio_manager.py::init()`, `_load_sound()`.

### What happens if the face model fails?

**Short:** Loading failure at startup is fatal (same reasoning as YOLO). An inference
exception mid-session degrades that frame to "no face detected" — never interpreted as
eyes closed — and logs a `VISION_ERROR`.

**Code:** `src/core/app.py::_process_frame()`; `src/face/face_analyzer.py`.

### How would you scale this?

**Short (honest, not exaggerated):** This is explicitly a single-user, single-process,
local desktop app — the PRD scopes out cloud backends, multi-user support, and
authentication entirely. "Scaling" it in the traditional sense isn't really the design
goal; the more interesting question is what's already reusable if the requirements
changed.

**Technical:** Because every manager is independently constructible and stateless
between sessions (aside from `SessionManager`'s own per-session accumulation), the CV
pipeline and temporal-filtering logic could in principle run per-connection in a
different process model without rewriting their internals — but that's speculative;
nothing in the current codebase does this or was built with it as a goal.

### How would you make it multi-user?

**Short:** Not implemented, and honestly out of scope for what this project set out to
be — a local, single-user, privacy-first desktop tool (no cloud APIs, no accounts, by
explicit design).

**Technical (if pressed for a direction):** Each user's session would need its own
`FocusGuardApp` instance (or at least its own `StateManager`/`SessionManager`/
`AudioManager` — those are the ones holding per-session state); the CV pipeline
managers themselves have no cross-user state to isolate. This is not planned or
partially built.

### How would you generate a parent productivity report?

**Short:** Not implemented — but the data shape is already there. Every session
already writes a self-contained JSON summary (duration, distraction counts, focus
score) to `logs/`. A report generator could read a directory of those files without
needing changes to `SessionManager` itself.

**Technical:** See [`SESSION_ANALYTICS.md`](SESSION_ANALYTICS.md#future-idea-a-productivity-report-not-implemented)
— explicitly marked there as speculation about a possible direction, not a plan.

### What would you improve next?

**Short, honest list**, not exaggerated: fix the dead `camera.target_fps` config value
(currently has no effect); resolve the PRD's own internal "FOCUS SCORE" vs. "ESTIMATED
FOCUS SCORE" label inconsistency; confirm whether the +11MB memory growth in the first
15 seconds of real inference actually plateaus over a longer real run; consider whether
`AWAY` deserves its own audio treatment now that phone/drowsiness/attention have
persistent reminders. All four are documented, known, and deliberately left as-is
rather than fixed silently — see
[`ARCHITECTURE.md`](ARCHITECTURE.md#implementation-vs-documentation-notes).
