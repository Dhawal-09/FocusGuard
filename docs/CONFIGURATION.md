# Configuration Reference

Source: `config/config.yaml`, loaded and validated by `src/core/config_manager.py`
(`ConfigManager.load()` → `AppConfig`). Every field below exists in the actual config
file and is actually read by the code — nothing here is invented or aspirational.
Invalid values (out of range, wrong type, or a missing section) raise `ConfigError`
with a readable message at startup; the app never runs with malformed configuration.

## `camera`

| Setting | Default | Valid range | Effect | If changed |
|---|---|---|---|---|
| `index` | `0` | integer ≥ 0 | Which OS camera device to open (`cv2.VideoCapture(index)`) | Selects a different physical/virtual webcam |
| `width` | `1280` | integer ≥ 1 | Requested capture width | Passed to `cv2.CAP_PROP_FRAME_WIDTH`; actual resolution depends on hardware support |
| `height` | `720` | integer ≥ 1 | Requested capture height | Passed to `cv2.CAP_PROP_FRAME_HEIGHT` |
| `target_fps` | `30` | integer ≥ 1 | *(see note)* | **Currently has no effect** — see below |

> **Documentation-accuracy note**: `target_fps` is parsed and validated but never
> actually consumed anywhere in `CameraManager` (it never calls `cv2.CAP_PROP_FPS`).
> Changing this value today does nothing. This is a known, previously-flagged gap —
> see [`ARCHITECTURE.md`](ARCHITECTURE.md#implementation-vs-documentation-notes) row 3.

## `yolo`

| Setting | Default | Valid range | Effect | If changed |
|---|---|---|---|---|
| `model` | `models/yolo11n.pt` | non-empty string | Path to the YOLO weights file | Auto-downloaded by Ultralytics on first use if the path doesn't exist yet |
| `confidence` | `0.45` | `0.0`–`1.0` | Minimum confidence to accept a **person** detection | Higher = fewer false person detections, but risks missing a real person in poor conditions |
| `phone_confidence` | `0.55` | `0.0`–`1.0` | Minimum confidence to accept a **cell phone** detection | Higher = fewer false phone-distraction triggers, but risks missing a real phone |
| `device` | `auto` | `auto` \| `cpu` \| `cuda` | Inference device | `auto`/`cuda` fall back to `cpu` automatically if CUDA isn't available |
| `detection_interval_seconds` | `0.1` | ≥ `0.0` | Minimum real time between actual YOLO calls; frames in between reuse the last detection result | `0.0` = run YOLO every frame (highest accuracy, lowest FPS — see [`PERFORMANCE.md`](PERFORMANCE.md)); larger = fewer YOLO calls, higher FPS, slightly staler person/phone detection between calls |

## `phone`

| Setting | Default | Valid range | Effect | If changed |
|---|---|---|---|---|
| `confirm_duration_seconds` | `0.35` | ≥ `0.0` | How long a phone must be continuously detected before `PHONE_DISTRACTION` is confirmed | Lower = faster (but noisier) triggering; `0.0` = confirms on the very next frame |
| `clear_duration_seconds` | `0.60` | ≥ `0.0` | Grace period before a confirmed phone distraction clears once the phone disappears | Higher = more tolerant of brief occlusion; `0.0` = clears immediately |
| `warning_cooldown_seconds` | `10` | ≥ `0.0` | Minimum gap between repeated `PHONE_DETECTED` **log entries** for rapid re-confirmation | Does not affect the persistent-reminder mechanism (a separate, independent setting — see `audio.persistent_warning_interval_seconds` below and [`AUDIO_SYSTEM.md`](AUDIO_SYSTEM.md)) |

## `face`

| Setting | Default | Effect | If changed |
|---|---|---|---|
| `model` | `models/face_landmarker.task` | Path to the MediaPipe Face Landmarker model file | Must be downloaded manually (not auto-fetched) — see `README.md` |

## `eyes`

| Setting | Default | Valid range | Effect | If changed |
|---|---|---|---|---|
| `closed_threshold` | `0.21` | ≥ `0.0` | Eye-openness metric (EAR) below which eyes classify as `CLOSED` | Lower = requires eyes to be more tightly shut before registering closed |
| `open_threshold` | `0.24` | ≥ `0.0`, must be `>` `closed_threshold` | EAR above which eyes classify as `OPEN` | Values between the two thresholds keep the previous classification (hysteresis dead zone) |
| `blink_max_duration_seconds` | `0.45` | ≥ `0.0` | **Documentation/tuning value only** — not read by `DrowsinessFilter` | See [`TEMPORAL_FILTERING.md`](TEMPORAL_FILTERING.md): a closure shorter than `drowsiness_duration_seconds` is already a "blink" by construction, this value doesn't gate anything in code |
| `drowsiness_duration_seconds` | `1.20` | ≥ `0.0` | How long eyes must stay continuously `CLOSED` before `DROWSINESS_SIGNAL` confirms | Lower = faster (but noisier) triggering |

## `head`

| Setting | Default | Valid range | Effect | If changed |
|---|---|---|---|---|
| `yaw_threshold_degrees` | `20` | `0.0`–`90.0` | Left/right rotation beyond which the head is classified off-center | Lower = more sensitive to small turns |
| `pitch_threshold_degrees` | `18` | `0.0`–`90.0` | Up/down tilt beyond which the head is classified off-center | Lower = more sensitive to small tilts |
| `confirmation_seconds` | `0.80` | ≥ `0.0` | How long the head must stay continuously off-center before `ATTENTION_DIVERTED` confirms | Lower = faster (but noisier) triggering |

## `person`

| Setting | Default | Valid range | Effect | If changed |
|---|---|---|---|---|
| `away_duration_seconds` | `3.0` | ≥ `0.0` | How long the person must be continuously undetected before `AWAY` confirms | Lower = more sensitive to brief occlusion being mistaken for absence |

## `audio`

| Setting | Default | Valid range | Effect | If changed |
|---|---|---|---|---|
| `enabled` | `true` | boolean | Master on/off switch for **all** warnings and music | `false` silences everything, but the app still runs normally |
| `volume` | `0.70` | `0.0`–`1.0` | Warning-sound playback volume | Applied to every loaded warning sound at load time and via `set_volume()` |
| `music_enabled` | `false` | boolean | Whether background focus music is started with a session | Requires `assets/music/focus_music.mp3` to exist |
| `music_volume` | `0.25` | `0.0`–`1.0` | Background music volume | — |
| `persistent_warning_interval_seconds` | `10.0` | ≥ `0.0` | *(Not part of the original PRD — see [`AUDIO_SYSTEM.md`](AUDIO_SYSTEM.md))* While a phone/drowsiness/attention distraction stays continuously confirmed, how often to repeat the warning | `0.0` fires a reminder on every single frame the condition remains active (effectively continuous); larger = less frequent nagging |

## `ui`

| Setting | Default | Effect | If changed |
|---|---|---|---|
| `debug` | `false` | Starts with the debug overlay (YOLO boxes, landmarks, timers, FPS, etc.) on | Toggle anytime at runtime with the `D` key regardless of this value |

## `score`

| Setting | Default | Valid range | Effect |
|---|---|---|---|
| `starting_score` | `100` | ≥ `0` | Focus score at session start (a demonstration metric, never claimed to be scientifically accurate) |
| `phone_event_penalty` | `10` | ≥ `0` | Deducted per confirmed `PHONE_DETECTED` |
| `drowsiness_event_penalty` | `5` | ≥ `0` | Deducted per confirmed `DROWSINESS_SIGNAL` |
| `attention_event_penalty` | `3` | ≥ `0` | Deducted per confirmed `ATTENTION_DIVERTED` |
| `away_event_penalty` | `5` | ≥ `0` | Deducted per confirmed `PERSON_LEFT` |

The score is clamped at a minimum of `0` regardless of how many penalties accumulate.

## `session`

| Setting | Default | Valid range | Effect |
|---|---|---|---|
| `max_event_log_entries` | `100` | ≥ `1` | How many recent events `EventManager`'s in-memory/on-screen log keeps (oldest dropped past this). Does **not** limit `SessionManager`'s own separate, unbounded event history used for the JSON summary — see [`SESSION_ANALYTICS.md`](SESSION_ANALYTICS.md). |

## Startup validation summary

`ConfigManager._validate()` enforces, for every section above: correct types (no
implicit coercion — a boolean where a number is expected fails), declared min/max
ranges, non-empty required strings, and one cross-field rule
(`eyes.open_threshold` must exceed `eyes.closed_threshold`). Any violation raises
`ConfigError` with the exact dotted field name and the offending value, and
`main.py`/`FocusGuardApp.run()` reports it as a readable startup failure rather than
letting the app run with invalid settings.
