# Temporal Filtering

## "Detected in one frame" vs. "confirmed as a real condition"

This is the single most important distinction in the whole codebase, and it's worth
stating precisely:

- **Detected in one frame** = a raw reading from YOLO or MediaPipe on *this specific
  frame* — e.g. "YOLO found a cell phone in this frame's boxes." This can be wrong for
  a single frame due to blur, occlusion, a blink, lighting, or model noise.
- **Confirmed as a real condition** = the raw reading has been **continuously true for
  a configured minimum duration**, measured by real elapsed time (not frame count).
  Only a *confirmed* condition is allowed to change `FocusState`, generate an `Event`,
  or trigger audio.

Nothing in FocusGuard treats a single frame's raw detector output as an event. This is
stated as an explicit rule in `AGENTS.md` ("never treat one noisy frame as an event")
and enforced structurally: `StateManager` (see [`STATE_MACHINE.md`](STATE_MACHINE.md))
never even sees raw per-frame booleans — only the *already-filtered* output of the four
classes documented below.

## Why temporal filtering is required

Without it: pick up your phone for one frame's worth of motion blur → false
`PHONE_DISTRACTION`. Blink → false `DROWSINESS_SIGNAL`. A single frame where MediaPipe
loses your face as you turn your head slightly → a flicker between states every frame.
PRD §15 states the requirement plainly: prevent a raw `YES / NO / YES / NO / YES`
signal from becoming repeated events, and do it **based on timestamps/durations, never
assuming a fixed FPS** — because FPS is not constant (throttled YOLO calls, variable
camera/model latency, occasional dropped frames all mean "10 frames" is not a
consistent amount of real time).

## The generic primitives (`src/state/temporal_filter.py`)

Every filter below is built from one or more of these four reusable, pure,
timestamp-based building blocks — none of them know anything about phones, eyes, or
heads:

| Primitive | Purpose | States/shape |
|---|---|---|
| `DurationConfirmer` | Generic confirm/clear-with-optional-grace-period state machine | `INACTIVE → CONFIRMING → CONFIRMED → CLEARING → INACTIVE` |
| `hysteresis()` | Pure dead-zone classification (below low = False, above high = True, between = keep previous) | stateless function |
| `Debouncer` | Suppresses rapid raw-signal transitions before confirmation logic sees them | — |
| `Cooldown` | Throttles how often a side effect (a sound, a log entry) may fire | — |

All boundary comparisons use a tiny epsilon (`1e-9`) to absorb floating-point
subtraction noise at realistic `time.monotonic()` magnitudes — a lesson learned and
then applied consistently everywhere in the codebase (see the epsilon comment in
`temporal_filter.py` and its reuse in `head_orientation_filter.py`,
`app.py`'s YOLO-interval gate, and `audio_manager.py`'s persistent-reminder timer).

## The four condition-specific filters

### 1. `PhoneTemporalFilter` (`src/state/phone_temporal_filter.py`)

| | |
|---|---|
| **Input** | `phone_detected: bool` (from `PerceptionSnapshot.phone_detected`) + timestamp |
| **Internal state** | `NOT_DETECTED → CONFIRMING → CONFIRMED → CLEARING → NOT_DETECTED` |
| **Confirm threshold** | `phone.confirm_duration_seconds` (default `0.35s`) |
| **Clear threshold (grace period)** | `phone.clear_duration_seconds` (default `0.60s`) |
| **Reset behavior** | Reappearing during `CLEARING` returns straight to `CONFIRMED` without restarting confirmation — this is what prevents `CONFIRMED → NOT_DETECTED → CONFIRMING → CONFIRMED` flapping from one missed detection |
| **Output** | `PhoneFilterResult(state, is_confirmed, just_confirmed, just_cleared, timestamp)` |

Example timeline (defaults):

```text
t=0.00  phone appears           -> CONFIRMING
t=0.35  still detected          -> CONFIRMED, just_confirmed=True   (PHONE_DISTRACTION becomes possible)
t=0.50  phone briefly obscured  -> CLEARING (grace period started)
t=0.80  phone reappears         -> back to CONFIRMED directly (no re-confirmation, no flap)
t=1.20  phone truly gone        -> CLEARING
t=1.80  still gone (0.60s)      -> NOT_DETECTED, just_cleared=True
```

### 2. `HeadOrientationFilter` (`src/state/head_orientation_filter.py`)

| | |
|---|---|
| **Input** | `HeadOrientation` enum + timestamp |
| **Internal state** | `CENTERED → DIVERTING → DIVERTED` |
| **Confirm threshold** | `head.confirmation_seconds` (default `0.80s`) |
| **Clear behavior** | **Immediate** — no grace period (PRD's `head:` config defines only one duration) |
| **UNKNOWN handling** | `UNKNOWN` counts as centered for timer purposes — unreliable geometry must never itself count as evidence of diversion |
| **Direction switching** | Switching directly between off-center directions (e.g. `LEFT` to `RIGHT`) without passing through `CENTER` does **not** reset the timer — PRD only cares about "outside center", not which direction |
| **Output** | `HeadOrientationFilterResult(state, raw_orientation, is_diverted, just_diverted, just_restored, timestamp)` |

Example timeline:

```text
t=0.00  look left           -> DIVERTING
t=0.50  switch to look right (still off-center) -> still DIVERTING, timer NOT reset
t=0.80  still off-center     -> DIVERTED, just_diverted=True
t=0.90  look back to center  -> CENTERED immediately, just_restored=True (no grace period)
```

### 3. `DrowsinessFilter` (`src/state/drowsiness_filter.py`)

A thin domain wrapper around `DurationConfirmer` — reuses the generic primitive instead
of reimplementing confirm/clear logic.

| | |
|---|---|
| **Input** | `EyeState` enum + timestamp |
| **"Active" means** | `EyeState.CLOSED` (never `UNKNOWN` — missing landmarks are treated the same as *open*, per PRD §10: "never classify missing landmarks as closed eyes") |
| **Confirm threshold** | `eyes.drowsiness_duration_seconds` (default `1.20s`) |
| **Clear behavior** | Immediate once eyes are no longer closed |
| **Output** | `DrowsinessFilterResult(state, is_drowsy, just_confirmed, just_cleared, timestamp)` |

**The blink/drowsiness distinction requires no special-case code at all.** A blink is
simply a closure shorter than `drowsiness_duration_seconds` — it enters `CONFIRMING`
and returns to `INACTIVE` before ever reaching `CONFIRMED`, so it never generates a
signal. `eyes.blink_max_duration_seconds` exists purely as a documentation/tuning value
for "what counts as a normal blink"; it isn't read by this filter at all.

```text
t=0.00  eyes close             -> CONFIRMING
t=0.30  eyes reopen (a blink)  -> back to INACTIVE, nothing fired
--- vs. ---
t=0.00  eyes close             -> CONFIRMING
t=1.20  still closed            -> CONFIRMED, just_confirmed=True   (DROWSINESS_SIGNAL)
t=1.50  eyes reopen             -> INACTIVE, just_cleared=True (immediate)
```

### 4. `PersonAwayFilter` (`src/state/person_away_filter.py`)

Also a thin `DurationConfirmer` wrapper.

| | |
|---|---|
| **Input** | `person_present: bool` (from `PerceptionSnapshot.person_present`) + timestamp |
| **"Active" means** | `not person_present` |
| **Confirm threshold** | `person.away_duration_seconds` (default `3.0s`) |
| **Clear behavior** | Immediate once the person is present again |
| **Output** | `PersonAwayFilterResult(state, is_away, just_confirmed, just_cleared, timestamp)` |

```text
t=0.0  person leaves frame     -> CONFIRMING
t=2.0  person returns          -> back to INACTIVE, nothing fired ("no AWAY event if they return before the threshold" — PRD §14)
--- vs. ---
t=0.0  person leaves frame     -> CONFIRMING
t=3.0  still gone               -> CONFIRMED, just_confirmed=True   (AWAY)
t=5.0  person returns           -> INACTIVE, just_cleared=True (PERSON_RETURNED)
```

## Timer visibility (debug mode)

All four filters expose `elapsed_in_state_seconds(now)` — returns the seconds elapsed
since the current transitional state began (`CONFIRMING`/`CLEARING`), or `None` while
steady (`INACTIVE`/`CONFIRMED`). This is a purely additive read accessor (added on top
of the existing state machines, changing no existing behavior) that exists specifically
to drive the debug overlay's confirmation timers (PRD §24). See
[`AUDIO_SYSTEM.md`](AUDIO_SYSTEM.md) for a related but distinct timer — the persistent
audio-reminder interval, which is a completely separate mechanism.

## Edge cases handled by design (not by special-casing)

- **Duplicate/zero-elapsed timestamps** — every filter validates monotonicity
  (`ValueError` on an out-of-order timestamp) but tolerates two calls at the exact same
  timestamp (zero elapsed, simply doesn't advance any timer).
- **Rapid oscillation shorter than the confirm duration never confirms** — this is the
  entire point of the confirm-duration mechanism, verified directly by dedicated tests
  for every filter (e.g. `test_rapid_oscillation_shorter_than_confirm_duration_never_confirms`).
- **Zero-duration configuration** — every filter correctly confirms/clears
  *immediately* if its threshold is configured to `0.0` (useful for tests, and a
  legitimate configuration choice, not a special case in the code).
- **Long gaps between calls use timestamp math, not call count** — a filter that
  receives only two calls with a 100-second gap between them confirms correctly on the
  second call, exactly as PRD §15 requires.
