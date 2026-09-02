# Project Evolution

## A note on phase numbering (read this first)

Two different numbering schemes appear throughout this repository's history, and
they're easy to conflate:

- **`FOCUSGUARD_PRD.md` §39** numbers phases 0–14.
- **The repository's actual branch/PR names** (`feature/phase-0-...` through
  `feature/phase-13-...`) run **one number behind** the PRD's numbers for every phase
  after the first, because the repo's own "Phase 0" branch commit combined the PRD's
  Phase 0 (environment) and Phase 1 (project structure/config) into a single
  foundational commit — confirmed independently, repeatedly, by cross-checking stub
  file docstrings (which cite the PRD's own numbers directly) against branch names.

The table below shows **both** numbers side by side.

## Timeline

| Repo branch # | PRD Phase # | PR | Feature | Why it was added | Architectural significance |
|---|---|---|---|---|---|
| 0 | 0 + 1 | direct commit | Project foundation, config loading | Establish the skeleton and a validated config before any real logic | Set the precedent: `ConfigManager` validates everything at startup, nothing runs on bad config |
| 1 | 2 | #1 | `CameraManager` | First real hardware boundary | Established the injectable-backend pattern (`capture_factory`) every later hardware-touching module reused |
| — | — | #2 | README/workflow docs | Document the branching/PR process | (docs only) |
| 2 | 3 | #3 | `YOLODetector` | Person/phone detection | Same injectable pattern reused (`model_factory`); introduced `Detection` as the shared detector-output contract |
| 3 | 5 | #4 | `FaceAnalyzer` + eye metrics | Face landmarks, eye openness | Introduced hysteresis classification (`EyeState`) and the "never classify missing data as the worst case" rule that recurs everywhere later |
| 4 | 4 | #5 | `PhoneTemporalFilter` | Phone-distraction temporal confirmation | First concrete instance of "detected once ≠ confirmed" — proved the pattern before it was generalized |
| 5 | 6 | #6 | Head orientation + `HeadOrientationFilter` | Approximate attention-diversion signal | Second independent temporal filter, still hand-written (not yet generalized) |
| 6 | 7 | #7 | Generic temporal-filtering toolkit (`DurationConfirmer`, `hysteresis()`, `Debouncer`, `Cooldown`) | Recognized that phone and head filters shared the same shape | **Key refactor-by-addition**: generalized the pattern *without* touching the two already-working filters — they were deliberately left alone and only new consumers (drowsiness, away, audio) were built on the new toolkit |
| 7 | 8 | #8 | `PerceptionSnapshot`, primary-person selection, `DrowsinessFilter`, `PersonAwayFilter`, `StateManager`, `EventManager` | The largest single phase: everything needed to go from raw signals to one decided status and a logged history | Established `StateManager`'s strict priority-`elif` chain and the `UNKNOWN`-vs-`AWAY`-via-`VisionQuality` distinction that the rest of the project depends on |
| 8 | 9 | #9 | Pygame dashboard (`UIManager`, `DashboardView`) | Visible output | Enforced "no CV logic in rendering code" as a real architectural boundary, not just a guideline — `UIManager` imports nothing CV-related |
| 9 | 10 | #10 | `AudioManager` | Warnings and music | First manager deliberately built to know *nothing* about `EventType`/`FocusState` — proved the "granular methods, caller decides when" pattern |
| 10 | 11 | #11 | `SessionManager` | Duration/streak/count/score tracking, JSON persistence | The one manager that *does* import `FocusState`/`EventType` — a documented, deliberate exception to the independence rule, because deriving statistics is its entire job |
| 11 | 12 | #12 | `FocusGuardApp` — full integration | Wire every standalone manager into one running app for the first time | The single largest architectural event: everything built independently across ten phases had to compose correctly on the first real attempt. It did, with only wiring code added — no manager's internals needed to change. Also added `DashboardView.paused` (the one small addition to an already-merged file, needed because `FocusState` has no PAUSED value) |
| 12 | 13 | #13 | YOLO detection-interval throttling | Real-hardware measurement showed FPS (~11) below the 20–30 target | First phase driven by *measurement* rather than a written spec — added `yolo.detection_interval_seconds` after proving the need with real numbers (see [`PERFORMANCE.md`](PERFORMANCE.md)) |
| 13 | 14 | #14 | README, demo walkthrough, config comments, LICENSE | Prepare the completed project for a reader (interviewer, new contributor, future self) | Documentation-only phase; verified the MediaPipe model download URL against the live official source rather than guessing |
| *(unmerged)* | *(not in PRD)* | — | Persistent audio reminders | Explicitly requested **after** V1 was complete: don't just warn once, keep reminding while a distraction continues | The first post-V1 feature. Implemented entirely inside `AudioManager` (a new private `_PersistentReminder` class) with the existing one-shot warning methods left completely untouched — a direct payoff of the independence rules established from phase 9 onward |

## What the shape of this history shows

1. **Independent modules were built in dependency order, not integration order.**
   Nine phases (1–9 by PRD numbering, camera through session) were each built,
   tested, and merged **before** anything wired them together in phase 12. This was
   only possible because none of them depended on each other's internals — each
   could be fully specified and tested against fakes without the others existing yet.
2. **Generalization happened once, retroactively, without breaking what already
   worked** (phase 7/PRD-7: the temporal-filtering toolkit). This is a concrete
   example of recognizing a repeated pattern *after* seeing it twice, rather than
   over-engineering an abstraction on the first instance.
3. **The integration phase (12/PRD-12) needed almost no new logic** — it's
   essentially the only phase whose diff is dominated by *calling* existing methods
   in the right order, rather than writing new decision logic. That's the direct
   result of every prior phase's discipline about not depending on siblings.
4. **The one post-V1 feature (persistent audio reminders) required zero changes**
   to detection, face analysis, temporal filters, the state machine, the event
   system, or session analytics — it was implementable entirely inside the one
   manager it conceptually belonged to. This is the architecture being validated by
   a real, unplanned change request rather than by design intent alone.
