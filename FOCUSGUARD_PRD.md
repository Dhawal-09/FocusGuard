\# FOCUSGUARD V1 — PRODUCT REQUIREMENTS \& TECHNICAL DESIGN DOCUMENT



\## 1. Project Objective



Build FocusGuard V1, a local desktop computer-vision application that monitors a user's study/work environment through a webcam and provides estimated focus feedback.



The system must demonstrate:



\* Python

\* OpenCV

\* YOLO object detection

\* facial landmark processing

\* eye-state analysis

\* approximate head orientation

\* temporal filtering

\* event detection

\* state machines

\* real-time processing

\* Pygame UI

\* audio feedback

\* session analytics

\* graceful error handling



This is a portfolio/technical demonstration.



It is NOT a medical, psychological, biometric, or scientifically validated attention-monitoring system.



Never claim medical-grade drowsiness detection or scientifically accurate attention measurement.



Use terminology such as:



\* Estimated Focus

\* Focus State

\* Potential Distraction

\* Drowsiness Signal

\* Attention Diverted



\---



\# 2. MVP SCOPE



\## MUST BUILD



1\. Webcam capture

2\. Person detection

3\. Cell-phone detection

4\. Face detection/landmarks

5\. Eye openness detection

6\. Blink vs prolonged eye closure

7\. Approximate head orientation

8\. Person-away detection

9\. Temporal filtering

10\. State machine

11\. Event manager

12\. Pygame UI

13\. Audio alerts

14\. Optional background focus music

15\. Session timer

16\. Estimated focus score

17\. Session analytics

18\. Event log

19\. Debug mode

20\. Configuration file

21\. CPU fallback

22\. Graceful error handling

23\. Unit tests

24\. README



\## DO NOT BUILD



Do not implement:



\* authentication

\* user accounts

\* cloud backend

\* web application

\* mobile application

\* database server

\* cloud analytics

\* payment system

\* custom ML training

\* dataset pipeline

\* browser blocking

\* face recognition

\* identity recognition

\* biometric database

\* automatic webcam recording

\* screenshots by default

\* multiprocessing unless clearly necessary

\* unnecessary framework abstractions



\---



\# 3. TECHNOLOGY



Use Python.



Preferred technologies:



\* OpenCV

\* Ultralytics YOLO

\* lightweight pretrained YOLO model

\* MediaPipe Face Landmarker or current compatible facial-landmark solution

\* NumPy

\* Pygame

\* PyYAML

\* pytest



Use current stable compatible APIs.



If a library API differs from this specification, adapt to the current API rather than using deprecated code.



Keep dependencies minimal.



\---



\# 4. MODEL STRATEGY



Use a lightweight pretrained YOLO model.



Preferred MVP baseline:



YOLO11n.



Use YOLO for:



\* person

\* cell phone



Do not use YOLO for eye closure.



Use facial landmarks for:



\* face

\* eye landmarks

\* eye openness

\* approximate head orientation



Do not train a custom model in V1.



The detector must be abstracted so that a future custom YOLO model can replace the pretrained model without rewriting the rest of the application.



\---



\# 5. USER FLOW



Application flow:



Launch

↓

Validate configuration

↓

Initialize Pygame

↓

Initialize camera

↓

Load YOLO

↓

Load face-landmark model

↓

Show camera preview

↓

IDLE

↓

User starts session

↓

Real-time monitoring

↓

Detection

↓

Temporal filtering

↓

State evaluation

↓

Events

↓

Audio/UI/session updates

↓

User ends session

↓

Session summary

↓

Return to idle or exit



\---



\# 6. CAMERA



Create:



CameraManager



Responsibilities:



\* open webcam

\* configure resolution

\* read frames

\* detect invalid frames

\* release camera

\* handle camera failures

\* expose frame timestamps



Handle:



\* no camera

\* camera already in use

\* camera permission failure

\* camera disconnect

\* invalid frame

\* frame-read failure



Do not silently crash.



Display a human-readable error.



\---



\# 7. YOLO DETECTION



Create:



YOLODetector



Responsibilities:



\* load model

\* select CPU/GPU

\* run inference

\* return detections

\* filter confidence

\* measure inference time



Detection structure should contain:



```text

class\_name

confidence

x1

y1

x2

y2

timestamp

```



For V1 only care about:



```text

person

cell phone

```



Initial configuration:



```yaml

yolo:

&#x20; model: "models/yolo11n.pt"

&#x20; confidence: 0.45

&#x20; phone\_confidence: 0.55

```



\---



\# 8. PRIMARY PERSON



For V1, if multiple people are visible:



Choose the largest person bounding box.



The largest detected person is considered the primary user.



Only the primary person should influence face/head/eye analysis.



Phone detection should preferably be associated with the primary person using simple spatial heuristics where practical.



Do not implement complex multi-person tracking.



\---



\# 9. PHONE DETECTION



Phone detection must NOT trigger immediately from one frame.



Use temporal confirmation.



Initial behavior:



```text

phone detected

↓

start confirmation timer

↓

phone remains detected

↓

PHONE\_CONFIRMED

```



Suggested initial configuration:



```yaml

phone:

&#x20; confidence: 0.55

&#x20; confirm\_duration\_seconds: 0.35

&#x20; clear\_duration\_seconds: 0.60

&#x20; warning\_cooldown\_seconds: 10

```



A single transient detection must not generate an event.



When confirmed:



```text

PHONE\_DETECTED

PHONE\_DISTRACTION

```



When the phone disappears for the configured clear duration:



```text

PHONE\_CLEARED

```



Audio must respect cooldown.



Never play the warning every frame.



\---



\# 10. FACE ANALYSIS



Create:



FaceAnalyzer



Responsibilities:



\* detect/track primary face

\* obtain facial landmarks

\* calculate eye metrics

\* estimate head orientation

\* return confidence/validity information



If the face cannot be reliably analyzed:



```text

FACE = UNKNOWN

EYES = UNKNOWN

HEAD = UNKNOWN

```



Never classify missing landmarks as closed eyes.



\---



\# 11. EYE ANALYSIS



Use facial landmarks.



Calculate an eye-openness metric such as Eye Aspect Ratio or an equivalent landmark-based metric.



Initial configuration:



```yaml

eyes:

&#x20; closed\_threshold: 0.21

&#x20; open\_threshold: 0.24

&#x20; blink\_max\_duration\_seconds: 0.45

&#x20; drowsiness\_duration\_seconds: 1.20

```



Use hysteresis:



```text

closed when metric < 0.21

open when metric > 0.24

```



Values between thresholds retain the previous state where valid.



States:



```text

OPEN

CLOSED

UNKNOWN

```



\---



\# 12. BLINK VS DROWSINESS



A normal blink must NOT generate a drowsiness event.



Logic:



```text

Eyes OPEN

↓

Eyes CLOSED

↓

closure duration < blink threshold

↓

BLINK

↓

Eyes OPEN

```



Prolonged closure:



```text

Eyes OPEN

↓

Eyes CLOSED

↓

closure persists

↓

duration >= drowsiness threshold

↓

DROWSINESS\_SIGNAL

```



Suggested initial threshold:



```text

1.20 seconds

```



This is configurable and must be tuned on the user's camera.



Do not claim this scientifically detects sleep.



\---



\# 13. HEAD ORIENTATION



Implement approximate head orientation.



Possible states:



```text

CENTER

LEFT

RIGHT

UP

DOWN

UNKNOWN

```



Estimate approximate yaw/pitch using facial landmarks/head-pose geometry.



Initial thresholds:



```yaml

head:

&#x20; yaw\_threshold\_degrees: 20

&#x20; pitch\_threshold\_degrees: 18

&#x20; confirmation\_seconds: 0.80

```



The exact implementation may use an appropriate landmark/head-pose method.



This is an approximate attention-diversion signal.



It is not gaze tracking.



\---



\# 14. PERSON AWAY



If the primary person disappears:



Do NOT immediately enter AWAY.



Start a timer.



Suggested:



```yaml

person:

&#x20; away\_duration\_seconds: 3.0

```



Behavior:



```text

PERSON PRESENT

↓

PERSON NOT DETECTED

↓

start timer

↓

still absent after 3 sec

↓

AWAY

```



If person returns before threshold:



No AWAY event.



When person returns after AWAY:



```text

PERSON\_RETURNED

```



and restore appropriate monitoring state.



\---



\# 15. TEMPORAL FILTERING



Create reusable:



TemporalFilter



The system must support:



\* confirmation duration

\* clear duration

\* debounce

\* hysteresis

\* cooldown



The goal is to prevent:



```text

YES

NO

YES

NO

YES

```



from becoming repeated events.



Temporal filtering must be based on timestamps/durations rather than assuming a fixed FPS.



\---



\# 16. PERCEPTION SNAPSHOT



Create a central immutable/current-frame perception structure.



Example:



```python

PerceptionSnapshot(

&#x20;   timestamp,

&#x20;   person\_present,

&#x20;   primary\_person,

&#x20;   phone\_detected,

&#x20;   phone\_confidence,

&#x20;   face\_present,

&#x20;   eyes\_state,

&#x20;   eye\_metric,

&#x20;   head\_orientation,

&#x20;   head\_yaw,

&#x20;   head\_pitch,

&#x20;   vision\_quality

)

```



The state machine consumes stable/filtered signals rather than raw detector output.



\---



\# 17. STATES



Implement:



```text

IDLE

FOCUSED

PHONE\_DISTRACTION

DROWSINESS\_SIGNAL

ATTENTION\_DIVERTED

AWAY

UNKNOWN

```



Definitions:



\## IDLE



No active session.



\## FOCUSED



Person present, no confirmed phone distraction, no prolonged eye closure, and head approximately centered.



\## PHONE\_DISTRACTION



Phone has been temporally confirmed.



\## DROWSINESS\_SIGNAL



Eyes have remained closed beyond configured duration.



\## ATTENTION\_DIVERTED



Head orientation has remained outside the center threshold beyond configured duration.



\## AWAY



Person absent beyond configured duration.



\## UNKNOWN



Vision information is insufficient to make a reliable classification.



\---



\# 18. STATE PRIORITY



When multiple conditions occur simultaneously, use:



```text

AWAY

↓

PHONE\_DISTRACTION

↓

DROWSINESS\_SIGNAL

↓

ATTENTION\_DIVERTED

↓

FOCUSED

↓

UNKNOWN

```



However, UNKNOWN must be used whenever required perception is unavailable rather than falsely assuming FOCUSED.



Priority must be implemented deterministically.



\---



\# 19. STATE TRANSITIONS



Minimum transitions:



```text

IDLE → FOCUSED

IDLE → UNKNOWN



FOCUSED → PHONE\_DISTRACTION

PHONE\_DISTRACTION → FOCUSED



FOCUSED → DROWSINESS\_SIGNAL

DROWSINESS\_SIGNAL → FOCUSED



FOCUSED → ATTENTION\_DIVERTED

ATTENTION\_DIVERTED → FOCUSED



FOCUSED → AWAY

AWAY → FOCUSED



Any active state → UNKNOWN

UNKNOWN → appropriate valid state

```



Do not generate repeated state-change events if the state has not changed.



\---



\# 20. EVENTS



Create:



Event



Fields:



```text

event\_type

timestamp

severity

metadata

```



Event types:



```text

SESSION\_STARTED

SESSION\_ENDED



PHONE\_DETECTED

PHONE\_CLEARED



DROWSINESS\_SIGNAL

DROWSINESS\_CLEARED



ATTENTION\_DIVERTED

ATTENTION\_RESTORED



PERSON\_LEFT

PERSON\_RETURNED



FOCUS\_RESTORED



CAMERA\_ERROR

MODEL\_ERROR

VISION\_ERROR

```



Events are generated on meaningful transitions, not every frame.



\---



\# 21. AUDIO SYSTEM



Create:



AudioManager



Use Pygame mixer.



Support:



```text

phone warning

drowsiness warning

attention warning

focus restored sound

session complete sound

background focus music

```



Suggested assets:



```text

assets/

├── sounds/

│   ├── phone\_warning.mp3

│   ├── drowsiness\_warning.mp3

│   ├── attention\_warning.mp3

│   ├── focus\_restored.mp3

│   └── session\_complete.mp3

└── music/

&#x20;   └── focus\_music.mp3

```



Audio files are user-provided assets.



Do not download copyrighted music automatically.



Configuration:



```yaml

audio:

&#x20; enabled: true

&#x20; volume: 0.70

&#x20; music\_enabled: false

&#x20; music\_volume: 0.25

```



Missing audio files must never crash the application.



Audio warnings must have cooldowns.



Optional behavior:



```text

Focus music playing

↓

phone distraction

↓

pause music

↓

play warning

↓

focus restored

↓

resume music

```



\---



\# 22. PYGAME UI



Create a clean desktop dashboard.



Display:



```text

FOCUSGUARD



\[ CAMERA FEED ]



STATUS

FOCUSED



PERSON

Detected



PHONE

Not Detected



EYES

Open



HEAD

Center



SESSION

00:32:14



FOCUS SCORE

91



FPS

28



INFERENCE

32 ms



EVENT LOG



10:22:10 Session started

10:28:41 Phone detected

10:29:03 Focus restored

```



UI must remain responsive.



Do not put CV logic inside the rendering code.



\---



\# 23. CONTROLS



Minimum controls:



```text

SPACE → Start/Pause/Resume session

Q / ESC → Exit

M → Toggle mute

D → Toggle debug mode

R → Reset session if safe

```



If a control conflicts with Pygame/window behavior, choose a sensible alternative and document it.



\---



\# 24. DEBUG MODE



Debug mode should display:



\* YOLO bounding boxes

\* class names

\* confidence

\* face landmarks

\* eye metric

\* head yaw

\* head pitch

\* current state

\* confirmation timers

\* away timer

\* FPS

\* inference latency

\* vision quality



Debug mode is primarily for development and interviews.



\---



\# 25. SESSION MANAGER



Create:



SessionManager



Track:



```text

session start

session end

total duration

focused duration

phone distraction duration/count

drowsiness count

attention diversion count

away count

longest focus streak

focus score

events

```



Do not use a database.



Use in-memory session data and optionally save a JSON summary.



\---



\# 26. FOCUS SCORE



Implement a simple configurable demonstration score.



Start:



```text

100

```



Possible deductions:



```yaml

score:

&#x20; starting\_score: 100

&#x20; phone\_event\_penalty: 10

&#x20; drowsiness\_event\_penalty: 5

&#x20; attention\_event\_penalty: 3

&#x20; away\_event\_penalty: 5

```



Score must not go below 0.



This is a demonstration metric.



Display:



```text

ESTIMATED FOCUS SCORE

```



Do not call it scientifically accurate.



\---



\# 27. ANALYTICS



At session end display:



```text

Session Duration

Focused Duration

Phone Distractions

Drowsiness Signals

Attention Diversions

Away Events

Longest Focus Streak

Estimated Focus Score

```



Save optional JSON:



```text

logs/session\_YYYYMMDD\_HHMMSS.json

```



Do not store webcam frames.



\---



\# 28. EVENT LOG



Maintain an in-memory bounded event log.



Maximum:



```yaml

session:

&#x20; max\_event\_log\_entries: 100

```



Display the latest events in the UI.



Each event should include timestamp and event type.



\---



\# 29. PERFORMANCE



Target:



```text

20–30 displayed FPS where hardware permits

```



Do not require YOLO inference at every displayed frame.



The implementation may:



```text

render frame at high frequency

run detection at controlled intervals

reuse latest detection result

```



if necessary.



Measure:



```text

FPS

YOLO inference latency

face-analysis latency if practical

```



Do not optimize prematurely.



First measure.



CPU must work.



If NVIDIA CUDA is available, GPU may be used.



Device selection:



```text

auto

cpu

cuda

```



Fallback to CPU if CUDA is unavailable.



\---



\# 30. CONFIGURATION



Use:



```text

config/config.yaml

```



Include:



```yaml

camera:

&#x20; index: 0

&#x20; width: 1280

&#x20; height: 720

&#x20; target\_fps: 30



yolo:

&#x20; model: models/yolo11n.pt

&#x20; confidence: 0.45

&#x20; phone\_confidence: 0.55

&#x20; device: auto



phone:

&#x20; confirm\_duration\_seconds: 0.35

&#x20; clear\_duration\_seconds: 0.60

&#x20; warning\_cooldown\_seconds: 10



eyes:

&#x20; closed\_threshold: 0.21

&#x20; open\_threshold: 0.24

&#x20; blink\_max\_duration\_seconds: 0.45

&#x20; drowsiness\_duration\_seconds: 1.20



head:

&#x20; yaw\_threshold\_degrees: 20

&#x20; pitch\_threshold\_degrees: 18

&#x20; confirmation\_seconds: 0.80



person:

&#x20; away\_duration\_seconds: 3.0



audio:

&#x20; enabled: true

&#x20; volume: 0.70

&#x20; music\_enabled: false

&#x20; music\_volume: 0.25



ui:

&#x20; debug: false



score:

&#x20; starting\_score: 100

&#x20; phone\_event\_penalty: 10

&#x20; drowsiness\_event\_penalty: 5

&#x20; attention\_event\_penalty: 3

&#x20; away\_event\_penalty: 5



session:

&#x20; max\_event\_log\_entries: 100

```



Validate configuration at startup.



Invalid values must produce readable errors.



\---



\# 31. PRIVACY



All webcam processing must occur locally.



Do not:



\* upload frames

\* call cloud vision APIs

\* store webcam frames

\* implement face recognition

\* create biometric profiles



The README must clearly state:



"FocusGuard processes webcam input locally and does not upload webcam frames."



Any future screenshot/dataset capture must be explicit and opt-in.



\---



\# 32. PROJECT STRUCTURE



Use a small modular structure:



```text

focusguard/

│

├── main.py

├── README.md

├── requirements.txt

├── .gitignore

│

├── config/

│   └── config.yaml

│

├── models/

│   └── .gitkeep

│

├── assets/

│   ├── sounds/

│   └── music/

│

├── logs/

│   └── .gitkeep

│

├── src/

│   ├── camera/

│   │   └── camera\_manager.py

│   │

│   ├── detection/

│   │   ├── yolo\_detector.py

│   │   └── detection\_types.py

│   │

│   ├── face/

│   │   ├── face\_analyzer.py

│   │   ├── eye\_metrics.py

│   │   └── head\_pose.py

│   │

│   ├── state/

│   │   ├── temporal\_filter.py

│   │   └── state\_manager.py

│   │

│   ├── events/

│   │   └── event\_manager.py

│   │

│   ├── audio/

│   │   └── audio\_manager.py

│   │

│   ├── session/

│   │   └── session\_manager.py

│   │

│   ├── ui/

│   │   └── ui\_manager.py

│   │

│   └── core/

│       └── types.py

│

└── tests/

&#x20;   ├── test\_temporal\_filter.py

&#x20;   ├── test\_eye\_metrics.py

&#x20;   ├── test\_state\_manager.py

&#x20;   ├── test\_event\_manager.py

&#x20;   └── test\_session\_manager.py

```



The agent may slightly adjust this structure if required for correct Python packaging, but should avoid unnecessary complexity.



\---



\# 33. MODULE RESPONSIBILITIES



\## CameraManager



Only camera operations.



\## YOLODetector



Only object detection.



\## FaceAnalyzer



Only facial landmark/face analysis.



\## TemporalFilter



Only temporal stabilization.



\## StateManager



Only state transitions.



\## EventManager



Only event generation/logging.



\## AudioManager



Only audio.



\## SessionManager



Only session statistics and persistence.



\## UIManager



Only presentation/input.



\## ConfigManager



Only configuration loading/validation.



\---



\# 34. MAIN LOOP



Preferred flow:



```text

capture frame

↓

run YOLO according to detection interval

↓

run face analysis

↓

create perception snapshot

↓

update temporal filters

↓

evaluate state

↓

generate events

↓

send events to audio

↓

update session

↓

render UI

↓

calculate FPS

↓

next frame

```



The UI must not block the CV pipeline unnecessarily.



Do not introduce multiprocessing unless profiling demonstrates a real need.



\---



\# 35. ERROR HANDLING



Handle:



\* missing model

\* invalid model

\* model loading failure

\* model inference exception

\* missing face model

\* face model initialization failure

\* camera failure

\* invalid frame

\* Pygame initialization failure

\* missing sound

\* missing music

\* invalid config

\* malformed config

\* runtime exception



Errors should be:



1\. logged

2\. displayed where possible

3\. actionable

4\. non-silent



Optional subsystems such as audio should fail gracefully without taking down the application.



\---



\# 36. TESTING



Unit-test logic that does not require the webcam.



Phone filter:



```text

single detection → no confirmed event

persistent detection → confirmed event

phone disappearance → cleared

cooldown → no repeated event

```



Eye logic:



```text

short closure → blink

long closure → drowsiness

missing landmarks → UNKNOWN

```



Away:



```text

short absence → no away

long absence → away

return → person returned

```



State machine:



```text

focused → phone

phone → focused

focused → drowsiness

focused → attention

focused → away

```



Priority:



```text

phone + drowsiness → PHONE\_DISTRACTION

away + phone → AWAY

```



Score:



```text

starting score = 100

penalties applied correctly

minimum = 0

```



Cooldown:



```text

same event cannot repeatedly trigger audio/event every frame

```



\---



\# 37. ACCEPTANCE CRITERIA



\## Camera



\* webcam opens

\* frames display

\* ESC/Q exits

\* camera releases cleanly

\* camera failure produces readable error



\## YOLO



\* model loads

\* person detection works

\* phone detection works

\* confidence shown in debug

\* inference latency measured



\## Phone



\* transient detection ignored

\* confirmed phone generates event

\* phone disappearance clears state

\* audio does not repeat every frame



\## Eyes



\* face detected

\* eye metric calculated

\* blink does not trigger drowsiness

\* prolonged closure triggers drowsiness

\* missing face produces UNKNOWN



\## Head



\* center approximately detected

\* left/right approximately detected

\* up/down approximately detected

\* temporal confirmation works



\## Away



\* person disappearance starts timer

\* threshold produces AWAY

\* return restores monitoring



\## State



\* state transitions are deterministic

\* simultaneous signals respect priority

\* no frame-by-frame state flapping



\## Audio



\* warnings play

\* mute works

\* volume works

\* missing files do not crash



\## UI



\* camera feed visible

\* status visible

\* signals visible

\* timer works

\* event log works

\* FPS visible

\* inference latency visible

\* debug mode works

\* UI remains responsive



\## Session



\* start works

\* pause/resume works

\* end works

\* summary works

\* JSON output works



\## Privacy



\* no webcam frames uploaded

\* no webcam frames automatically stored

\* no face recognition

\* no cloud vision APIs



\---



\# 38. DEMO



The complete demo should take approximately 2–3 minutes.



1\. Launch:



```bash

python main.py

```



2\. Start session.



3\. Sit normally.



Expected:



```text

FOCUSED

```



4\. Pick up phone.



Expected:



```text

PHONE DISTRACTION

```



Play phone warning.



5\. Put phone away.



Expected:



```text

FOCUS RESTORED

```



6\. Close eyes for > configured drowsiness threshold.



Expected:



```text

DROWSINESS SIGNAL

```



Play drowsiness audio.



7\. Look left/right.



Expected:



```text

ATTENTION DIVERTED

```



8\. Leave camera.



Expected:



```text

AWAY

```



9\. Return.



Expected:



```text

FOCUS RESTORED

```



10\. End session.



Show:



```text

duration

focus duration

phone events

drowsiness events

attention events

away events

longest streak

estimated focus score

```



11\. Enable DEBUG mode and explain:



```text

YOLO boxes

confidence

eye metric

head pose

timers

FPS

inference latency

```



\---



\# 39. DEVELOPMENT PHASES



Implement strictly in this order.



\## PHASE 0



Environment and dependencies.



\## PHASE 1



Project structure/config.



\## PHASE 2



Camera.



\## PHASE 3



YOLO.



\## PHASE 4



Phone temporal filtering.



\## PHASE 5



Face and eye analysis.



\## PHASE 6



Head orientation.



\## PHASE 7



Temporal filtering architecture.



\## PHASE 8



State machine/events.



\## PHASE 9



Pygame UI.



\## PHASE 10



Audio/music.



\## PHASE 11



Session/score/analytics.



\## PHASE 12



Full integration/error handling.



\## PHASE 13



Testing/performance.



\## PHASE 14



README/demo/polish.



\---



\# 40. CODING AGENT RULES



The coding agent must:



1\. Read this entire document before implementation.

2\. Implement one phase at a time.

3\. Never silently skip a phase.

4\. Never add future-scope features.

5\. Run tests after relevant phases.

6\. Keep modules independent.

7\. Avoid magic numbers.

8\. Put thresholds in config.

9\. Never treat one noisy frame as an event.

10\. Never classify missing eye landmarks as closed.

11\. Never upload webcam frames.

12\. Never add cloud services.

13\. Avoid unnecessary dependencies.

14\. Use current compatible library APIs.

15\. Explain dependency/API changes.

16\. Preserve working code.

17\. Prefer simple solutions.

18\. Optimize only after measuring.

19\. Report errors instead of hiding them.

20\. Stop after each phase and wait for user confirmation.



\---



\# 41. PHASE REPORT FORMAT



After each phase, report:



```text

PHASE:

Objective:



Files created:



Files modified:



Implementation:



Tests executed:



Result:



Known issues:



My verification command:



Definition of Done:



Next phase:

```



Do not start the next phase until the user confirms.



\---



\# 42. FINAL COMMAND



The finished application must run using:



```bash

python main.py

```



The project must include:



```bash

pytest -q

```



for automated tests.



\---



\# 43. FUTURE V2 — DO NOT IMPLEMENT



Future versions may include:



\* custom FocusGuard dataset

\* custom YOLO training

\* hard-negative mining

\* improved phone/person association

\* ONNX

\* TensorRT

\* model quantization

\* better head-pose calibration

\* richer analytics

\* browser integration

\* application packaging

\* personalized calibration



These are explicitly outside V1.



\---



\# 44. FINAL ENGINEERING PRINCIPLE



FocusGuard V1 should be:



Small.



Functional.



Real-time.



Visually convincing.



Technically defensible.



Easy to explain in an interview.



Do not optimize for feature count.



Optimize for a complete and understandable computer-vision pipeline:



```text

Camera

→ Detection

→ Face Analysis

→ Temporal Filtering

→ State Machine

→ Events

→ Audio/UI

→ Session Analytics

```



The final implementation must prioritize reliability and explain ability over complexity.



