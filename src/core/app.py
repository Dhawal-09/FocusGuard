"""FocusGuardApp: full integration and error handling (PRD sections 5, 34, 35).

This is the only module that wires every previously-standalone manager
together into one running application - the main-loop step order matches
PRD section 34 exactly:

    capture frame -> run YOLO -> run face analysis -> build perception
    snapshot -> update temporal filters -> evaluate state -> generate
    events -> send events to audio -> update session -> render UI ->
    calculate FPS -> next frame

Every manager is constructed from AppConfig by default but is injectable
(camera_manager=..., yolo_detector=..., face_analyzer=..., ui_manager=...,
audio_manager=...) so FocusGuardApp itself is fully unit-testable by
reusing each manager's own existing fake-backend injection points
(CameraManager's capture_factory, YOLODetector's model_factory,
FaceAnalyzer's landmarker_factory, AudioManager's mixer/music/sound
backends, UIManager under a dummy SDL video driver) - no new mocking
layer is needed for the orchestration logic itself.

Error handling (PRD section 35): startup failures (Pygame/camera/model/
face-model) are fatal and reported with a readable message before a clean
shutdown. Per-frame YOLO/face-analysis inference exceptions are caught,
reported as a MODEL_ERROR/VISION_ERROR event, and degrade that single
frame to "no detections"/"no face" (never crashing the session or
treating the failure as silent). A camera read failure stops the
application cleanly rather than crashing or retry-looping indefinitely.
Audio subsystem failure at startup is caught and logged; the app
continues without sound - AudioManager's own play_*/music methods already
no-op safely whenever it was never successfully initialized.

Detection interval (PRD section 29): "do not require YOLO inference at
every displayed frame... run detection at controlled intervals, reuse
latest detection result... if necessary." A real-hardware benchmark
during Phase 13 measured YOLO at ~58ms/frame on CPU-only hardware (face
analysis ~17ms), yielding ~11fps - below the 20-30fps target - while the
orchestration layer itself (filters/state/events/session) measured at
>10,000 synthetic frames/sec, confirming YOLO inference as the actual
bottleneck. yolo.detection_interval_seconds throttles real YOLO calls to
at most once per that many seconds, reusing the last detection result
in between; face analysis, temporal filters, state evaluation, event
generation, and session recording all still run every frame regardless -
the PRD's main-loop diagram only qualifies YOLO with "according to
detection interval", and face analysis is cheap enough (and drowsiness
timing sensitive enough) to need no throttling.

Pause semantics: while paused (or before a session has ever started),
camera capture and UI rendering continue (a live preview stays visible
and the UI stays responsive), but the CV pipeline itself - detection,
face analysis, temporal filters, state evaluation, event generation, and
session recording - is skipped entirely. FocusState has no PAUSED value;
the state simply stops being re-evaluated while paused. DashboardView's
`paused` field (not the frozen FocusState) is what tells the UI to show
"PAUSED" instead of a stale status label.
"""

from __future__ import annotations

import time
from pathlib import Path

from src.audio.audio_manager import AudioError, AudioManager
from src.camera.camera_manager import CameraError, CameraManager, Frame
from src.core.config_manager import AppConfig
from src.core.types import PerceptionSnapshot, build_perception_snapshot
from src.detection.detection_types import Detection
from src.detection.yolo_detector import DetectionError, YOLODetector
from src.events.event_manager import EventManager
from src.face.eye_metrics import EyeState
from src.face.face_analyzer import FaceAnalysisError, FaceAnalysisResult, FaceAnalyzer
from src.face.head_pose import HeadOrientation, estimate_head_pose
from src.session.session_manager import SessionManager, SessionSummary
from src.state.drowsiness_filter import DrowsinessFilter
from src.state.head_orientation_filter import HeadOrientationFilter
from src.state.person_away_filter import PersonAwayFilter
from src.state.phone_temporal_filter import PhoneTemporalFilter
from src.state.state_manager import FocusState, StateManager
from src.ui.dashboard_view import DashboardView, DebugInfo, UIAction
from src.ui.ui_manager import UIError, UIManager

# Absorbs float64 subtraction noise at realistic time.monotonic() magnitudes,
# the same tolerance and reasoning src/state/temporal_filter.py documents for
# its own duration-boundary comparisons.
_DETECTION_INTERVAL_EPSILON_SECONDS = 1e-9


class FocusGuardApp:
    """Composes every FocusGuard manager and runs the real-time monitoring loop."""

    def __init__(
        self,
        config: AppConfig,
        *,
        camera_manager: CameraManager | None = None,
        yolo_detector: YOLODetector | None = None,
        face_analyzer: FaceAnalyzer | None = None,
        ui_manager: UIManager | None = None,
        audio_manager: AudioManager | None = None,
        logs_directory: Path | None = None,
    ) -> None:
        self._config = config
        self._logs_directory = logs_directory

        self._camera = camera_manager if camera_manager is not None else CameraManager(config.camera)
        self._yolo = yolo_detector if yolo_detector is not None else YOLODetector(config.yolo)
        self._face = face_analyzer if face_analyzer is not None else FaceAnalyzer(config.face, config.eyes)
        self._ui = ui_manager if ui_manager is not None else UIManager()
        self._audio = audio_manager if audio_manager is not None else AudioManager(config.audio, config.phone)

        self._phone_filter = PhoneTemporalFilter(config.phone)
        self._head_filter = HeadOrientationFilter(config.head)
        self._drowsiness_filter = DrowsinessFilter(config.eyes)
        self._away_filter = PersonAwayFilter(config.person)
        self._state_manager = StateManager()
        self._event_manager = EventManager(config.session, config.phone)
        self._session_manager = SessionManager(config.score)

        self._debug = config.ui.debug
        self._running = False
        self._session_start_timestamp: float | None = None

        self._fps: float | None = None
        self._last_frame_time: float | None = None
        self._last_snapshot: PerceptionSnapshot | None = None
        self._last_detections: list[Detection] = []
        self._last_face_result: FaceAnalysisResult | None = None
        self._last_yolo_timestamp: float | None = None

    # --- Top-level run -----------------------------------------------------------

    def run(self) -> int:
        """Startup, main loop, and shutdown. Returns a process exit code."""
        try:
            self._startup()
        except (CameraError, DetectionError, FaceAnalysisError, UIError) as exc:
            print(f"Fatal startup error: {exc}")
            self._shutdown()
            return 1

        try:
            self._main_loop()
        except Exception as exc:  # PRD section 35: runtime exceptions must be reported, never silent
            print(f"Unexpected runtime error: {exc}")
            return 1
        finally:
            self._shutdown()
        return 0

    def _startup(self) -> None:
        """PRD section 5 user flow: init Pygame -> init camera -> load YOLO
        -> load face model. Audio is best-effort - a failure here is
        caught and logged, never fatal (PRD section 35: "optional
        subsystems such as audio should fail gracefully")."""
        self._ui.init()
        self._camera.open()
        self._yolo.load()
        self._face.load()
        try:
            self._audio.init()
        except AudioError as exc:
            print(f"Audio unavailable, continuing without sound: {exc}")

    def _shutdown(self) -> None:
        """Release every subsystem. Safe to call even after a partial/failed
        startup - every manager's release/shutdown is itself safe to call
        multiple times or before init."""
        self._camera.release()
        self._audio.shutdown()
        self._ui.shutdown()

    def _main_loop(self) -> None:
        self._running = True
        while self._running:
            self._run_one_iteration()

    def _run_one_iteration(self) -> None:
        try:
            frame = self._camera.read_frame()
        except CameraError as exc:
            self._handle_camera_error(exc)
            return

        actions = self._ui.poll_input()
        self._handle_actions(actions, frame.timestamp)
        if not self._running:
            return

        if self._session_manager.is_active and not self._session_manager.is_paused:
            self._process_frame(frame)

        view = self._build_dashboard_view(frame)
        self._ui.render(view, frame.image)
        self._track_fps(frame.timestamp)

    def _handle_camera_error(self, exc: CameraError) -> None:
        """A camera read failure stops the app cleanly rather than crashing
        or retry-looping indefinitely - PRD section 6 requires a
        human-readable, non-silent error, not an automatic reconnect
        policy the PRD never specifies."""
        print(f"Camera error: {exc}")
        if self._session_manager.is_active:
            event = self._event_manager.camera_error(time.monotonic(), str(exc))
            self._session_manager.record_event(event)
        self._running = False

    # --- Per-frame CV pipeline (PRD section 34) -----------------------------------

    def _should_run_detection(self, timestamp: float) -> bool:
        """PRD section 29 detection interval: True on the very first frame
        (no prior detection to reuse), or once yolo.detection_interval_seconds
        has elapsed since the last real YOLO call. detection_interval_seconds
        of 0.0 means "every frame" - identical to the pre-Phase-13 behavior."""
        if self._last_yolo_timestamp is None:
            return True
        elapsed = timestamp - self._last_yolo_timestamp
        return elapsed >= self._config.yolo.detection_interval_seconds - _DETECTION_INTERVAL_EPSILON_SECONDS

    def _process_frame(self, frame: Frame) -> None:
        if self._should_run_detection(frame.timestamp):
            try:
                detections = self._yolo.detect(frame.image, frame.timestamp)
            except DetectionError as exc:
                print(f"YOLO inference error: {exc}")
                event = self._event_manager.model_error(frame.timestamp, str(exc))
                self._session_manager.record_event(event)
                detections = []
            self._last_yolo_timestamp = frame.timestamp
        else:
            # Reuse the last detection result rather than running YOLO again -
            # face analysis, filters, state evaluation, events, and session
            # recording below still run every frame regardless.
            detections = self._last_detections

        try:
            face_result = self._face.analyze(frame.image, frame.timestamp)
        except FaceAnalysisError as exc:
            print(f"Face analysis error: {exc}")
            event = self._event_manager.vision_error(frame.timestamp, str(exc))
            self._session_manager.record_event(event)
            face_result = FaceAnalysisResult(
                face_detected=False, eyes_state=EyeState.UNKNOWN, eye_metric=None, timestamp=frame.timestamp
            )

        head_result = None
        if face_result.face_detected and face_result.landmarks:
            head_result = estimate_head_pose(
                face_result.landmarks,
                frame.image.shape[1],
                frame.image.shape[0],
                self._config.head.yaw_threshold_degrees,
                self._config.head.pitch_threshold_degrees,
            )

        snapshot = build_perception_snapshot(
            frame.timestamp,
            detections,
            face_present=face_result.face_detected,
            eyes_state=face_result.eyes_state,
            eye_metric=face_result.eye_metric,
            head_orientation=head_result.orientation if head_result else HeadOrientation.UNKNOWN,
            head_yaw=head_result.yaw_degrees if head_result else None,
            head_pitch=head_result.pitch_degrees if head_result else None,
        )

        phone_result = self._phone_filter.update(snapshot.phone_detected, snapshot.timestamp)
        head_filter_result = self._head_filter.update(snapshot.head_orientation, snapshot.timestamp)
        drowsy_result = self._drowsiness_filter.update(snapshot.eyes_state, snapshot.timestamp)
        away_result = self._away_filter.update(snapshot.person_present, snapshot.timestamp)

        transition = self._state_manager.evaluate(
            is_away=away_result.is_away,
            is_phone_distraction=phone_result.is_confirmed,
            is_drowsy=drowsy_result.is_drowsy,
            is_diverted=head_filter_result.is_diverted,
            vision_quality=snapshot.vision_quality,
            timestamp=snapshot.timestamp,
        )
        self._session_manager.record_transition(transition)
        self._emit_and_route_events(
            phone_result, drowsy_result, head_filter_result, away_result, transition, snapshot.timestamp
        )

        self._last_snapshot = snapshot
        self._last_detections = detections
        self._last_face_result = face_result

    def _emit_and_route_events(
        self, phone_result, drowsy_result, head_result, away_result, transition, timestamp: float
    ) -> None:
        """Signal-level events come straight from each filter's
        just_confirmed/just_cleared edge; FOCUS_RESTORED is the one
        state-level event, fired whenever StateManager transitions INTO
        FOCUSED from an actual distraction state (PHONE_DISTRACTION,
        DROWSINESS_SIGNAL, ATTENTION_DIVERTED, AWAY) - not from IDLE or
        UNKNOWN. A session's very first evaluate() call always has
        previous_state=UNKNOWN (StateManager.start_session() lands there
        before any real signal has been read), so without excluding
        UNKNOWN too, simply sitting down focused at the start of a
        session would spuriously fire "focus restored" before any
        distraction ever happened - the PRD section 38 demo only expects
        this event after a genuine distraction clears (e.g. "put phone
        away")."""
        if phone_result.just_confirmed:
            event = self._event_manager.phone_confirmed(timestamp)
            if event is not None:  # None means the phone-warning cooldown suppressed it
                self._session_manager.record_event(event)
                self._audio.play_phone_warning(timestamp)
        if phone_result.just_cleared:
            self._session_manager.record_event(self._event_manager.phone_cleared(timestamp))
        # Repeats the phone warning every audio.persistent_warning_interval_seconds
        # while still confirmed, without spamming every frame - called
        # unconditionally (not gated on just_confirmed/just_cleared) since
        # it needs the continuous is_confirmed signal, not just the edges.
        self._audio.notify_phone_distraction(phone_result.is_confirmed, timestamp)

        if drowsy_result.just_confirmed:
            self._session_manager.record_event(self._event_manager.drowsiness_confirmed(timestamp))
            self._audio.play_drowsiness_warning()
        if drowsy_result.just_cleared:
            self._session_manager.record_event(self._event_manager.drowsiness_cleared(timestamp))
        self._audio.notify_drowsiness(drowsy_result.is_drowsy, timestamp)

        if head_result.just_diverted:
            self._session_manager.record_event(self._event_manager.attention_diverted(timestamp))
            self._audio.play_attention_warning()
        if head_result.just_restored:
            self._session_manager.record_event(self._event_manager.attention_restored(timestamp))
        self._audio.notify_attention_diverted(head_result.is_diverted, timestamp)

        if away_result.just_confirmed:
            self._session_manager.record_event(self._event_manager.person_left(timestamp))
        if away_result.just_cleared:
            self._session_manager.record_event(self._event_manager.person_returned(timestamp))

        restorable_states = (
            FocusState.PHONE_DISTRACTION,
            FocusState.DROWSINESS_SIGNAL,
            FocusState.ATTENTION_DIVERTED,
            FocusState.AWAY,
        )
        if transition.changed and transition.state == FocusState.FOCUSED and transition.previous_state in restorable_states:
            self._session_manager.record_event(self._event_manager.focus_restored(timestamp))
            self._audio.play_focus_restored()

    # --- Controls (PRD section 23) --------------------------------------------------

    def _handle_actions(self, actions: list[UIAction], now: float) -> None:
        for action in actions:
            if action == UIAction.EXIT:
                self._handle_exit(now)
                return
            elif action == UIAction.START_PAUSE_RESUME:
                self._handle_start_pause_resume(now)
            elif action == UIAction.TOGGLE_MUTE:
                self._audio.toggle_mute()
            elif action == UIAction.TOGGLE_DEBUG:
                self._debug = not self._debug
            elif action == UIAction.RESET:
                self._handle_reset(now)

    def _handle_start_pause_resume(self, now: float) -> None:
        if not self._session_manager.is_active:
            self._start_session(now)
        elif self._session_manager.is_paused:
            self._resume_session(now)
        else:
            self._pause_session(now)

    def _start_session(self, now: float) -> None:
        self._state_manager.start_session(now)
        self._session_manager.start_session(now)
        self._session_start_timestamp = now
        self._last_snapshot = None
        self._last_detections = []
        self._last_face_result = None
        self._last_yolo_timestamp = None
        self._audio.reset_persistent_reminders()
        self._session_manager.record_event(self._event_manager.session_started(now))
        self._audio.start_music()

    def _pause_session(self, now: float) -> None:
        self._session_manager.pause_session(now)
        self._audio.pause_music()
        # A paused session isn't being monitored - without this, a long
        # pause would otherwise look like the condition "continued" across
        # the whole gap once resumed, firing a spurious immediate reminder.
        self._audio.reset_persistent_reminders()

    def _resume_session(self, now: float) -> None:
        self._session_manager.resume_session(now)
        self._audio.resume_music()

    def _handle_exit(self, now: float) -> None:
        """If idle, exit directly; if a session is active or paused, end it
        (summary computed, shown, and saved) before exiting."""
        if self._session_manager.is_active:
            self._end_session(now)
        self._running = False

    def _end_session(self, now: float) -> None:
        self._session_manager.record_event(self._event_manager.session_ended(now))
        summary = self._session_manager.end_session(now)
        self._state_manager.end_session(now)
        self._audio.stop_music()
        self._audio.play_session_complete()
        self._audio.reset_persistent_reminders()
        self._session_start_timestamp = None

        try:
            path = SessionManager.save_summary_json(summary, directory=self._logs_directory)
            print(f"Session summary saved to {path}")
        except OSError as exc:
            print(f"Could not save session summary: {exc}")

        self._print_summary(summary)

    def _handle_reset(self, now: float) -> None:
        """R is only honored while paused or idle - never mid-active-
        session, so a single keystroke can never silently discard
        in-progress analytics (PRD section 23: "Reset session if safe")."""
        if self._session_manager.is_active and not self._session_manager.is_paused:
            return
        self._session_manager.reset()
        self._state_manager.end_session(now)
        self._audio.stop_music()
        self._audio.reset_persistent_reminders()
        self._session_start_timestamp = None
        self._last_snapshot = None
        self._last_detections = []
        self._last_face_result = None
        self._last_yolo_timestamp = None

    @staticmethod
    def _print_summary(summary: SessionSummary) -> None:
        """PRD section 27: display the analytics at session end. Printed to
        the console - this phase does not add a dedicated Pygame summary
        screen (a materially bigger UI feature the PRD does not mandate;
        it only requires the data be "displayed")."""
        print("=== Session Summary (Estimated) ===")
        print(f"Session Duration:      {summary.total_duration_seconds:.1f}s")
        print(f"Focused Duration:      {summary.focused_duration_seconds:.1f}s")
        print(f"Phone Distractions:    {summary.phone_distraction_count}")
        print(f"Drowsiness Signals:    {summary.drowsiness_count}")
        print(f"Attention Diversions:  {summary.attention_diversion_count}")
        print(f"Away Events:           {summary.away_count}")
        print(f"Longest Focus Streak:  {summary.longest_focus_streak_seconds:.1f}s")
        print(f"Estimated Focus Score: {summary.focus_score}")

    # --- Dashboard assembly -----------------------------------------------------------

    def _build_dashboard_view(self, frame: Frame) -> DashboardView:
        snapshot = self._last_snapshot
        debug_info: DebugInfo | None = None
        if self._debug:
            landmarks = self._last_face_result.landmarks if self._last_face_result else None
            debug_info = DebugInfo(
                detections=tuple(self._last_detections),
                landmarks=tuple(landmarks) if landmarks else None,
                eye_metric=snapshot.eye_metric if snapshot else None,
                head_yaw=snapshot.head_yaw if snapshot else None,
                head_pitch=snapshot.head_pitch if snapshot else None,
                vision_quality=snapshot.vision_quality if snapshot else None,
                phone_timer_seconds=self._phone_filter.elapsed_in_state_seconds(frame.timestamp),
                drowsiness_timer_seconds=self._drowsiness_filter.elapsed_in_state_seconds(frame.timestamp),
                attention_timer_seconds=self._head_filter.elapsed_in_state_seconds(frame.timestamp),
                away_timer_seconds=self._away_filter.elapsed_in_state_seconds(frame.timestamp),
            )

        # PRD section 22 has a single INFERENCE slot; YOLO + face-analysis
        # latency are summed into it (the total per-frame CV cost).
        inference_ms = (self._yolo.last_inference_ms or 0.0) + (self._face.last_inference_ms or 0.0)

        return DashboardView(
            status=self._state_manager.state,
            person_present=snapshot.person_present if snapshot else False,
            phone_detected=snapshot.phone_detected if snapshot else False,
            eyes_state=snapshot.eyes_state if snapshot else EyeState.UNKNOWN,
            head_orientation=snapshot.head_orientation if snapshot else HeadOrientation.UNKNOWN,
            session_elapsed_seconds=self._session_manager.elapsed_seconds(frame.timestamp),
            focus_score=self._session_manager.focus_score,
            fps=self._fps or 0.0,
            inference_latency_ms=inference_ms,
            recent_events=tuple(self._event_manager.events),
            session_start_timestamp=self._session_start_timestamp,
            debug=self._debug,
            debug_info=debug_info,
            paused=self._session_manager.is_paused,
        )

    def _track_fps(self, now: float) -> None:
        """Exponential moving average - smooths frame-to-frame jitter
        without needing a stored window of past timestamps."""
        if self._last_frame_time is not None:
            delta = now - self._last_frame_time
            if delta > 0:
                instantaneous = 1.0 / delta
                self._fps = instantaneous if self._fps is None else (0.9 * self._fps + 0.1 * instantaneous)
        self._last_frame_time = now
