"""Pygame dashboard rendering and input handling (PRD sections 22-24).

UIManager owns only presentation/input (PRD section 33): it turns a
DashboardView (src/ui/dashboard_view.py) plus an optional current camera
frame into pixels on screen, and turns raw Pygame input events into
UIAction values. It never computes FPS, focus score, or session duration,
and never decides what to do with a UIAction (starting/pausing a session,
muting audio, resetting) - those decisions belong to whatever assembles
the DashboardView and consumes poll_input()'s result (Phase 12
integration, not this phase). It never runs YOLO/face-analysis logic and
never reads the webcam itself - "do not put CV logic inside the
rendering code" (PRD section 22).

Window size is a hardcoded, sensible default (no config change this
phase - PRD does not ask for configurable window geometry).

Controls (PRD section 23): SPACE start/pause/resume, Q/ESC exit, M toggle
mute, D toggle debug, R reset. The window's own close button is also
mapped to EXIT - standard Pygame-app behavior, not a PRD conflict.
"""

from __future__ import annotations

import numpy as np
import pygame

from src.ui.dashboard_view import (
    MAX_EVENT_LOG_LINES,
    DashboardView,
    UIAction,
    format_confidence,
    format_duration,
    format_event_timestamp,
    format_event_type,
    format_eye_state,
    format_head_orientation,
    format_presence,
    format_status,
    format_timer,
    format_vision_quality,
    recent_events,
)

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 760

CAMERA_PANEL_X = 10
CAMERA_PANEL_Y = 40
CAMERA_PANEL_WIDTH = 680
CAMERA_PANEL_HEIGHT = 470

SIDEBAR_X = CAMERA_PANEL_X + CAMERA_PANEL_WIDTH + 20
SIDEBAR_WIDTH = WINDOW_WIDTH - SIDEBAR_X - 10

EVENT_LOG_Y = CAMERA_PANEL_Y + CAMERA_PANEL_HEIGHT + 15
EVENT_LOG_HEIGHT = WINDOW_HEIGHT - EVENT_LOG_Y - 10
EVENT_LOG_HEADER_OFFSET = 26
EVENT_LOG_LINE_HEIGHT = 18

BACKGROUND_COLOR = (18, 18, 22)
PANEL_COLOR = (30, 30, 36)
TITLE_COLOR = (240, 240, 245)
TEXT_COLOR = (225, 225, 230)
LABEL_COLOR = (145, 145, 155)
ACCENT_COLOR = (90, 170, 255)
WARNING_COLOR = (240, 170, 60)
DEBUG_BOX_COLOR = (80, 220, 120)
DEBUG_LANDMARK_COLOR = (255, 210, 90)
DEBUG_TEXT_COLOR = (150, 220, 150)

_KEY_ACTIONS: dict[int, UIAction] = {
    pygame.K_SPACE: UIAction.START_PAUSE_RESUME,
    pygame.K_q: UIAction.EXIT,
    pygame.K_ESCAPE: UIAction.EXIT,
    pygame.K_m: UIAction.TOGGLE_MUTE,
    pygame.K_d: UIAction.TOGGLE_DEBUG,
    pygame.K_r: UIAction.RESET,
}


class UIError(Exception):
    """Raised for Pygame initialization failures with a human-readable message."""


class UIManager:
    """Renders the FocusGuard dashboard and translates input into UIActions.

    All Pygame calls are isolated to this class so the rest of the
    application never imports pygame directly (PRD section 33).
    """

    def __init__(self, window_size: tuple[int, int] = (WINDOW_WIDTH, WINDOW_HEIGHT)) -> None:
        self._window_size = window_size
        self._screen: pygame.Surface | None = None
        self._font: pygame.font.Font | None = None
        self._label_font: pygame.font.Font | None = None
        self._title_font: pygame.font.Font | None = None
        self._debug_font: pygame.font.Font | None = None
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def init(self) -> None:
        """Initialize Pygame's display/font subsystems. No-op if already
        initialized. Raises UIError (never lets a raw Pygame exception
        escape) on failure, per PRD section 35."""
        if self._is_initialized:
            return
        try:
            pygame.init()
            pygame.display.set_caption("FocusGuard")
            self._screen = pygame.display.set_mode(self._window_size)
            self._font = pygame.font.SysFont(None, 22)
            self._label_font = pygame.font.SysFont(None, 18)
            self._title_font = pygame.font.SysFont(None, 30)
            self._debug_font = pygame.font.SysFont(None, 16)
        except Exception as exc:
            raise UIError(f"Failed to initialize Pygame UI: {exc}") from exc
        self._is_initialized = True

    def shutdown(self) -> None:
        """Release Pygame resources. Safe to call multiple times."""
        if self._is_initialized:
            pygame.quit()
        self._screen = None
        self._font = None
        self._label_font = None
        self._title_font = None
        self._debug_font = None
        self._is_initialized = False

    def poll_input(self) -> list[UIAction]:
        """Drain the Pygame event queue and return the UIActions implied
        by this frame's input (possibly empty, possibly more than one)."""
        self._require_initialized("poll_input")

        actions: list[UIAction] = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actions.append(UIAction.EXIT)
            elif event.type == pygame.KEYDOWN:
                action = _KEY_ACTIONS.get(event.key)
                if action is not None:
                    actions.append(action)
        return actions

    def render(self, view: DashboardView, frame: np.ndarray | None = None) -> None:
        """Draw one full dashboard frame and flip the display."""
        self._require_initialized("render")
        assert self._screen is not None  # narrowed by _require_initialized

        self._screen.fill(BACKGROUND_COLOR)
        self._blit_text(self._title_font, "FOCUSGUARD", (CAMERA_PANEL_X, 5), TITLE_COLOR)
        self._render_camera_panel(frame)
        self._render_dashboard(view)
        self._render_event_log(view)
        if view.debug and view.debug_info is not None:
            self._render_debug_overlay(view, frame)
        pygame.display.flip()

    # --- Internal rendering helpers -----------------------------------------

    def _require_initialized(self, method_name: str) -> None:
        if not self._is_initialized or self._screen is None:
            raise UIError(f"UIManager is not initialized. Call init() before {method_name}().")

    def _blit_text(
        self,
        font: pygame.font.Font | None,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        assert self._screen is not None
        assert font is not None
        surface = font.render(text, True, color)
        self._screen.blit(surface, position)

    @staticmethod
    def _camera_fit(frame_width: int, frame_height: int) -> tuple[float, float, float]:
        """Aspect-ratio-preserving fit of a frame into the camera panel.

        Returns (scale, offset_x, offset_y): a source pixel (px, py) maps
        to panel-relative (offset_x + px*scale, offset_y + py*scale).
        Stretching the frame to fill the panel unconditionally (ignoring
        its aspect ratio) would visibly distort a real webcam's default
        16:9 feed against this panel's ~16:11 shape - letterboxing instead
        keeps circles circular and faces the right proportions.
        Detection boxes and landmarks use this same fit (see
        _render_debug_boxes) so they stay aligned with the letterboxed
        video rather than the old, unscaled panel bounds.
        """
        if frame_width <= 0 or frame_height <= 0:
            return 1.0, 0.0, 0.0
        scale = min(CAMERA_PANEL_WIDTH / frame_width, CAMERA_PANEL_HEIGHT / frame_height)
        offset_x = (CAMERA_PANEL_WIDTH - frame_width * scale) / 2.0
        offset_y = (CAMERA_PANEL_HEIGHT - frame_height * scale) / 2.0
        return scale, offset_x, offset_y

    def _render_camera_panel(self, frame: np.ndarray | None) -> None:
        assert self._screen is not None
        panel_rect = pygame.Rect(CAMERA_PANEL_X, CAMERA_PANEL_Y, CAMERA_PANEL_WIDTH, CAMERA_PANEL_HEIGHT)
        pygame.draw.rect(self._screen, PANEL_COLOR, panel_rect)

        if frame is None:
            self._blit_text(self._label_font, "NO CAMERA FEED", (panel_rect.x + 12, panel_rect.y + 12), LABEL_COLOR)
            return

        frame_height, frame_width = frame.shape[0], frame.shape[1]
        scale, offset_x, offset_y = self._camera_fit(frame_width, frame_height)
        scaled_size = (max(1, int(frame_width * scale)), max(1, int(frame_height * scale)))

        surface = self.frame_to_surface(frame)
        scaled = pygame.transform.smoothscale(surface, scaled_size)
        self._screen.blit(scaled, (panel_rect.x + offset_x, panel_rect.y + offset_y))

    def _render_dashboard(self, view: DashboardView) -> None:
        # FocusState has no PAUSED value (it simply stops being
        # re-evaluated while paused) - view.paused is what tells the UI to
        # show "PAUSED" instead of the frozen, now-stale status label.
        status_text = "PAUSED" if view.paused else format_status(view.status)
        status_color = WARNING_COLOR if view.paused else ACCENT_COLOR

        rows: list[tuple[str, str, tuple[int, int, int]]] = [
            ("STATUS", status_text, status_color),
            ("PERSON", format_presence(view.person_present), TEXT_COLOR),
            ("PHONE", format_presence(view.phone_detected), WARNING_COLOR if view.phone_detected else TEXT_COLOR),
            ("EYES", format_eye_state(view.eyes_state), TEXT_COLOR),
            ("HEAD", format_head_orientation(view.head_orientation), TEXT_COLOR),
            ("SESSION", format_duration(view.session_elapsed_seconds), TEXT_COLOR),
            ("FOCUS SCORE", str(view.focus_score), TEXT_COLOR),
            ("FPS", f"{view.fps:.0f}", TEXT_COLOR),
            ("INFERENCE", f"{view.inference_latency_ms:.0f} ms", TEXT_COLOR),
        ]

        y = CAMERA_PANEL_Y
        for label, value, color in rows:
            self._blit_text(self._label_font, label, (SIDEBAR_X, y), LABEL_COLOR)
            self._blit_text(self._font, value, (SIDEBAR_X, y + 16), color)
            y += 42

    def _render_event_log(self, view: DashboardView) -> None:
        assert self._screen is not None
        log_rect = pygame.Rect(CAMERA_PANEL_X, EVENT_LOG_Y, WINDOW_WIDTH - 2 * CAMERA_PANEL_X, EVENT_LOG_HEIGHT)
        pygame.draw.rect(self._screen, PANEL_COLOR, log_rect)
        self._blit_text(self._label_font, "EVENT LOG", (log_rect.x + 10, log_rect.y + 6), LABEL_COLOR)

        # PRD: show the latest MAX_EVENT_LOG_LINES events, but never more
        # than the panel can actually fit vertically - a fixed line count
        # blindly assumed to fit is what clipped the last entry off the
        # bottom of the panel during visual verification.
        available_height = log_rect.height - EVENT_LOG_HEADER_OFFSET
        lines_that_fit = max(0, available_height // EVENT_LOG_LINE_HEIGHT)
        limit = min(MAX_EVENT_LOG_LINES, lines_that_fit)

        y = log_rect.y + EVENT_LOG_HEADER_OFFSET
        for event in recent_events(view.recent_events, limit=limit):
            line = f"{format_event_timestamp(event.timestamp, view.session_start_timestamp)}  {format_event_type(event)}"
            self._blit_text(self._debug_font, line, (log_rect.x + 10, y), TEXT_COLOR)
            y += EVENT_LOG_LINE_HEIGHT

    def _render_debug_overlay(self, view: DashboardView, frame: np.ndarray | None) -> None:
        debug_info = view.debug_info
        assert debug_info is not None

        if frame is not None:
            self._render_debug_boxes(debug_info, frame)

        lines = [
            f"eye_metric: {debug_info.eye_metric:.3f}" if debug_info.eye_metric is not None else "eye_metric: --",
            f"head_yaw: {debug_info.head_yaw:.1f}deg" if debug_info.head_yaw is not None else "head_yaw: --",
            f"head_pitch: {debug_info.head_pitch:.1f}deg" if debug_info.head_pitch is not None else "head_pitch: --",
            (
                f"vision_quality: {format_vision_quality(debug_info.vision_quality)}"
                if debug_info.vision_quality is not None
                else "vision_quality: --"
            ),
            f"phone_timer: {format_timer(debug_info.phone_timer_seconds)}",
            f"drowsiness_timer: {format_timer(debug_info.drowsiness_timer_seconds)}",
            f"attention_timer: {format_timer(debug_info.attention_timer_seconds)}",
            f"away_timer: {format_timer(debug_info.away_timer_seconds)}",
        ]

        line_height = 16
        block_height = line_height * len(lines) + 8
        block_width = 230
        block_y = CAMERA_PANEL_Y + CAMERA_PANEL_HEIGHT - block_height

        # A semi-transparent backing keeps this text legible over
        # unpredictable camera-feed content (e.g. a bright region) instead
        # of blitting text directly on top of the video with no contrast
        # guarantee.
        backing = pygame.Surface((block_width, block_height), pygame.SRCALPHA)
        backing.fill((10, 10, 12, 190))
        assert self._screen is not None
        self._screen.blit(backing, (CAMERA_PANEL_X, block_y))

        y = block_y + 4
        for line in lines:
            self._blit_text(self._debug_font, line, (CAMERA_PANEL_X + 8, y), DEBUG_TEXT_COLOR)
            y += line_height

    def _blit_label_with_backing(
        self,
        text: str,
        position: tuple[float, float],
        text_color: tuple[int, int, int],
    ) -> None:
        """Draw a small text label with an opaque backing rectangle sized
        to fit it - unlike the main dashboard text (already on a solid
        PANEL_COLOR background), labels drawn directly over unpredictable
        camera-feed content have no guaranteed contrast otherwise (e.g. a
        bright frame region can make plain colored text unreadable)."""
        assert self._screen is not None
        assert self._debug_font is not None
        surface = self._debug_font.render(text, True, text_color)
        backing = pygame.Surface((surface.get_width() + 4, surface.get_height() + 2), pygame.SRCALPHA)
        backing.fill((10, 10, 12, 190))
        self._screen.blit(backing, (position[0] - 2, position[1] - 1))
        self._screen.blit(surface, position)

    def _render_debug_boxes(self, debug_info, frame: np.ndarray) -> None:  # noqa: ANN001
        assert self._screen is not None
        frame_height, frame_width = frame.shape[0], frame.shape[1]
        if frame_width <= 0 or frame_height <= 0:
            return
        scale, offset_x, offset_y = self._camera_fit(frame_width, frame_height)
        base_x = CAMERA_PANEL_X + offset_x
        base_y = CAMERA_PANEL_Y + offset_y

        for detection in debug_info.detections:
            x1 = base_x + detection.x1 * scale
            y1 = base_y + detection.y1 * scale
            x2 = base_x + detection.x2 * scale
            y2 = base_y + detection.y2 * scale
            rect = pygame.Rect(x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))
            pygame.draw.rect(self._screen, DEBUG_BOX_COLOR, rect, width=2)
            label = f"{detection.class_name} {format_confidence(detection.confidence)}"
            self._blit_label_with_backing(label, (rect.x, max(0, rect.y - 15)), DEBUG_BOX_COLOR)

        if debug_info.landmarks:
            for point in debug_info.landmarks:
                x = base_x + point.x * frame_width * scale
                y = base_y + point.y * frame_height * scale
                pygame.draw.circle(self._screen, DEBUG_LANDMARK_COLOR, (int(x), int(y)), 1)

    @staticmethod
    def frame_to_surface(frame: np.ndarray) -> pygame.Surface:
        """Convert a BGR (H, W, 3) OpenCV-style frame into a Pygame Surface.

        pygame.surfarray expects a (W, H, 3) RGB array, so this swaps the
        color channels (BGR -> RGB) and transposes the first two axes.
        """
        rgb = frame[:, :, ::-1]
        return pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))

    def __enter__(self) -> "UIManager":
        self.init()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
