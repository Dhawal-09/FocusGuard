"""Tests for UIManager (FOCUSGUARD_PRD.md sections 22-24, 33).

Runs headless via SDL's dummy video driver (SDL_VIDEODRIVER=dummy), the
standard approach for testing Pygame code without a real display or GPU.
No webcam, YOLO, or MediaPipe model - synthetic frames/view-models only.
A separate real (non-headless) visual smoke test is performed manually
outside this suite; see the Phase 8 implementation report.
"""

from __future__ import annotations

import os
from collections import namedtuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from src.core.types import VisionQuality
from src.detection.detection_types import Detection
from src.events.event_manager import Event, EventType, Severity
from src.face.eye_metrics import EyeState
from src.face.head_pose import HeadOrientation
from src.state.state_manager import FocusState
from src.ui.dashboard_view import DashboardView, DebugInfo, UIAction
from src.ui.ui_manager import UIError, UIManager

_Landmark = namedtuple("_Landmark", ["x", "y"])


@pytest.fixture
def manager():
    ui = UIManager()
    ui.init()
    yield ui
    ui.shutdown()


def make_view(**overrides) -> DashboardView:
    defaults = dict(
        status=FocusState.FOCUSED,
        person_present=True,
        phone_detected=False,
        eyes_state=EyeState.OPEN,
        head_orientation=HeadOrientation.CENTER,
        session_elapsed_seconds=125.0,
        focus_score=91,
        fps=28.0,
        inference_latency_ms=32.0,
    )
    defaults.update(overrides)
    return DashboardView(**defaults)


def make_frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


# --- Lifecycle -------------------------------------------------------------------


def test_not_initialized_by_default() -> None:
    ui = UIManager()
    assert ui.is_initialized is False


def test_init_sets_initialized() -> None:
    ui = UIManager()
    ui.init()
    try:
        assert ui.is_initialized is True
    finally:
        ui.shutdown()


def test_init_is_idempotent() -> None:
    ui = UIManager()
    ui.init()
    try:
        ui.init()
        assert ui.is_initialized is True
    finally:
        ui.shutdown()


def test_shutdown_is_safe_to_call_multiple_times() -> None:
    ui = UIManager()
    ui.init()
    ui.shutdown()
    ui.shutdown()
    assert ui.is_initialized is False


def test_shutdown_without_init_does_not_raise() -> None:
    ui = UIManager()
    ui.shutdown()
    assert ui.is_initialized is False


def test_render_before_init_raises_uierror() -> None:
    ui = UIManager()
    with pytest.raises(UIError):
        ui.render(make_view())


def test_poll_input_before_init_raises_uierror() -> None:
    ui = UIManager()
    with pytest.raises(UIError):
        ui.poll_input()


def test_context_manager_initializes_and_shuts_down() -> None:
    with UIManager() as ui:
        assert ui.is_initialized is True
    assert ui.is_initialized is False


# --- Rendering: headless smoke (must not raise) -----------------------------------


def test_render_with_no_frame_does_not_raise(manager: UIManager) -> None:
    manager.render(make_view(), frame=None)


def test_render_with_frame_does_not_raise(manager: UIManager) -> None:
    manager.render(make_view(), frame=make_frame())


def test_render_with_debug_and_no_frame_does_not_raise(manager: UIManager) -> None:
    view = make_view(debug=True, debug_info=DebugInfo())
    manager.render(view, frame=None)


def test_render_with_debug_and_no_debug_info_does_not_raise(manager: UIManager) -> None:
    """debug=True but debug_info=None (caller forgot to populate it) must
    not crash - the overlay is simply skipped."""
    view = make_view(debug=True, debug_info=None)
    manager.render(view, frame=make_frame())


def test_render_with_debug_detections_does_not_raise(manager: UIManager) -> None:
    detections = (
        Detection(class_name="person", confidence=0.91, x1=10, y1=10, x2=200, y2=400, timestamp=1.0),
        Detection(class_name="cell phone", confidence=0.62, x1=220, y1=100, x2=260, y2=160, timestamp=1.0),
    )
    debug_info = DebugInfo(
        detections=detections,
        eye_metric=0.19,
        head_yaw=-12.5,
        head_pitch=3.0,
        vision_quality=VisionQuality.GOOD,
        phone_timer_seconds=0.12,
    )
    view = make_view(debug=True, debug_info=debug_info)
    manager.render(view, frame=make_frame())


def test_render_with_debug_landmarks_does_not_raise(manager: UIManager) -> None:
    landmarks = tuple(_Landmark(x=i / 20.0, y=(20 - i) / 20.0) for i in range(20))
    debug_info = DebugInfo(landmarks=landmarks)
    view = make_view(debug=True, debug_info=debug_info)
    manager.render(view, frame=make_frame())


def test_render_with_all_debug_timers_populated_does_not_raise(manager: UIManager) -> None:
    debug_info = DebugInfo(
        phone_timer_seconds=0.1,
        drowsiness_timer_seconds=0.5,
        attention_timer_seconds=0.3,
        away_timer_seconds=1.2,
    )
    view = make_view(debug=True, debug_info=debug_info)
    manager.render(view, frame=make_frame())


def test_render_with_no_detections_and_debug_on_does_not_raise(manager: UIManager) -> None:
    view = make_view(debug=True, debug_info=DebugInfo(detections=()))
    manager.render(view, frame=make_frame())


def test_render_with_empty_event_log_does_not_raise(manager: UIManager) -> None:
    manager.render(make_view(recent_events=()), frame=make_frame())


def test_render_with_more_than_eight_events_does_not_raise(manager: UIManager) -> None:
    events = tuple(
        Event(event_type=EventType.SESSION_STARTED, timestamp=float(i), severity=Severity.INFO)
        for i in range(20)
    )
    view = make_view(recent_events=events, session_start_timestamp=0.0)
    manager.render(view, frame=make_frame())


def test_render_with_no_session_start_timestamp_does_not_raise(manager: UIManager) -> None:
    events = (Event(event_type=EventType.SESSION_STARTED, timestamp=5.0, severity=Severity.INFO),)
    view = make_view(recent_events=events, session_start_timestamp=None)
    manager.render(view, frame=make_frame())


def test_render_with_various_frame_sizes_does_not_raise(manager: UIManager) -> None:
    for width, height in [(1280, 720), (640, 480), (320, 240), (1, 1)]:
        manager.render(make_view(), frame=make_frame(width, height))


def test_render_full_cycle_of_focus_states_does_not_raise(manager: UIManager) -> None:
    for state in FocusState:
        manager.render(make_view(status=state), frame=make_frame())


def test_render_multiple_consecutive_frames_does_not_raise(manager: UIManager) -> None:
    for i in range(5):
        manager.render(make_view(session_elapsed_seconds=float(i)), frame=make_frame())


def test_render_phone_detected_true_does_not_raise(manager: UIManager) -> None:
    manager.render(make_view(phone_detected=True), frame=make_frame())


# --- frame_to_surface --------------------------------------------------------------


def test_frame_to_surface_returns_surface_with_swapped_dimensions() -> None:
    frame = make_frame(width=64, height=48)
    surface = UIManager.frame_to_surface(frame)
    assert surface.get_size() == (64, 48)


def test_frame_to_surface_swaps_bgr_to_rgb() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)  # BGR order
    surface = UIManager.frame_to_surface(frame)
    pixel = surface.get_at((0, 0))
    assert (pixel.r, pixel.g, pixel.b) == (30, 20, 10)


def test_frame_to_surface_preserves_distinct_pixels() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)  # B
    frame[1, 1] = (0, 255, 0)  # G
    surface = UIManager.frame_to_surface(frame)
    top_left = surface.get_at((0, 0))
    bottom_right = surface.get_at((1, 1))
    assert (top_left.r, top_left.g, top_left.b) == (0, 0, 255)
    assert (bottom_right.r, bottom_right.g, bottom_right.b) == (0, 255, 0)


# --- Input mapping (PRD section 23) --------------------------------------------------


def _post_keydown(key: int) -> None:
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))


@pytest.mark.parametrize(
    "key,expected",
    [
        (pygame.K_SPACE, UIAction.START_PAUSE_RESUME),
        (pygame.K_q, UIAction.EXIT),
        (pygame.K_ESCAPE, UIAction.EXIT),
        (pygame.K_m, UIAction.TOGGLE_MUTE),
        (pygame.K_d, UIAction.TOGGLE_DEBUG),
        (pygame.K_r, UIAction.RESET),
    ],
)
def test_each_control_key_maps_to_expected_action(manager: UIManager, key: int, expected: UIAction) -> None:
    pygame.event.clear()
    _post_keydown(key)

    actions = manager.poll_input()

    assert actions == [expected]


def test_quit_event_maps_to_exit(manager: UIManager) -> None:
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.QUIT))

    actions = manager.poll_input()

    assert actions == [UIAction.EXIT]


def test_unmapped_key_produces_no_action(manager: UIManager) -> None:
    pygame.event.clear()
    _post_keydown(pygame.K_z)

    actions = manager.poll_input()

    assert actions == []


def test_no_events_returns_empty_list(manager: UIManager) -> None:
    pygame.event.clear()

    actions = manager.poll_input()

    assert actions == []


def test_multiple_queued_events_all_returned_in_order(manager: UIManager) -> None:
    pygame.event.clear()
    _post_keydown(pygame.K_SPACE)
    _post_keydown(pygame.K_d)

    actions = manager.poll_input()

    assert actions == [UIAction.START_PAUSE_RESUME, UIAction.TOGGLE_DEBUG]


def test_keyup_event_produces_no_action(manager: UIManager) -> None:
    """Only KEYDOWN should map to an action - a KEYUP for the same key
    must not double-fire."""
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE))

    actions = manager.poll_input()

    assert actions == []
