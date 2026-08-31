"""Tests for CameraManager (FOCUSGUARD_PRD.md section 6).

All tests use a fake VideoCapture double injected via capture_factory, so no
physical webcam is required.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.camera.camera_manager import CameraError, CameraManager
from src.core.config_manager import CameraConfig


def make_config(index: int = 0, width: int = 1280, height: int = 720, target_fps: int = 30) -> CameraConfig:
    return CameraConfig(index=index, width=width, height=height, target_fps=target_fps)


def make_image(width: int = 4, height: int = 3) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


class FakeVideoCapture:
    """Test double standing in for cv2.VideoCapture."""

    def __init__(self, opened: bool = True, read_results: list | None = None) -> None:
        self._opened = opened
        self._read_results = list(read_results) if read_results is not None else [(True, make_image())]
        self.set_calls: list[tuple[int, float]] = []
        self.release_calls = 0

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop_id: int, value: float) -> bool:
        self.set_calls.append((prop_id, value))
        return True

    def read(self):
        if not self._read_results:
            return False, None
        return self._read_results.pop(0)

    def release(self) -> None:
        self.release_calls += 1
        self._opened = False


def factory_for(fake: FakeVideoCapture, received_indexes: list[int] | None = None):
    def _factory(index: int) -> FakeVideoCapture:
        if received_indexes is not None:
            received_indexes.append(index)
        return fake

    return _factory


# --- open() ---------------------------------------------------------------


def test_open_succeeds_and_configures_resolution() -> None:
    fake = FakeVideoCapture(opened=True)
    manager = CameraManager(make_config(width=1280, height=720), capture_factory=factory_for(fake))

    manager.open()

    assert manager.is_open is True
    assert (cv2.CAP_PROP_FRAME_WIDTH, 1280.0) in fake.set_calls
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 720.0) in fake.set_calls


def test_open_passes_configured_camera_index_to_capture_factory() -> None:
    fake = FakeVideoCapture(opened=True)
    received_indexes: list[int] = []
    config = make_config(index=7)
    manager = CameraManager(config, capture_factory=factory_for(fake, received_indexes))

    manager.open()

    assert received_indexes == [config.index]


def test_open_raises_camera_error_when_device_not_opened() -> None:
    fake = FakeVideoCapture(opened=False)
    manager = CameraManager(make_config(index=5), capture_factory=factory_for(fake))

    with pytest.raises(CameraError, match="Could not open camera"):
        manager.open()

    assert manager.is_open is False
    assert fake.release_calls == 1


def test_open_wraps_factory_exception_in_camera_error() -> None:
    def failing_factory(index: int):
        raise OSError("no such device")

    manager = CameraManager(make_config(), capture_factory=failing_factory)

    with pytest.raises(CameraError, match="Failed to open camera"):
        manager.open()


def test_open_is_idempotent() -> None:
    fake = FakeVideoCapture(opened=True)
    factory_calls = []

    def factory(index: int) -> FakeVideoCapture:
        factory_calls.append(index)
        return fake

    manager = CameraManager(make_config(), capture_factory=factory)

    manager.open()
    manager.open()

    assert len(factory_calls) == 1


# --- read_frame() -----------------------------------------------------------


def test_read_frame_before_open_raises_camera_error() -> None:
    manager = CameraManager(make_config(), capture_factory=factory_for(FakeVideoCapture()))

    with pytest.raises(CameraError, match="not open"):
        manager.read_frame()


def test_read_frame_returns_frame_with_incrementing_index_and_timestamp() -> None:
    image_a = make_image()
    image_b = make_image()
    fake = FakeVideoCapture(read_results=[(True, image_a), (True, image_b)])
    manager = CameraManager(make_config(), capture_factory=factory_for(fake))
    manager.open()

    frame1 = manager.read_frame()
    frame2 = manager.read_frame()

    assert frame1.index == 1
    assert frame2.index == 2
    assert frame1.image is image_a
    assert frame2.image is image_b
    assert frame2.timestamp >= frame1.timestamp


def test_read_frame_raises_camera_error_when_read_fails() -> None:
    fake = FakeVideoCapture(read_results=[(False, None)])
    manager = CameraManager(make_config(), capture_factory=factory_for(fake))
    manager.open()

    with pytest.raises(CameraError, match="Failed to read a valid frame"):
        manager.read_frame()


def test_read_frame_raises_camera_error_when_image_is_none() -> None:
    fake = FakeVideoCapture(read_results=[(True, None)])
    manager = CameraManager(make_config(), capture_factory=factory_for(fake))
    manager.open()

    with pytest.raises(CameraError, match="Failed to read a valid frame"):
        manager.read_frame()


def test_read_frame_raises_camera_error_when_image_is_empty() -> None:
    empty_image = np.zeros((0, 0, 3), dtype=np.uint8)
    fake = FakeVideoCapture(read_results=[(True, empty_image)])
    manager = CameraManager(make_config(), capture_factory=factory_for(fake))
    manager.open()

    with pytest.raises(CameraError, match="Failed to read a valid frame"):
        manager.read_frame()


def test_read_frame_raises_camera_error_when_read_throws() -> None:
    class ThrowingCapture(FakeVideoCapture):
        def read(self):
            raise RuntimeError("device disconnected")

    fake = ThrowingCapture()
    manager = CameraManager(make_config(), capture_factory=factory_for(fake))
    manager.open()

    with pytest.raises(CameraError, match="disconnected"):
        manager.read_frame()


# --- release() ---------------------------------------------------------------


def test_release_releases_capture_and_resets_state() -> None:
    fake = FakeVideoCapture()
    manager = CameraManager(make_config(), capture_factory=factory_for(fake))
    manager.open()

    manager.release()

    assert manager.is_open is False
    assert fake.release_calls == 1


def test_release_is_idempotent_and_safe_without_open() -> None:
    manager = CameraManager(make_config(), capture_factory=factory_for(FakeVideoCapture()))

    manager.release()  # never opened
    manager.release()  # already released

    assert manager.is_open is False


def test_release_then_read_frame_raises_camera_error() -> None:
    fake = FakeVideoCapture()
    manager = CameraManager(make_config(), capture_factory=factory_for(fake))
    manager.open()
    manager.release()

    with pytest.raises(CameraError, match="not open"):
        manager.read_frame()


# --- context manager ---------------------------------------------------------


def test_context_manager_opens_and_releases() -> None:
    fake = FakeVideoCapture()

    with CameraManager(make_config(), capture_factory=factory_for(fake)) as manager:
        assert manager.is_open is True

    assert manager.is_open is False
    assert fake.release_calls == 1


def test_context_manager_releases_on_exception() -> None:
    fake = FakeVideoCapture()

    with pytest.raises(ValueError):
        with CameraManager(make_config(), capture_factory=factory_for(fake)) as manager:
            raise ValueError("boom")

    assert manager.is_open is False
    assert fake.release_calls == 1
