"""Webcam capture and lifecycle management (PRD section 6).

CameraManager owns only camera operations: opening the device, configuring
resolution, reading frames, detecting invalid frames, and releasing the
device. It never performs detection, analysis, or UI work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import cv2

from src.core.config_manager import CameraConfig


class CameraError(Exception):
    """Raised for camera open/read/release failures with a human-readable message."""


class VideoCapture(Protocol):
    """The subset of cv2.VideoCapture's interface CameraManager depends on."""

    def isOpened(self) -> bool: ...

    def set(self, prop_id: int, value: float) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[int], VideoCapture]


@dataclass(frozen=True)
class Frame:
    """A single captured frame with a monotonic capture timestamp."""

    image: np.ndarray
    timestamp: float
    index: int


class CameraManager:
    """Opens a webcam, reads frames, and releases the device.

    The capture backend is injectable via ``capture_factory`` so this class
    is fully unit-testable without a physical webcam.
    """

    def __init__(
        self,
        config: CameraConfig,
        capture_factory: CaptureFactory = cv2.VideoCapture,
    ) -> None:
        self._config = config
        self._capture_factory = capture_factory
        self._capture: VideoCapture | None = None
        self._is_open = False
        self._frame_count = 0

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        """Open the configured camera device. No-op if already open."""
        if self._is_open:
            return

        try:
            capture = self._capture_factory(self._config.index)
        except Exception as exc:
            raise CameraError(
                f"Failed to open camera at index {self._config.index}: {exc}"
            ) from exc

        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            raise CameraError(
                f"Could not open camera at index {self._config.index}. "
                "Check that a webcam is connected, not already in use by "
                "another application, and that camera permissions are "
                "granted to this application."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)

        self._capture = capture
        self._is_open = True
        self._frame_count = 0

    def read_frame(self) -> Frame:
        """Read the next frame. Raises CameraError if unavailable or invalid."""
        if not self._is_open or self._capture is None:
            raise CameraError("Camera is not open. Call open() before read_frame().")

        try:
            ok, image = self._capture.read()
        except Exception as exc:
            raise CameraError(
                "Failed to read frame from camera. The camera may have been "
                f"disconnected: {exc}"
            ) from exc

        if not ok or image is None or image.size == 0:
            raise CameraError(
                "Failed to read a valid frame from the camera. It may have "
                "been disconnected or is no longer available."
            )

        self._frame_count += 1
        return Frame(image=image, timestamp=time.monotonic(), index=self._frame_count)

    def release(self) -> None:
        """Release the camera device. Safe to call multiple times."""
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._is_open = False

    def __enter__(self) -> "CameraManager":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
