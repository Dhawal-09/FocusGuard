"""Configuration loading and validation for FocusGuard (PRD section 30)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

VALID_YOLO_DEVICES = {"auto", "cpu", "cuda"}


class ConfigError(Exception):
    """Raised when configuration is missing, malformed, or invalid."""


@dataclass(frozen=True)
class CameraConfig:
    index: int
    width: int
    height: int
    target_fps: int


@dataclass(frozen=True)
class YoloConfig:
    model: str
    confidence: float
    phone_confidence: float
    device: str


@dataclass(frozen=True)
class PhoneConfig:
    confirm_duration_seconds: float
    clear_duration_seconds: float
    warning_cooldown_seconds: float


@dataclass(frozen=True)
class EyesConfig:
    closed_threshold: float
    open_threshold: float
    blink_max_duration_seconds: float
    drowsiness_duration_seconds: float


@dataclass(frozen=True)
class HeadConfig:
    yaw_threshold_degrees: float
    pitch_threshold_degrees: float
    confirmation_seconds: float


@dataclass(frozen=True)
class PersonConfig:
    away_duration_seconds: float


@dataclass(frozen=True)
class AudioConfig:
    enabled: bool
    volume: float
    music_enabled: bool
    music_volume: float


@dataclass(frozen=True)
class UIConfig:
    debug: bool


@dataclass(frozen=True)
class ScoreConfig:
    starting_score: int
    phone_event_penalty: int
    drowsiness_event_penalty: int
    attention_event_penalty: int
    away_event_penalty: int


@dataclass(frozen=True)
class SessionConfig:
    max_event_log_entries: int


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    yolo: YoloConfig
    phone: PhoneConfig
    eyes: EyesConfig
    head: HeadConfig
    person: PersonConfig
    audio: AudioConfig
    ui: UIConfig
    score: ScoreConfig
    session: SessionConfig
    source_path: Path


class ConfigManager:
    """Loads config/config.yaml and validates it into an AppConfig."""

    def __init__(self, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        self._path = Path(path)

    def load(self) -> AppConfig:
        raw = self._read_yaml()
        return self._validate(raw)

    def _read_yaml(self) -> dict[str, Any]:
        if not self._path.exists():
            raise ConfigError(f"Configuration file not found: {self._path}")
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Configuration file is not valid YAML: {self._path} ({exc})"
            ) from exc
        if not isinstance(data, dict):
            raise ConfigError(
                f"Configuration file must contain a mapping at the top level: {self._path}"
            )
        return data

    def _validate(self, raw: dict[str, Any]) -> AppConfig:
        camera = self._section(raw, "camera")
        yolo = self._section(raw, "yolo")
        phone = self._section(raw, "phone")
        eyes = self._section(raw, "eyes")
        head = self._section(raw, "head")
        person = self._section(raw, "person")
        audio = self._section(raw, "audio")
        ui = self._section(raw, "ui")
        score = self._section(raw, "score")
        session = self._section(raw, "session")

        camera_cfg = CameraConfig(
            index=self._int(camera, "camera", "index", minimum=0),
            width=self._int(camera, "camera", "width", minimum=1),
            height=self._int(camera, "camera", "height", minimum=1),
            target_fps=self._int(camera, "camera", "target_fps", minimum=1),
        )

        yolo_device = self._str(yolo, "yolo", "device")
        if yolo_device not in VALID_YOLO_DEVICES:
            raise ConfigError(
                f"yolo.device must be one of {sorted(VALID_YOLO_DEVICES)}, got: {yolo_device!r}"
            )
        yolo_cfg = YoloConfig(
            model=self._str(yolo, "yolo", "model"),
            confidence=self._float(yolo, "yolo", "confidence", minimum=0.0, maximum=1.0),
            phone_confidence=self._float(yolo, "yolo", "phone_confidence", minimum=0.0, maximum=1.0),
            device=yolo_device,
        )

        phone_cfg = PhoneConfig(
            confirm_duration_seconds=self._float(phone, "phone", "confirm_duration_seconds", minimum=0.0),
            clear_duration_seconds=self._float(phone, "phone", "clear_duration_seconds", minimum=0.0),
            warning_cooldown_seconds=self._float(phone, "phone", "warning_cooldown_seconds", minimum=0.0),
        )

        closed_threshold = self._float(eyes, "eyes", "closed_threshold", minimum=0.0)
        open_threshold = self._float(eyes, "eyes", "open_threshold", minimum=0.0)
        if open_threshold <= closed_threshold:
            raise ConfigError(
                "eyes.open_threshold "
                f"({open_threshold}) must be greater than eyes.closed_threshold ({closed_threshold})"
            )
        eyes_cfg = EyesConfig(
            closed_threshold=closed_threshold,
            open_threshold=open_threshold,
            blink_max_duration_seconds=self._float(eyes, "eyes", "blink_max_duration_seconds", minimum=0.0),
            drowsiness_duration_seconds=self._float(eyes, "eyes", "drowsiness_duration_seconds", minimum=0.0),
        )

        head_cfg = HeadConfig(
            yaw_threshold_degrees=self._float(head, "head", "yaw_threshold_degrees", minimum=0.0, maximum=90.0),
            pitch_threshold_degrees=self._float(head, "head", "pitch_threshold_degrees", minimum=0.0, maximum=90.0),
            confirmation_seconds=self._float(head, "head", "confirmation_seconds", minimum=0.0),
        )

        person_cfg = PersonConfig(
            away_duration_seconds=self._float(person, "person", "away_duration_seconds", minimum=0.0),
        )

        audio_cfg = AudioConfig(
            enabled=self._bool(audio, "audio", "enabled"),
            volume=self._float(audio, "audio", "volume", minimum=0.0, maximum=1.0),
            music_enabled=self._bool(audio, "audio", "music_enabled"),
            music_volume=self._float(audio, "audio", "music_volume", minimum=0.0, maximum=1.0),
        )

        ui_cfg = UIConfig(debug=self._bool(ui, "ui", "debug"))

        score_cfg = ScoreConfig(
            starting_score=self._int(score, "score", "starting_score", minimum=0),
            phone_event_penalty=self._int(score, "score", "phone_event_penalty", minimum=0),
            drowsiness_event_penalty=self._int(score, "score", "drowsiness_event_penalty", minimum=0),
            attention_event_penalty=self._int(score, "score", "attention_event_penalty", minimum=0),
            away_event_penalty=self._int(score, "score", "away_event_penalty", minimum=0),
        )

        session_cfg = SessionConfig(
            max_event_log_entries=self._int(session, "session", "max_event_log_entries", minimum=1),
        )

        return AppConfig(
            camera=camera_cfg,
            yolo=yolo_cfg,
            phone=phone_cfg,
            eyes=eyes_cfg,
            head=head_cfg,
            person=person_cfg,
            audio=audio_cfg,
            ui=ui_cfg,
            score=score_cfg,
            session=session_cfg,
            source_path=self._path,
        )

    @staticmethod
    def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
        value = raw.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"Missing or invalid configuration section: {name}")
        return value

    @staticmethod
    def _int(section: dict[str, Any], section_name: str, field: str, *, minimum: int | None = None) -> int:
        key = f"{section_name}.{field}"
        value = section.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{key} must be an integer, got: {value!r}")
        if minimum is not None and value < minimum:
            raise ConfigError(f"{key} must be >= {minimum}, got: {value}")
        return value

    @staticmethod
    def _float(
        section: dict[str, Any],
        section_name: str,
        field: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        key = f"{section_name}.{field}"
        value = section.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{key} must be a number, got: {value!r}")
        value = float(value)
        if minimum is not None and value < minimum:
            raise ConfigError(f"{key} must be >= {minimum}, got: {value}")
        if maximum is not None and value > maximum:
            raise ConfigError(f"{key} must be <= {maximum}, got: {value}")
        return value

    @staticmethod
    def _bool(section: dict[str, Any], section_name: str, field: str) -> bool:
        key = f"{section_name}.{field}"
        value = section.get(field)
        if not isinstance(value, bool):
            raise ConfigError(f"{key} must be a boolean, got: {value!r}")
        return value

    @staticmethod
    def _str(section: dict[str, Any], section_name: str, field: str) -> str:
        key = f"{section_name}.{field}"
        value = section.get(field)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{key} must be a non-empty string, got: {value!r}")
        return value
