from __future__ import annotations

import os
from dataclasses import dataclass

SUPPORTED_BACKENDS = {"development", "transformer"}
TRUE_VALUES = {"1", "true", "yes"}


class ConfigurationError(Exception):
    pass


@dataclass(frozen=True)
class AppConfig:
    predictor_backend: str
    model_dir: str
    dry_run: bool
    apply_labels: bool
    log_level: str
    request_timeout_seconds: int
    allow_development_writes: bool


def _boolean(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in TRUE_VALUES


def load_config() -> AppConfig:
    backend = os.environ.get("PREDICTOR_BACKEND", "development").strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ConfigurationError(f"Unsupported PREDICTOR_BACKEND: {backend}")
    try:
        timeout = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))
    except ValueError as error:
        raise ConfigurationError(
            "REQUEST_TIMEOUT_SECONDS must be an integer"
        ) from error
    if timeout <= 0:
        raise ConfigurationError("REQUEST_TIMEOUT_SECONDS must be positive")
    return AppConfig(
        predictor_backend=backend,
        model_dir=os.environ.get("MODEL_DIR", "artifacts/model"),
        dry_run=_boolean("DRY_RUN"),
        apply_labels=_boolean("APPLY_LABELS"),
        log_level=os.environ.get("LOG_LEVEL", "INFO").strip().upper(),
        request_timeout_seconds=timeout,
        allow_development_writes=_boolean("ALLOW_DEVELOPMENT_WRITES"),
    )
