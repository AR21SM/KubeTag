import os
from dataclasses import dataclass

class ConfigurationError(Exception):
    """Exception raised for configuration mismatch or invalid runtime settings."""
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

def load_config() -> AppConfig:
    """Load and validate the application configuration from environment variables."""
    predictor_backend = os.environ.get("PREDICTOR_BACKEND", "development").lower().strip()
    model_dir = os.environ.get("MODEL_DIR", "artifacts/model")
    
    dry_run_str = os.environ.get("DRY_RUN", "false").lower()
    dry_run = dry_run_str in ("true", "1", "yes")
    
    apply_labels_str = os.environ.get("APPLY_LABELS", "false").lower()
    apply_labels = apply_labels_str in ("true", "1", "yes")
    
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    
    timeout_str = os.environ.get("REQUEST_TIMEOUT_SECONDS", "30")
    try:
        request_timeout_seconds = int(timeout_str)
    except ValueError:
        request_timeout_seconds = 30
        
    allow_dev_str = os.environ.get("ALLOW_DEVELOPMENT_WRITES", "false").lower()
    allow_development_writes = allow_dev_str in ("true", "1", "yes")
        
    return AppConfig(
        predictor_backend=predictor_backend,
        model_dir=model_dir,
        dry_run=dry_run,
        apply_labels=apply_labels,
        log_level=log_level,
        request_timeout_seconds=request_timeout_seconds,
        allow_development_writes=allow_development_writes,
    )
