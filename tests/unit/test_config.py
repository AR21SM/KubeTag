import os
from unittest.mock import patch
from kubetag.config import load_config

def test_load_config_defaults() -> None:
    # Clear any environment variables that might interfere
    with patch.dict(os.environ, {}, clear=True):
        config = load_config()
        assert config.predictor_backend == "development"
        assert config.model_dir == "artifacts/model"
        assert config.dry_run is False
        assert config.apply_labels is False
        assert config.log_level == "INFO"
        assert config.request_timeout_seconds == 30
        assert config.allow_development_writes is False

def test_load_config_from_env() -> None:
    env_vars = {
        "PREDICTOR_BACKEND": "transformer",
        "MODEL_DIR": "/custom/path",
        "DRY_RUN": "true",
        "APPLY_LABELS": "yes",
        "LOG_LEVEL": "DEBUG",
        "REQUEST_TIMEOUT_SECONDS": "45",
        "ALLOW_DEVELOPMENT_WRITES": "true",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        config = load_config()
        assert config.predictor_backend == "transformer"
        assert config.model_dir == "/custom/path"
        assert config.dry_run is True
        assert config.apply_labels is True
        assert config.log_level == "DEBUG"
        assert config.request_timeout_seconds == 45
        assert config.allow_development_writes is True
