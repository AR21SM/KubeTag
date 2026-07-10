import os
from unittest.mock import MagicMock, patch
from kubetag.application import run_application
from kubetag.config import AppConfig

@patch("kubetag.application.load_config")
@patch("kubetag.application.validate_artifacts")
def test_development_backend_blocked_in_live_mode(mock_val, mock_load) -> None:
    mock_load.return_value = AppConfig(
        predictor_backend="development",
        model_dir="artifacts/model",
        dry_run=False,
        apply_labels=True,
        log_level="INFO",
        request_timeout_seconds=30,
        allow_development_writes=False
    )
    
    code = run_application(title="title", body="body")
    assert code == 1
    mock_val.assert_not_called()

@patch("kubetag.application.load_config")
@patch("kubetag.application.validate_artifacts")
@patch("kubetag.application.create_predictor")
@patch("kubetag.application.postprocess_predictions")
def test_missing_github_token_allowed_during_dry_run(mock_post, mock_pred, mock_val, mock_load) -> None:
    mock_load.return_value = AppConfig(
        predictor_backend="development",
        model_dir="artifacts/model",
        dry_run=True,
        apply_labels=False,
        log_level="INFO",
        request_timeout_seconds=30,
        allow_development_writes=False
    )
    
    mock_predictor = MagicMock()
    mock_pred.return_value = mock_predictor
    
    mock_post.return_value = MagicMock(predictions=[])
    
    with patch.dict(os.environ, {}, clear=True):
        code = run_application(title="title", body="body")
        assert code == 0

@patch("kubetag.application.load_config")
@patch("kubetag.application.validate_artifacts")
@patch("kubetag.application.create_predictor")
@patch("kubetag.application.postprocess_predictions")
@patch("kubetag.application.parse_issue_event")
def test_missing_github_token_rejected_in_live_mode(mock_parse, mock_post, mock_pred, mock_val, mock_load) -> None:
    mock_load.return_value = AppConfig(
        predictor_backend="transformer",
        model_dir="artifacts/model",
        dry_run=False,
        apply_labels=True,
        log_level="INFO",
        request_timeout_seconds=30,
        allow_development_writes=False
    )
    
    from kubetag.domain import LabelPrediction, PredictionResult
    mock_post.return_value = PredictionResult(
        model_version="v1",
        backend="transformer",
        inference_duration_ms=1.0,
        predictions=[LabelPrediction(label="kind/bug", taxonomy="kind", score=0.9, threshold=0.5, selected=True)]
    )
    
    mock_parse.return_value = MagicMock()
    
    with patch.dict(os.environ, {}, clear=True):
        code = run_application(event_path="event.json")
        assert code == 1

@patch("kubetag.application.load_config")
@patch("kubetag.application.validate_artifacts")
@patch("kubetag.application.create_predictor")
@patch("kubetag.application.postprocess_predictions")
@patch("kubetag.application.parse_issue_event")
@patch("kubetag.application.GitHubClient")
def test_nonexistent_repository_label_error(mock_client_class, mock_parse, mock_post, mock_pred, mock_val, mock_load) -> None:
    mock_load.return_value = AppConfig(
        predictor_backend="transformer",
        model_dir="artifacts/model",
        dry_run=False,
        apply_labels=True,
        log_level="INFO",
        request_timeout_seconds=30,
        allow_development_writes=False
    )
    
    from kubetag.domain import LabelPrediction, PredictionResult
    mock_post.return_value = PredictionResult(
        model_version="v1",
        backend="transformer",
        inference_duration_ms=1.0,
        predictions=[LabelPrediction(label="kind/bug", taxonomy="kind", score=0.9, threshold=0.5, selected=True)]
    )
    
    mock_parse.return_value = MagicMock(owner="owner", repo="repo", issue_number=123)
    
    mock_client = MagicMock()
    mock_client.get_repo_labels.return_value = ["sig/auth"]
    mock_client_class.return_value = mock_client
    
    with patch.dict(os.environ, {"GITHUB_TOKEN": "fake"}):
        code = run_application(event_path="event.json")
        assert code == 1
        mock_client.add_labels_to_issue.assert_not_called()

@patch("kubetag.application.load_config")
@patch("kubetag.application.validate_artifacts")
@patch("kubetag.application.create_predictor")
@patch("kubetag.application.postprocess_predictions")
@patch("kubetag.application.parse_issue_event")
@patch("kubetag.application.GitHubClient")
def test_apply_labels_false_early_exit(mock_client_class, mock_parse, mock_post, mock_pred, mock_val, mock_load) -> None:
    mock_load.return_value = AppConfig(
        predictor_backend="transformer",
        model_dir="artifacts/model",
        dry_run=False,
        apply_labels=False,
        log_level="INFO",
        request_timeout_seconds=30,
        allow_development_writes=False
    )
    
    code = run_application(event_path="event.json")
    assert code == 0
    mock_val.assert_not_called()
    mock_client_class.assert_not_called()

