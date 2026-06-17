from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kubetag.application import run_application
from kubetag.config import AppConfig
from kubetag.domain import IssueEvent, LabelPrediction, PredictionResult


def _config(**overrides) -> AppConfig:
    values = {
        "predictor_backend": "transformer",
        "model_dir": "artifacts/model",
        "dry_run": True,
        "apply_labels": False,
        "log_level": "INFO",
        "request_timeout_seconds": 30,
        "allow_development_writes": False,
    }
    values.update(overrides)
    return AppConfig(**values)


def _predictor(score=0.9):
    artifacts = SimpleNamespace(
        labels=("kind/bug",),
        schema={"labels": [{"name": "kind/bug", "taxonomy": "kind"}]},
        thresholds={"kind/bug": 0.5},
    )
    predictor = MagicMock(artifacts=artifacts)
    predictor.predict.return_value = PredictionResult(
        model_version="test-v1",
        backend="transformer",
        inference_duration_ms=1.0,
        predictions=[LabelPrediction("kind/bug", "kind", score, 0.0, False)],
    )
    return predictor


def _event() -> IssueEvent:
    return IssueEvent(
        owner="owner",
        repo="repo",
        issue_number=42,
        title="Kubelet bug",
        body="The node is failing",
        action="opened",
        html_url="https://github.com/owner/repo/issues/42",
    )


@patch("kubetag.application.create_predictor")
def test_dry_run_prints_selected_labels(create_predictor, capsys) -> None:
    create_predictor.return_value = _predictor()
    code = run_application(title="Kubelet bug", body="Node failure", config=_config())
    assert code == 0
    assert "Labels: ['kind/bug']" in capsys.readouterr().out


@patch("kubetag.application.create_predictor")
def test_development_writes_are_blocked(create_predictor) -> None:
    config = _config(
        predictor_backend="development",
        dry_run=False,
        apply_labels=True,
    )
    assert run_application(title="Bug", config=config) == 1
    create_predictor.assert_not_called()


@patch("kubetag.application.create_predictor")
def test_disabled_live_writes_exit_before_loading_model(create_predictor) -> None:
    config = _config(dry_run=False, apply_labels=False)
    assert run_application(event_path="event.json", config=config) == 0
    create_predictor.assert_not_called()


@patch("kubetag.application._resolve_input")
@patch("kubetag.application.create_predictor")
def test_live_write_requires_token(
    create_predictor, resolve_input, monkeypatch
) -> None:
    create_predictor.return_value = _predictor()
    resolve_input.return_value = ("Kubelet bug", "Node failure", _event())
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    config = _config(dry_run=False, apply_labels=True)
    assert run_application(event_path="event.json", config=config) == 1


@patch("kubetag.application.GitHubClient")
@patch("kubetag.application._resolve_input")
@patch("kubetag.application.create_predictor")
def test_live_write_validates_repository_labels(
    create_predictor,
    resolve_input,
    client_class,
    monkeypatch,
) -> None:
    create_predictor.return_value = _predictor()
    resolve_input.return_value = ("Kubelet bug", "Node failure", _event())
    client_class.return_value.get_repo_labels.return_value = ["sig/node"]
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    config = _config(dry_run=False, apply_labels=True)
    assert run_application(event_path="event.json", config=config) == 1
    client_class.return_value.add_labels_to_issue.assert_not_called()


@patch("kubetag.application.GitHubClient")
@patch("kubetag.application._resolve_input")
@patch("kubetag.application.create_predictor")
def test_live_write_applies_valid_labels(
    create_predictor,
    resolve_input,
    client_class,
    monkeypatch,
) -> None:
    create_predictor.return_value = _predictor()
    resolve_input.return_value = ("Kubelet bug", "Node failure", _event())
    client_class.return_value.get_repo_labels.return_value = ["kind/bug"]
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    config = _config(dry_run=False, apply_labels=True)
    assert run_application(event_path="event.json", config=config) == 0
    client_class.return_value.add_labels_to_issue.assert_called_once_with(
        "owner",
        "repo",
        42,
        ["kind/bug"],
    )
