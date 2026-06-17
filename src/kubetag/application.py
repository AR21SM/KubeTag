from __future__ import annotations

import logging
import os

from kubetag.config import AppConfig, load_config
from kubetag.domain import IssueEvent
from kubetag.github.client import GitHubClient, GitHubClientError
from kubetag.github.events import (
    IgnoredEventError,
    MalformedEventError,
    parse_issue_event,
)
from kubetag.inference.artifacts import ArtifactValidationError
from kubetag.inference.factory import create_predictor
from kubetag.inference.postprocessing import (
    get_labels_to_apply,
    postprocess_predictions,
)
from kubetag.text_processing import prepare_text

logger = logging.getLogger(__name__)


def _resolve_input(
    title: str | None,
    body: str | None,
    event_path: str | None,
) -> tuple[str, str | None, IssueEvent | None]:
    if title is not None:
        return title, body, None
    resolved_path = event_path or os.environ.get("GITHUB_EVENT_PATH")
    if not resolved_path:
        raise MalformedEventError("No issue input or GITHUB_EVENT_PATH was provided")
    event = parse_issue_event(resolved_path)
    return event.title, event.body, event


def _is_dry_run(config: AppConfig, override: bool | None) -> bool:
    return config.dry_run if override is None else override


def run_application(
    title: str | None = None,
    body: str | None = None,
    event_path: str | None = None,
    dry_run_override: bool | None = None,
    config: AppConfig | None = None,
) -> int:
    config = config or load_config()
    dry_run = _is_dry_run(config, dry_run_override)

    if (
        not dry_run
        and config.predictor_backend == "development"
        and not config.allow_development_writes
    ):
        logger.error("Development predictions cannot be written in live mode")
        return 1
    if not dry_run and not config.apply_labels:
        logger.info("Live label application is disabled")
        return 0

    try:
        issue_title, issue_body, event = _resolve_input(title, body, event_path)
    except IgnoredEventError as error:
        logger.info("Event ignored: %s", error)
        return 0
    except MalformedEventError as error:
        logger.error("Invalid issue event: %s", error)
        return 1

    try:
        predictor = create_predictor(config.predictor_backend, config.model_dir)
        text = prepare_text(issue_title, issue_body, predictor.artifacts.labels)
        raw_result = predictor.predict(text)
        result = postprocess_predictions(
            raw_result,
            predictor.artifacts.schema,
            predictor.artifacts.thresholds,
        )
    except (ArtifactValidationError, RuntimeError, ValueError, OSError) as error:
        logger.error("Prediction failed: %s", error)
        return 1

    selected_labels = get_labels_to_apply(result)
    scores = {
        prediction.label: {
            "score": round(prediction.score, 4),
            "threshold": prediction.threshold,
        }
        for prediction in result.predictions
        if prediction.selected
    }
    logger.info(
        "Prediction completed: labels=%s model=%s duration_ms=%.2f scores=%s",
        selected_labels,
        result.model_version,
        result.inference_duration_ms,
        scores,
    )

    if dry_run:
        print(f"Labels: {selected_labels}")
        return 0
    if not selected_labels:
        logger.info("No labels met their decision threshold")
        return 0
    if event is None:
        logger.error("Live writes require a GitHub issue event")
        return 1

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN is required for live writes")
        return 1

    try:
        client = GitHubClient(token, timeout_seconds=config.request_timeout_seconds)
        repository_labels = client.get_repo_labels(event.owner, event.repo)
        missing_labels = [
            label for label in selected_labels if label not in repository_labels
        ]
        if missing_labels:
            logger.error("Predicted repository labels do not exist: %s", missing_labels)
            return 1
        client.add_labels_to_issue(
            event.owner,
            event.repo,
            event.issue_number,
            selected_labels,
        )
    except GitHubClientError as error:
        logger.error("GitHub label application failed: %s", error)
        return 1

    logger.info("Applied labels to %s", event.html_url)
    return 0
