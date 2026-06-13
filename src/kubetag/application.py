import json
import logging
import os
from typing import Optional

from kubetag.config import load_config
from kubetag.domain import IssueEvent
from kubetag.github.client import GitHubClient
from kubetag.github.events import (
    IgnoredEventError,
    MalformedEventError,
    parse_issue_event,
)
from kubetag.inference.factory import create_predictor
from kubetag.inference.postprocessing import (
    postprocess_predictions,
    validate_artifacts,
    get_labels_to_apply,
    ArtifactValidationError,
)
from kubetag.text_processing import prepare_text

logger = logging.getLogger(__name__)

def run_application(
    title: Optional[str] = None,
    body: Optional[str] = None,
    event_path: Optional[str] = None,
    dry_run_override: Optional[bool] = None,
) -> int:
    """Run the KubeTag pipeline.
    
    Args:
        title: Explicit issue title (local dry-run).
        body: Explicit issue body (local dry-run).
        event_path: Path to the event file (GHA or local fixture).
        dry_run_override: Manual override for dry-run configuration.
        
    Returns:
        Exit code: 0 for success (or gracefully ignored event), non-zero for failure.
    """
    try:
        config = load_config()
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        return 1
        
    is_dry_run = config.dry_run
    if dry_run_override is not None:
        is_dry_run = dry_run_override
        
    if (
        not is_dry_run
        and config.predictor_backend == "development"
        and not config.allow_development_writes
    ):
        logger.error(
            "Development predictor cannot perform live writes unless "
            "ALLOW_DEVELOPMENT_WRITES=true."
        )
        return 1

    if not is_dry_run and not config.apply_labels:
        logger.info(
            "Live label application is disabled because APPLY_LABELS is false."
        )
        return 0

    try:
        validate_artifacts(config.model_dir, config.predictor_backend)
    except ArtifactValidationError as e:
        logger.error("Artifact validation failed: %s", e)
        return 1
        
    logger.info("Starting pipeline (dry_run=%s, backend=%s)", is_dry_run, config.predictor_backend)

    event_info: Optional[IssueEvent] = None
    
    if title is not None:
        logger.info("Running triage with explicit title/body inputs.")
        combined_text = prepare_text(title, body)
    elif event_path is not None:
        logger.info("Parsing event file from path: %s", event_path)
        try:
            event_info = parse_issue_event(event_path)
            combined_text = prepare_text(event_info.title, event_info.body)
        except IgnoredEventError as e:
            logger.info("Event ignored: %s", e)
            return 0
        except MalformedEventError as e:
            logger.error("Malformed event error: %s", e)
            return 1
    else:
        env_event_path = os.environ.get("GITHUB_EVENT_PATH")
        if not env_event_path:
            logger.error("Missing GITHUB_EVENT_PATH environment variable and no dry-run inputs provided.")
            return 1
            
        logger.info("Parsing event file from environment path: %s", env_event_path)
        try:
            event_info = parse_issue_event(env_event_path)
            combined_text = prepare_text(event_info.title, event_info.body)
        except IgnoredEventError as e:
            logger.info("Event ignored: %s", e)
            return 0
        except MalformedEventError as e:
            logger.error("Malformed event error: %s", e)
            return 1

    schema_path = os.path.join(config.model_dir, "label_schema.json")
    thresholds_path = os.path.join(config.model_dir, "thresholds.json")
    
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
    except Exception as e:
        logger.error(
            "Failed to load model artifacts schema/thresholds from '%s'. "
            "Ensure label_schema.json and thresholds.json exist: %s",
            config.model_dir, e
        )
        return 1

    try:
        predictor = create_predictor(backend_name=config.predictor_backend, model_dir=config.model_dir)
        raw_result = predictor.predict(combined_text)
    except NotImplementedError as e:
        logger.error("Backend not implemented: %s", e)
        return 1
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        return 1

    try:
        final_result = postprocess_predictions(raw_result, schema, thresholds)
    except Exception as e:
        logger.error("Prediction postprocessing failed: %s", e)
        return 1

    selected_labels = get_labels_to_apply(final_result)
    
    if not selected_labels:
        logger.info(
            "Abstaining: No predicted labels met the confidence threshold. "
            "Model Version: %s, Backend: %s, Inference Duration: %.2f ms",
            final_result.model_version, final_result.backend, final_result.inference_duration_ms
        )
        return 0
        
    scores_log = {
        p.label: f"{p.score:.4f} (threshold: {p.threshold:.2f})"
        for p in final_result.predictions
        if p.selected
    }
    
    logger.info(
        "Triage result: Applying labels: %s. Model Version: %s. Inference Duration: %.2f ms. Scores: %s",
        selected_labels, final_result.model_version, final_result.inference_duration_ms, scores_log
    )
    
    if is_dry_run:
        logger.info("=== DRY RUN: Labels to apply ===")
        print(f"Labels: {selected_labels}")
        logger.info("=== END DRY RUN ===")
        return 0

    if event_info is None:
        logger.error("Cannot perform GitHub label application without valid event payload.")
        return 1

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN environment variable is not set. Cannot authenticate with GitHub.")
        return 1

    try:
        client = GitHubClient(token=token, timeout_seconds=config.request_timeout_seconds)
        
        logger.info(
            "Verifying predicted labels exist in the target repository %s/%s...",
            event_info.owner, event_info.repo
        )
        repo_labels = client.get_repo_labels(event_info.owner, event_info.repo)
        
        missing_labels = [label for label in selected_labels if label not in repo_labels]
        if missing_labels:
            logger.error(
                "Repository Setup Error: The following predicted labels do not exist in repository '%s/%s': %s. "
                "Please create them first or update KubeTag schema definitions.",
                event_info.owner, event_info.repo, missing_labels
            )
            return 1
            
        logger.info(
            "Applying labels %s to issue %s/%s#%d...",
            selected_labels, event_info.owner, event_info.repo, event_info.issue_number
        )
        
        applied = client.add_labels_to_issue(
            owner=event_info.owner,
            repo=event_info.repo,
            issue_number=event_info.issue_number,
            labels=selected_labels
        )
        
        logger.info("Successfully applied labels. Current labels on issue: %s", applied)
        return 0
        
    except Exception as e:
        logger.error("Failed to perform GitHub label application: %s", e)
        return 1
