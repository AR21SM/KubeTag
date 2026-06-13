import json
import logging
import os
from typing import Any, Dict, List
from kubetag.domain import LabelPrediction, PredictionResult

logger = logging.getLogger(__name__)

SUPPORTED_TAXONOMIES = {"kind", "sig", "area"}

class ArtifactValidationError(Exception):
    """Raised when model artifact validation fails."""
    pass

def validate_artifacts(model_dir: str, configured_backend: str) -> None:
    """Validate that the model artifacts exist and match schema/threshold rules.
    
    Raises:
        ArtifactValidationError: If any validation rule is violated.
    """
    schema_path = os.path.join(model_dir, "label_schema.json")
    thresholds_path = os.path.join(model_dir, "thresholds.json")
    manifest_path = os.path.join(model_dir, "model_manifest.json")
    
    if not os.path.exists(schema_path):
        raise ArtifactValidationError(f"Missing model artifact: label_schema.json at {schema_path}")
    if not os.path.exists(thresholds_path):
        raise ArtifactValidationError(f"Missing model artifact: thresholds.json at {thresholds_path}")
    if not os.path.exists(manifest_path):
        raise ArtifactValidationError(f"Missing model artifact: model_manifest.json at {manifest_path}")
        
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        raise ArtifactValidationError(f"Failed to parse JSON artifacts: {e}") from e
        
    model_version = manifest.get("model_version")
    if not model_version:
        raise ArtifactValidationError("model_manifest.json is missing 'model_version'.")
        
    manifest_backend = manifest.get("backend")
    if not manifest_backend:
        raise ArtifactValidationError("model_manifest.json is missing 'backend'.")
    if manifest_backend.lower().strip() != configured_backend.lower().strip():
        raise ArtifactValidationError(
            f"Configured backend '{configured_backend}' does not match manifest backend '{manifest_backend}'."
        )
        
    schema_labels = schema.get("labels")
    if not isinstance(schema_labels, list):
        raise ArtifactValidationError("label_schema.json must contain a list of labels.")
        
    schema_label_names = []
    for idx, item in enumerate(schema_labels):
        if not isinstance(item, dict):
            raise ArtifactValidationError(f"Label item at index {idx} is not a dictionary.")
        name = item.get("name")
        taxonomy = item.get("taxonomy")
        if not name:
            raise ArtifactValidationError(f"Label item at index {idx} is missing 'name'.")
        if not taxonomy:
            raise ArtifactValidationError(f"Label '{name}' is missing 'taxonomy'.")
        if taxonomy not in SUPPORTED_TAXONOMIES:
            raise ArtifactValidationError(
                f"Label '{name}' has unsupported taxonomy '{taxonomy}'. Must be kind, sig, or area."
            )
        schema_label_names.append(name)
        
    manifest_labels = manifest.get("labels")
    if not isinstance(manifest_labels, list):
        raise ArtifactValidationError("model_manifest.json must contain a list of labels.")
    if manifest_labels != schema_label_names:
        raise ArtifactValidationError(
            "Labels defined in model_manifest.json do not exactly match the names or order in label_schema.json."
        )
        
    for label_name in schema_label_names:
        if label_name not in thresholds:
            raise ArtifactValidationError(f"Schema label '{label_name}' has no threshold defined in thresholds.json.")
            
    for threshold_key, val in thresholds.items():
        if threshold_key not in schema_label_names:
            raise ArtifactValidationError(
                f"Threshold defined for unknown label '{threshold_key}' not present in label_schema.json."
            )
        if not isinstance(val, (int, float)):
            raise ArtifactValidationError(f"Threshold for label '{threshold_key}' must be a number.")
        if not (0.0 <= val <= 1.0):
            raise ArtifactValidationError(f"Threshold for label '{threshold_key}' must be between 0 and 1.")

def postprocess_predictions(
    result: PredictionResult,
    schema: Dict[str, Any],
    thresholds: Dict[str, Any]
) -> PredictionResult:
    """Validate, threshold, and order predictions based on the label schema.
    
    Args:
        result: The raw PredictionResult from the predictor backend.
        schema: The label schema dict loaded from label_schema.json.
        thresholds: The decision thresholds dict loaded from thresholds.json.
        
    Returns:
        A new PredictionResult with validated, thresholded, and ordered predictions.
    """
    labels_in_schema = schema.get("labels", [])
    
    schema_order: Dict[str, int] = {}
    schema_taxonomy: Dict[str, str] = {}
    
    for idx, item in enumerate(labels_in_schema):
        name = item.get("name")
        taxonomy = item.get("taxonomy")
        if name and taxonomy:
            schema_order[name] = idx
            schema_taxonomy[name] = taxonomy

    validated_predictions: List[LabelPrediction] = []
    
    for pred in result.predictions:
        label_name = pred.label
        
        if label_name not in schema_order:
            logger.warning("Rejecting prediction: label '%s' not present in schema.", label_name)
            continue
            
        expected_taxonomy = schema_taxonomy[label_name]
        if expected_taxonomy not in SUPPORTED_TAXONOMIES:
            logger.warning("Rejecting prediction: taxonomy '%s' is not supported.", expected_taxonomy)
            continue
            
        if pred.taxonomy != expected_taxonomy:
            logger.warning(
                "Rejecting prediction: taxonomy mismatch for '%s' (got '%s', expected '%s').",
                label_name, pred.taxonomy, expected_taxonomy
            )
            continue
            
        threshold = thresholds.get(label_name, 0.5)
        is_selected = pred.score >= threshold
        
        validated_predictions.append(
            LabelPrediction(
                label=label_name,
                taxonomy=expected_taxonomy,
                score=pred.score,
                threshold=threshold,
                selected=is_selected
            )
        )
        
    validated_predictions.sort(key=lambda p: schema_order[p.label])
    
    return PredictionResult(
        model_version=result.model_version,
        backend=result.backend,
        inference_duration_ms=result.inference_duration_ms,
        predictions=validated_predictions
    )

def get_labels_to_apply(result: PredictionResult) -> List[str]:
    """Extract the list of selected label names from the validated PredictionResult.
    
    Args:
        result: The postprocessed PredictionResult.
        
    Returns:
        A list of label strings to be applied.
    """
    seen = set()
    labels = []
    for p in result.predictions:
        if p.selected and p.label not in seen:
            seen.add(p.label)
            labels.append(p.label)
    return labels
