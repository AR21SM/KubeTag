import json
import pytest
from kubetag.domain import LabelPrediction, PredictionResult
from kubetag.inference.postprocessing import (
    postprocess_predictions,
    validate_artifacts,
    get_labels_to_apply,
    ArtifactValidationError,
)

def test_validate_artifacts_success(tmp_path) -> None:
    schema = {"labels": [{"name": "kind/bug", "taxonomy": "kind"}, {"name": "sig/auth", "taxonomy": "sig"}]}
    thresholds = {"kind/bug": 0.5, "sig/auth": 0.7}
    manifest = {
        "model_version": "v1.0",
        "backend": "development",
        "labels": ["kind/bug", "sig/auth"]
    }
    
    (tmp_path / "label_schema.json").write_text(json.dumps(schema))
    (tmp_path / "thresholds.json").write_text(json.dumps(thresholds))
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest))
    
    # Should not raise any exceptions
    validate_artifacts(str(tmp_path), "development")

def test_validate_artifacts_missing_file(tmp_path) -> None:
    with pytest.raises(ArtifactValidationError, match="Missing model artifact"):
        validate_artifacts(str(tmp_path), "development")

def test_validate_artifacts_mismatch_backend(tmp_path) -> None:
    schema = {"labels": [{"name": "kind/bug", "taxonomy": "kind"}]}
    thresholds = {"kind/bug": 0.5}
    manifest = {
        "model_version": "v1.0",
        "backend": "transformer",
        "labels": ["kind/bug"]
    }
    
    (tmp_path / "label_schema.json").write_text(json.dumps(schema))
    (tmp_path / "thresholds.json").write_text(json.dumps(thresholds))
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest))
    
    with pytest.raises(ArtifactValidationError, match="does not match manifest backend"):
        validate_artifacts(str(tmp_path), "development")

def test_validate_artifacts_invalid_threshold_range(tmp_path) -> None:
    schema = {"labels": [{"name": "kind/bug", "taxonomy": "kind"}]}
    thresholds = {"kind/bug": 1.5}
    manifest = {
        "model_version": "v1.0",
        "backend": "development",
        "labels": ["kind/bug"]
    }
    
    (tmp_path / "label_schema.json").write_text(json.dumps(schema))
    (tmp_path / "thresholds.json").write_text(json.dumps(thresholds))
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest))
    
    with pytest.raises(ArtifactValidationError, match="must be between 0 and 1"):
        validate_artifacts(str(tmp_path), "development")

def test_validate_artifacts_unsupported_taxonomy(tmp_path) -> None:
    schema = {"labels": [{"name": "kind/bug", "taxonomy": "invalid"}]}
    thresholds = {"kind/bug": 0.5}
    manifest = {
        "model_version": "v1.0",
        "backend": "development",
        "labels": ["kind/bug"]
    }
    
    (tmp_path / "label_schema.json").write_text(json.dumps(schema))
    (tmp_path / "thresholds.json").write_text(json.dumps(thresholds))
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest))
    
    with pytest.raises(ArtifactValidationError, match="unsupported taxonomy"):
        validate_artifacts(str(tmp_path), "development")

def test_postprocess_predictions_exact_equality_score() -> None:
    schema = {"labels": [{"name": "kind/bug", "taxonomy": "kind"}]}
    thresholds = {"kind/bug": 0.5}
    
    raw = PredictionResult(
        model_version="1.0",
        backend="development",
        inference_duration_ms=1.0,
        predictions=[LabelPrediction(label="kind/bug", taxonomy="kind", score=0.5, threshold=0.0, selected=False)]
    )
    
    res = postprocess_predictions(raw, schema, thresholds)
    assert len(res.predictions) == 1
    assert res.predictions[0].selected is True  # score >= threshold

def test_get_labels_to_apply_removes_duplicates() -> None:
    raw = PredictionResult(
        model_version="1.0",
        backend="development",
        inference_duration_ms=1.0,
        predictions=[
            LabelPrediction(label="kind/bug", taxonomy="kind", score=0.9, threshold=0.5, selected=True),
            LabelPrediction(label="kind/bug", taxonomy="kind", score=0.9, threshold=0.5, selected=True),
            LabelPrediction(label="sig/auth", taxonomy="sig", score=0.8, threshold=0.5, selected=True),
        ]
    )
    
    labels = get_labels_to_apply(raw)
    assert labels == ["kind/bug", "sig/auth"]
