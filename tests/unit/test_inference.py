from kubetag.domain import LabelPrediction, PredictionResult
from kubetag.inference.development import DevelopmentPredictor
from kubetag.inference.postprocessing import postprocess_predictions

def test_deterministic_development_predictions(tmp_path) -> None:
    # Setup mock schema, thresholds, and manifest in a temp directory
    schema_file = tmp_path / "label_schema.json"
    thresholds_file = tmp_path / "thresholds.json"
    manifest_file = tmp_path / "model_manifest.json"
    
    schema_file.write_text("""{
        "labels": [
            {"name": "kind/bug", "taxonomy": "kind"},
            {"name": "sig/auth", "taxonomy": "sig"},
            {"name": "area/kubectl", "taxonomy": "area"}
        ]
    }""")
    thresholds_file.write_text("""{
        "kind/bug": 0.5,
        "sig/auth": 0.5,
        "area/kubectl": 0.5
    }""")
    manifest_file.write_text("""{
        "model_name": "test-model",
        "model_version": "1.0.0",
        "backend": "development",
        "trained_at": "now",
        "max_length": 512,
        "labels": ["kind/bug", "sig/auth", "area/kubectl"],
        "artifact_format": "manifest"
    }""")
    
    predictor = DevelopmentPredictor(model_dir=str(tmp_path))
    
    # Run prediction for text containing keywords "bug" and "kubectl"
    result = predictor.predict("This is a bug in the kubectl parser.")
    assert "1.0.0" in result.model_version
    assert result.backend == "development"
    
    # Check that predictions contain the list of labels
    preds = {p.label: p.score for p in result.predictions}
    assert preds["kind/bug"] == 0.8
    assert preds["area/kubectl"] == 0.9
    assert preds["sig/auth"] == 0.1  # no keyword auth or security

def test_threshold_filtering_and_sorting() -> None:
    schema = {
        "labels": [
            {"name": "kind/bug", "taxonomy": "kind"},
            {"name": "sig/auth", "taxonomy": "sig"},
            {"name": "area/kubectl", "taxonomy": "area"}
        ]
    }
    thresholds = {
        "kind/bug": 0.5,
        "sig/auth": 0.6,
        "area/kubectl": 0.7
    }
    
    # Input predictions out of order with varying scores
    raw_predictions = [
        LabelPrediction(label="area/kubectl", taxonomy="area", score=0.8, threshold=0.0, selected=False),
        LabelPrediction(label="kind/bug", taxonomy="kind", score=0.4, threshold=0.0, selected=False),
        LabelPrediction(label="sig/auth", taxonomy="sig", score=0.55, threshold=0.0, selected=False),
    ]
    
    raw_result = PredictionResult(
        model_version="1.0.0",
        backend="test",
        inference_duration_ms=1.5,
        predictions=raw_predictions
    )
    
    processed = postprocess_predictions(raw_result, schema, thresholds)
    
    # Verify sorting matches schema order: kind/bug, sig/auth, area/kubectl
    assert [p.label for p in processed.predictions] == ["kind/bug", "sig/auth", "area/kubectl"]
    
    # Check selection status:
    # area/kubectl: score 0.8 >= threshold 0.7 -> True
    # kind/bug: score 0.4 < threshold 0.5 -> False
    # sig/auth: score 0.55 < threshold 0.6 -> False
    preds_dict = {p.label: p for p in processed.predictions}
    assert preds_dict["area/kubectl"].selected is True
    assert preds_dict["kind/bug"].selected is False
    assert preds_dict["sig/auth"].selected is False
    assert preds_dict["area/kubectl"].threshold == 0.7

def test_unknown_label_rejection() -> None:
    schema = {
        "labels": [
            {"name": "kind/bug", "taxonomy": "kind"}
        ]
    }
    thresholds = {"kind/bug": 0.5}
    
    raw_predictions = [
        LabelPrediction(label="kind/bug", taxonomy="kind", score=0.8, threshold=0.0, selected=False),
        LabelPrediction(label="sig/unknown", taxonomy="sig", score=0.9, threshold=0.0, selected=False),
    ]
    
    raw_result = PredictionResult(
        model_version="1.0.0",
        backend="test",
        inference_duration_ms=1.0,
        predictions=raw_predictions
    )
    
    processed = postprocess_predictions(raw_result, schema, thresholds)
    
    # sig/unknown should be filtered out
    assert len(processed.predictions) == 1
    assert processed.predictions[0].label == "kind/bug"
