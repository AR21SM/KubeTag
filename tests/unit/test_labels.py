from kubetag.domain import LabelPrediction, PredictionResult
from kubetag.inference.postprocessing import get_labels_to_apply

def test_get_labels_to_apply_success() -> None:
    predictions = [
        LabelPrediction(label="kind/bug", taxonomy="kind", score=0.8, threshold=0.5, selected=True),
        LabelPrediction(label="sig/auth", taxonomy="sig", score=0.9, threshold=0.5, selected=True),
        LabelPrediction(label="area/kubectl", taxonomy="area", score=0.4, threshold=0.5, selected=False),
    ]
    result = PredictionResult(
        model_version="1.0.0",
        backend="development",
        inference_duration_ms=0.5,
        predictions=predictions
    )
    
    labels = get_labels_to_apply(result)
    assert labels == ["kind/bug", "sig/auth"]

def test_get_labels_to_apply_empty() -> None:
    predictions = [
        LabelPrediction(label="kind/bug", taxonomy="kind", score=0.3, threshold=0.5, selected=False),
        LabelPrediction(label="sig/auth", taxonomy="sig", score=0.2, threshold=0.5, selected=False),
    ]
    result = PredictionResult(
        model_version="1.0.0",
        backend="development",
        inference_duration_ms=0.5,
        predictions=predictions
    )
    
    labels = get_labels_to_apply(result)
    assert labels == []
