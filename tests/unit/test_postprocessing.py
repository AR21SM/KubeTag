from kubetag.domain import LabelPrediction, PredictionResult
from kubetag.inference.postprocessing import (
    get_labels_to_apply,
    postprocess_predictions,
)


def _result(predictions) -> PredictionResult:
    return PredictionResult(
        model_version="test-v1",
        backend="transformer",
        inference_duration_ms=1.0,
        predictions=predictions,
    )


def test_thresholds_and_orders_predictions() -> None:
    schema = {
        "labels": [
            {"name": "kind/bug", "taxonomy": "kind"},
            {"name": "sig/node", "taxonomy": "sig"},
        ]
    }
    raw = _result(
        [
            LabelPrediction("sig/node", "sig", 0.8, 0.0, False),
            LabelPrediction("kind/bug", "kind", 0.4, 0.0, False),
        ]
    )
    result = postprocess_predictions(raw, schema, {"kind/bug": 0.5, "sig/node": 0.7})
    assert [prediction.label for prediction in result.predictions] == [
        "kind/bug",
        "sig/node",
    ]
    assert get_labels_to_apply(result) == ["sig/node"]


def test_ignores_unknown_and_mismatched_predictions() -> None:
    schema = {"labels": [{"name": "kind/bug", "taxonomy": "kind"}]}
    raw = _result(
        [
            LabelPrediction("kind/bug", "sig", 0.9, 0.0, False),
            LabelPrediction("sig/unknown", "sig", 0.9, 0.0, False),
        ]
    )
    result = postprocess_predictions(raw, schema, {"kind/bug": 0.5})
    assert result.predictions == []


def test_removes_selected_label_duplicates() -> None:
    result = _result(
        [
            LabelPrediction("kind/bug", "kind", 0.9, 0.5, True),
            LabelPrediction("kind/bug", "kind", 0.9, 0.5, True),
            LabelPrediction("sig/node", "sig", 0.8, 0.5, True),
        ]
    )
    assert get_labels_to_apply(result) == ["kind/bug", "sig/node"]
