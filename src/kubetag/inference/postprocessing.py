from __future__ import annotations

import logging
from typing import Any

from kubetag.domain import LabelPrediction, PredictionResult
from kubetag.inference.artifacts import SUPPORTED_TAXONOMIES

logger = logging.getLogger(__name__)


def postprocess_predictions(
    result: PredictionResult,
    schema: dict[str, Any],
    thresholds: dict[str, float],
) -> PredictionResult:
    schema_order = {
        item["name"]: index for index, item in enumerate(schema.get("labels", []))
    }
    schema_taxonomy = {
        item["name"]: item["taxonomy"] for item in schema.get("labels", [])
    }
    predictions = []
    for prediction in result.predictions:
        taxonomy = schema_taxonomy.get(prediction.label)
        if taxonomy not in SUPPORTED_TAXONOMIES:
            logger.warning("Ignoring unknown prediction label: %s", prediction.label)
            continue
        if prediction.taxonomy != taxonomy:
            logger.warning("Ignoring taxonomy mismatch for: %s", prediction.label)
            continue
        threshold = thresholds[prediction.label]
        predictions.append(
            LabelPrediction(
                label=prediction.label,
                taxonomy=taxonomy,
                score=prediction.score,
                threshold=threshold,
                selected=prediction.score >= threshold,
            )
        )
    predictions.sort(key=lambda prediction: schema_order[prediction.label])
    return PredictionResult(
        model_version=result.model_version,
        backend=result.backend,
        inference_duration_ms=result.inference_duration_ms,
        predictions=predictions,
    )


def get_labels_to_apply(result: PredictionResult) -> list[str]:
    return list(
        dict.fromkeys(
            prediction.label for prediction in result.predictions if prediction.selected
        )
    )
