from __future__ import annotations

import time
from pathlib import Path

from kubetag.domain import LabelPrediction, PredictionResult
from kubetag.inference.artifacts import load_artifacts


class DevelopmentPredictor:
    def __init__(self, model_dir: str | Path) -> None:
        self.artifacts = load_artifacts(model_dir, require_runtime=False)

    def predict(self, text: str) -> PredictionResult:
        started = time.perf_counter()
        text_lower = text.lower()
        keywords = {
            "kind/bug": ("bug",),
            "kind/feature": ("feature",),
            "kind/cleanup": ("cleanup",),
            "sig/auth": ("auth", "security"),
            "sig/cli": ("cli", "kubectl"),
            "sig/scheduling": ("scheduling", "scheduler"),
            "sig/node": ("node", "kubelet"),
            "sig/api-machinery": ("api", "apiserver"),
            "area/kubectl": ("kubectl",),
            "area/provider": ("provider", "cloud"),
            "area/security": ("security", "cert"),
        }
        predictions = []
        for label in self.artifacts.labels:
            matched = any(word in text_lower for word in keywords.get(label, ()))
            predictions.append(
                LabelPrediction(
                    label=label,
                    taxonomy=label.split("/", 1)[0],
                    score=0.9 if matched else 0.1,
                    threshold=0.0,
                    selected=False,
                )
            )
        return PredictionResult(
            model_version=f"{self.artifacts.version} (non-production)",
            backend="development",
            inference_duration_ms=(time.perf_counter() - started) * 1000.0,
            predictions=predictions,
        )
