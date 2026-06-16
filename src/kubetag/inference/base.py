from typing import Protocol

from kubetag.domain import PredictionResult
from kubetag.inference.artifacts import ModelArtifacts


class Predictor(Protocol):
    artifacts: ModelArtifacts

    def predict(self, text: str) -> PredictionResult: ...
