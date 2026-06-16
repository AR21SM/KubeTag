from __future__ import annotations

import time
from pathlib import Path

from kubetag.domain import LabelPrediction, PredictionResult
from kubetag.inference.artifacts import ArtifactValidationError, load_artifacts
from kubetag.text_processing import encode_head_tail


class TransformerPredictor:
    def __init__(self, model_dir: str | Path) -> None:
        self.artifacts = load_artifacts(model_dir, "transformer")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Transformer dependencies are missing; install KubeTag with the model extra"
            ) from error

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.artifacts.directory / "tokenizer",
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.artifacts.directory / "final_model",
            local_files_only=True,
        )
        model_labels = [
            self.model.config.id2label[index]
            for index in range(self.model.config.num_labels)
        ]
        if model_labels != list(self.artifacts.labels):
            raise ArtifactValidationError("Model and manifest label order do not match")
        self.model.to(self.device).eval()

    def predict(self, text: str) -> PredictionResult:
        started = time.perf_counter()
        encoded = encode_head_tail(
            self.tokenizer,
            [text],
            self.artifacts.max_length,
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with self._torch.inference_mode():
            logits = self.model(**encoded).logits.float()
            scores = self._torch.sigmoid(logits).cpu().tolist()[0]
        predictions = [
            LabelPrediction(
                label=label,
                taxonomy=label.split("/", 1)[0],
                score=float(score),
                threshold=0.0,
                selected=False,
            )
            for label, score in zip(self.artifacts.labels, scores)
        ]
        return PredictionResult(
            model_version=self.artifacts.version,
            backend="transformer",
            inference_duration_ms=(time.perf_counter() - started) * 1000.0,
            predictions=predictions,
        )
