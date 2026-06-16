from __future__ import annotations

import json

import pytest

from kubetag.text_processing import PREPROCESSING_VERSION


@pytest.fixture
def artifact_factory():
    def create(path, backend="development", labels=None, nested_thresholds=False):
        labels = labels or ["kind/bug", "sig/node", "area/kubectl"]
        schema = {
            "labels": [
                {"name": label, "taxonomy": label.split("/", 1)[0]} for label in labels
            ]
        }
        thresholds = {label: 0.5 for label in labels}
        manifest = {
            "model_name": "test-model",
            "model_version": "test-v1",
            "backend": backend,
            "max_length": 512,
            "labels": labels,
        }
        if backend == "transformer":
            manifest["preprocessing_version"] = PREPROCESSING_VERSION
            (path / "final_model").mkdir()
            (path / "tokenizer").mkdir()
            (path / "COMPLETED").write_text("{}", encoding="utf-8")
        (path / "label_schema.json").write_text(json.dumps(schema), encoding="utf-8")
        (path / "thresholds.json").write_text(
            json.dumps({"strategy": "taxonomy", "values": thresholds})
            if nested_thresholds
            else json.dumps(thresholds),
            encoding="utf-8",
        )
        (path / "model_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return path

    return create
