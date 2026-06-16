import json

import pytest

from kubetag.inference.artifacts import ArtifactValidationError, load_artifacts


def test_loads_nested_training_thresholds(tmp_path, artifact_factory) -> None:
    artifact_factory(tmp_path, backend="transformer", nested_thresholds=True)
    artifacts = load_artifacts(tmp_path, "transformer")
    assert artifacts.labels == ("kind/bug", "sig/node", "area/kubectl")
    assert artifacts.thresholds["sig/node"] == 0.5
    assert artifacts.max_length == 512


def test_rejects_backend_mismatch(tmp_path, artifact_factory) -> None:
    artifact_factory(tmp_path)
    with pytest.raises(ArtifactValidationError, match="does not match"):
        load_artifacts(tmp_path, "transformer")


def test_rejects_label_order_mismatch(tmp_path, artifact_factory) -> None:
    artifact_factory(tmp_path)
    manifest_path = tmp_path / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["labels"] = list(reversed(manifest["labels"]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="label order"):
        load_artifacts(tmp_path, "development")


def test_rejects_incomplete_transformer_bundle(tmp_path, artifact_factory) -> None:
    artifact_factory(tmp_path, backend="transformer")
    (tmp_path / "COMPLETED").unlink()
    with pytest.raises(ArtifactValidationError, match="not marked complete"):
        load_artifacts(tmp_path, "transformer")
