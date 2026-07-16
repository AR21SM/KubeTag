from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kubetag.text_processing import PREPROCESSING_VERSION

SUPPORTED_TAXONOMIES = {"kind", "sig", "area"}


class ArtifactValidationError(Exception):
    pass


@dataclass(frozen=True)
class ModelArtifacts:
    directory: Path
    manifest: dict[str, Any]
    schema: dict[str, Any]
    labels: tuple[str, ...]
    thresholds: dict[str, float]
    version: str
    backend: str
    max_length: int


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactValidationError(f"Missing model artifact: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(
            f"Invalid model artifact {path.name}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ArtifactValidationError(
            f"Model artifact {path.name} must contain an object"
        )
    return payload


def load_artifacts(
    model_dir: str | Path,
    configured_backend: str | None = None,
    require_runtime: bool = True,
) -> ModelArtifacts:
    directory = Path(model_dir)
    manifest = _read_json(directory / "model_manifest.json")
    schema = _read_json(directory / "label_schema.json")
    threshold_payload = _read_json(directory / "thresholds.json")

    version = manifest.get("model_version")
    backend = manifest.get("backend")
    manifest_labels = manifest.get("labels")
    max_length = manifest.get("max_length")
    if not isinstance(version, str) or not version:
        raise ArtifactValidationError("model_manifest.json requires model_version")
    if not isinstance(backend, str):
        raise ArtifactValidationError("model_manifest.json requires backend")
    if configured_backend is not None and backend != configured_backend:
        raise ArtifactValidationError(
            f"Configured backend '{configured_backend}' does not match manifest backend '{backend}'"
        )
    if not isinstance(manifest_labels, list) or not all(
        isinstance(label, str) for label in manifest_labels
    ):
        raise ArtifactValidationError(
            "model_manifest.json requires an ordered labels list"
        )
    if len(manifest_labels) != len(set(manifest_labels)):
        raise ArtifactValidationError("model_manifest.json contains duplicate labels")
    if not isinstance(max_length, int) or max_length < 8:
        raise ArtifactValidationError("model_manifest.json requires a valid max_length")

    schema_items = schema.get("labels")
    if not isinstance(schema_items, list):
        raise ArtifactValidationError("label_schema.json requires a labels list")
    schema_labels = []
    for item in schema_items:
        if not isinstance(item, dict):
            raise ArtifactValidationError("label_schema.json contains an invalid label")
        name = item.get("name")
        taxonomy = item.get("taxonomy")
        if not isinstance(name, str) or not isinstance(taxonomy, str):
            raise ArtifactValidationError("Each label requires name and taxonomy")
        if taxonomy not in SUPPORTED_TAXONOMIES or not name.startswith(f"{taxonomy}/"):
            raise ArtifactValidationError(f"Invalid taxonomy for label '{name}'")
        schema_labels.append(name)
    if schema_labels != manifest_labels:
        raise ArtifactValidationError("Manifest and schema label order do not match")

    raw_thresholds = threshold_payload.get("values", threshold_payload)
    if not isinstance(raw_thresholds, dict) or set(raw_thresholds) != set(
        manifest_labels
    ):
        raise ArtifactValidationError("Threshold labels do not match the manifest")
    thresholds: dict[str, float] = {}
    for label in manifest_labels:
        value = raw_thresholds[label]
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ArtifactValidationError(f"Invalid threshold for '{label}'")
        thresholds[label] = float(value)

    if backend == "transformer":
        if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ArtifactValidationError(
                "Transformer preprocessing version does not match"
            )
    if backend == "transformer" and require_runtime:
        for name in ("final_model", "tokenizer"):
            if not (directory / name).is_dir():
                raise ArtifactValidationError(
                    f"Missing transformer artifact directory: {name}"
                )
        if not (directory / "COMPLETED").is_file():
            raise ArtifactValidationError("Transformer artifact is not marked complete")

    return ModelArtifacts(
        directory=directory,
        manifest=manifest,
        schema=schema,
        labels=tuple(manifest_labels),
        thresholds=thresholds,
        version=version,
        backend=backend,
        max_length=max_length,
    )
