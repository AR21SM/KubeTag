from __future__ import annotations

import os
from pathlib import Path

from kubetag.inference.artifacts import ArtifactValidationError, load_artifacts


def main() -> None:
    repository = os.environ.get("KUBETAG_MODEL_REPOSITORY")
    if not repository:
        raise SystemExit("KUBETAG_MODEL_REPOSITORY is required")
    destination = Path(os.environ.get("MODEL_DIR", "artifacts/model"))
    revision = os.environ.get("KUBETAG_MODEL_REVISION") or None
    token = os.environ.get("HF_TOKEN") or None
    try:
        artifacts = load_artifacts(destination, "transformer")
    except ArtifactValidationError:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repository,
            revision=revision,
            local_dir=destination,
            token=token,
        )
        artifacts = load_artifacts(destination, "transformer")
    print(f"Model ready: {artifacts.version}")
