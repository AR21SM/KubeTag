import pytest

from kubetag.inference.development import DevelopmentPredictor
from kubetag.inference.factory import create_predictor


def test_development_predictor_is_deterministic(tmp_path, artifact_factory) -> None:
    artifact_factory(tmp_path)
    predictor = DevelopmentPredictor(tmp_path)
    first = predictor.predict("A kubelet bug")
    second = predictor.predict("A kubelet bug")
    assert [prediction.score for prediction in first.predictions] == [
        prediction.score for prediction in second.predictions
    ]
    scores = {prediction.label: prediction.score for prediction in first.predictions}
    assert scores["kind/bug"] == 0.9
    assert scores["sig/node"] == 0.9
    assert scores["area/kubectl"] == 0.1


def test_factory_rejects_unknown_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown PREDICTOR_BACKEND"):
        create_predictor("unknown", str(tmp_path))
