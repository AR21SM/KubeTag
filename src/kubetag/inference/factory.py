from kubetag.inference.base import Predictor
from kubetag.inference.development import DevelopmentPredictor


def create_predictor(backend_name: str, model_dir: str) -> Predictor:
    backend = backend_name.strip().lower()
    if backend == "development":
        return DevelopmentPredictor(model_dir)
    if backend == "transformer":
        from kubetag.inference.transformer import TransformerPredictor

        return TransformerPredictor(model_dir)
    raise ValueError(f"Unknown PREDICTOR_BACKEND: '{backend_name}'")
