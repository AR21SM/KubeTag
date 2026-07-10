import logging
from kubetag.inference.base import Predictor
from kubetag.inference.development import DevelopmentPredictor

logger = logging.getLogger(__name__)

def create_predictor(backend_name: str, model_dir: str) -> Predictor:
    """Factory function to instantiate a Predictor based on the configuration.
    
    Args:
        backend_name: Name of the predictor backend (development, transformer, linear_svc).
        model_dir: Path to directory containing model artifacts.
        
    Returns:
        An instantiated Predictor.
        
    Raises:
        ValueError: If backend_name is unknown.
        NotImplementedError: If requesting a backend that is not yet fully implemented.
    """
    normalized_backend = backend_name.lower().strip()
    logger.info("Initializing predictor backend: '%s' from dir: '%s'", normalized_backend, model_dir)
    
    if normalized_backend == "development":
        return DevelopmentPredictor(model_dir=model_dir)
        
    elif normalized_backend == "transformer":
        raise NotImplementedError("The 'transformer' backend is not implemented.")
        
    elif normalized_backend == "linear_svc":
        raise NotImplementedError("The 'linear_svc' backend is not implemented.")
        
    else:
        raise ValueError(f"Unknown PREDICTOR_BACKEND: '{backend_name}'")
