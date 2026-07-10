from typing import Protocol
from kubetag.domain import PredictionResult

class Predictor(Protocol):
    """Protocol defining the Predictor interface."""
    
    def predict(self, text: str) -> PredictionResult:
        """Run inference on the preprocessed text and return predictions.
        
        Args:
            text: Preprocessed text content of the issue (title + body).
            
        Returns:
            A PredictionResult containing predictions and inference metadata.
        """
        ...
