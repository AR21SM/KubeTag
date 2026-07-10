import json
import os
import time
from typing import List
from kubetag.domain import LabelPrediction, PredictionResult
from kubetag.inference.base import Predictor

class DevelopmentPredictor(Predictor):
    """A deterministic predictor that simulates classification using simple keyword matching.
    
    Used only for testing and development.
    """
    
    def __init__(self, model_dir: str) -> None:
        """Initialize the development predictor by loading schema and manifest files.
        
        Args:
            model_dir: Directory containing the model metadata files.
        """
        self.model_dir = model_dir
        self.version = "0.0.1-dev-mock"
        
        schema_path = os.path.join(model_dir, "label_schema.json")
        thresholds_path = os.path.join(model_dir, "thresholds.json")
        manifest_path = os.path.join(model_dir, "model_manifest.json")
        
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load label schema from {schema_path}: {e}")
            
        try:
            with open(thresholds_path, "r", encoding="utf-8") as f:
                self.thresholds = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load thresholds from {thresholds_path}: {e}")
            
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                self.version = manifest.get("model_version", self.version)
        except Exception as e:
            raise RuntimeError(f"Failed to load model manifest from {manifest_path}: {e}")

    def predict(self, text: str) -> PredictionResult:
        """Run deterministic keyword-based mock inference on input text.
        
        Args:
            text: Preprocessed text content (title + body).
            
        Returns:
            A PredictionResult populated with mock predictions.
        """
        start_time = time.perf_counter()
        text_lower = text.lower()
        
        predictions: List[LabelPrediction] = []
        labels_list = self.schema.get("labels", [])
        
        for label_info in labels_list:
            label_name = label_info.get("name")
            taxonomy = label_info.get("taxonomy")
            if not label_name or not taxonomy:
                continue
                
            threshold = self.thresholds.get(label_name, 0.5)
            score = 0.1
            
            if label_name == "kind/bug" and "bug" in text_lower:
                score = 0.8
            elif label_name == "kind/feature" and "feature" in text_lower:
                score = 0.8
            elif label_name == "kind/cleanup" and "cleanup" in text_lower:
                score = 0.8
            elif label_name == "sig/auth" and ("auth" in text_lower or "security" in text_lower):
                score = 0.9
            elif label_name == "sig/cli" and ("cli" in text_lower or "kubectl" in text_lower):
                score = 0.85
            elif label_name == "sig/scheduling" and ("scheduling" in text_lower or "scheduler" in text_lower):
                score = 0.9
            elif label_name == "sig/node" and ("node" in text_lower or "kubelet" in text_lower):
                score = 0.8
            elif label_name == "sig/api-machinery" and ("api" in text_lower or "apiserver" in text_lower):
                score = 0.8
            elif label_name == "area/kubectl" and "kubectl" in text_lower:
                score = 0.9
            elif label_name == "area/provider" and ("provider" in text_lower or "cloud" in text_lower):
                score = 0.8
            elif label_name == "area/security" and ("security" in text_lower or "cert" in text_lower):
                score = 0.85
                
            predictions.append(LabelPrediction(
                label=label_name,
                taxonomy=taxonomy,
                score=score,
                threshold=threshold,
                selected=False
            ))
            
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        
        return PredictionResult(
            model_version=f"{self.version} (non-production)",
            backend="development",
            inference_duration_ms=duration_ms,
            predictions=predictions
        )
