from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(frozen=True)
class IssueEvent:
    owner: str
    repo: str
    issue_number: int
    title: str
    body: Optional[str]
    action: str
    html_url: str

@dataclass(frozen=True)
class LabelPrediction:
    label: str
    taxonomy: str
    score: float
    threshold: float
    selected: bool

@dataclass(frozen=True)
class PredictionResult:
    model_version: str
    backend: str
    inference_duration_ms: float
    predictions: List[LabelPrediction] = field(default_factory=list)
