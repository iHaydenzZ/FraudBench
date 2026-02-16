from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any


class BaseModel(ABC):
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns probability of positive class (fraud)."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """Load model checkpoint from disk. Returns a new instance."""
        pass

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Simple evaluation returning basic metrics."""
        from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score

        y_probs = self.predict_proba(X)
        y_pred = (y_probs >= 0.5).astype(int)

        return {
            "pr_auc": average_precision_score(y, y_probs),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "accuracy": accuracy_score(y, y_pred),
        }
