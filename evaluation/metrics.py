import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score, f1_score


def compute_metrics(y_true: pd.Series, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes fraud-detection metrics."""
    y_pred = (y_probs >= threshold).astype(int)

    return {
        "pr_auc": average_precision_score(y_true, y_probs),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
