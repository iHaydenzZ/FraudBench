import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(y_true: pd.Series, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes fraud-detection metrics.

    Includes both PR-AUC and ROC-AUC: the ICDM benchmark contrasts them to show
    where ROC-AUC stays high while PR-AUC reveals rare-class collapse (spec §3.1,
    NB3 prauc_vs_rocauc). roc_auc is NaN when y_true has a single class (e.g. a
    degenerate split), which keeps a batch from crashing on that edge case.
    """
    y_pred = (y_probs >= threshold).astype(int)

    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        # roc_auc_score raises when only one class is present in y_true.
        roc_auc = float("nan")

    return {
        "pr_auc": average_precision_score(y_true, y_probs),
        "roc_auc": roc_auc,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
