"""Tests for evaluation metrics, focusing on the ROC-AUC addition (spec §3.1)."""

import numpy as np
import pandas as pd

from evaluation.metrics import compute_metrics


class TestComputeMetrics:
    def test_includes_roc_auc(self):
        y_true = pd.Series([0, 0, 1, 1])
        y_probs = np.array([0.1, 0.4, 0.35, 0.8])
        m = compute_metrics(y_true, y_probs)
        assert "roc_auc" in m
        assert "pr_auc" in m
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_perfect_separation_roc_auc_is_one(self):
        y_true = pd.Series([0, 0, 1, 1])
        y_probs = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_metrics(y_true, y_probs)["roc_auc"] == 1.0

    def test_single_class_roc_auc_is_nan_not_crash(self):
        # A degenerate batch (all negatives) must not crash the run.
        y_true = pd.Series([0, 0, 0])
        y_probs = np.array([0.1, 0.2, 0.3])
        m = compute_metrics(y_true, y_probs)
        assert np.isnan(m["roc_auc"])
