"""Tests for ensemble defence (heterogeneous model ensemble)."""

import numpy as np
import pandas as pd
import os


def _make_dummy_data(n_samples=200, n_features=10, seed=42):
    """Create a small dummy dataset for testing."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.randint(0, 2, size=n_samples))
    return X, y


class TestEnsembleModel:
    """Tests for EnsembleModel (LR + XGBoost + MLP ensemble)."""

    def test_fit_and_predict_proba(self):
        """Ensemble trains all sub-models and returns valid probabilities."""
        from defences.ensemble import EnsembleModel

        X, y = _make_dummy_data()
        model = EnsembleModel({"epochs": 2, "hidden_dim": 8, "batch_size": 64})
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (200,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_exposes_pytorch_model(self):
        """EnsembleModel.model is an nn.Module for CAPGD compatibility."""
        import torch.nn as nn
        from defences.ensemble import EnsembleModel

        X, y = _make_dummy_data()
        model = EnsembleModel({"epochs": 2, "hidden_dim": 8, "batch_size": 64})
        model.fit(X, y)

        assert hasattr(model, "model")
        assert isinstance(model.model, nn.Module)
        assert hasattr(model, "device")
        assert hasattr(model, "_use_logits")

    def test_save_load_roundtrip(self, tmp_path):
        """Save/load preserves predictions."""
        from defences.ensemble import EnsembleModel

        X, y = _make_dummy_data()
        model = EnsembleModel({"epochs": 2, "hidden_dim": 8, "batch_size": 64})
        model.fit(X, y)

        path = str(tmp_path / "ensemble.joblib")
        model.save(path)
        assert os.path.exists(path)

        loaded = EnsembleModel.load(path)
        preds_original = model.predict_proba(X)
        preds_loaded = loaded.predict_proba(X)
        np.testing.assert_allclose(preds_original, preds_loaded, atol=1e-5)

    def test_load_returns_ensemble_instance(self, tmp_path):
        """Loaded model is an EnsembleModel."""
        from defences.ensemble import EnsembleModel

        X, y = _make_dummy_data()
        model = EnsembleModel({"epochs": 1, "hidden_dim": 8, "batch_size": 64})
        model.fit(X, y)

        path = str(tmp_path / "ensemble.joblib")
        model.save(path)

        loaded = EnsembleModel.load(path)
        assert isinstance(loaded, EnsembleModel)

    def test_soft_voting_averages_three_models(self):
        """Ensemble output is the average of three sub-model predictions."""
        from defences.ensemble import EnsembleModel

        X, y = _make_dummy_data(n_samples=50, n_features=5)
        model = EnsembleModel({"epochs": 2, "hidden_dim": 8, "batch_size": 32})
        model.fit(X, y)

        # Get individual sub-model predictions
        import torch
        lr_probs = model.lr_model.predict_proba(X)[:, 1]
        xgb_probs = model.xgb_model.predict_proba(X)[:, 1]

        model.mlp.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(model.device)
        with torch.no_grad():
            mlp_out = model.mlp(X_tensor)
            mlp_probs = torch.sigmoid(mlp_out).cpu().numpy().flatten()

        expected = (lr_probs + xgb_probs + mlp_probs) / 3.0
        actual = model.predict_proba(X)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_extends_base_model(self):
        """EnsembleModel is a BaseModel subclass."""
        from defences.ensemble import EnsembleModel
        from models.base import BaseModel

        assert issubclass(EnsembleModel, BaseModel)
