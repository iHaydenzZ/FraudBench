"""Tests for model checkpoint save/load functionality."""
import pytest
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


class TestNeuralModelCheckpoint:
    """NeuralModel save/load round-trip tests."""

    def test_save_creates_file(self, tmp_path):
        from models.neural import NeuralModel

        X, y = _make_dummy_data()
        model = NeuralModel({"epochs": 2, "hidden_dim": 16, "batch_size": 64})
        model.fit(X, y)

        path = str(tmp_path / "neural.pt")
        model.save(path)
        assert os.path.exists(path)

    def test_load_restores_predictions(self, tmp_path):
        from models.neural import NeuralModel

        X, y = _make_dummy_data()
        model = NeuralModel({"epochs": 2, "hidden_dim": 16, "batch_size": 64})
        model.fit(X, y)

        path = str(tmp_path / "neural.pt")
        model.save(path)

        loaded = NeuralModel.load(path)
        preds_original = model.predict_proba(X)
        preds_loaded = loaded.predict_proba(X)
        np.testing.assert_allclose(preds_original, preds_loaded, atol=1e-6)

    def test_load_returns_neural_model_instance(self, tmp_path):
        from models.neural import NeuralModel

        X, y = _make_dummy_data()
        model = NeuralModel({"epochs": 1, "hidden_dim": 16, "batch_size": 64})
        model.fit(X, y)

        path = str(tmp_path / "neural.pt")
        model.save(path)

        loaded = NeuralModel.load(path)
        assert isinstance(loaded, NeuralModel)


class TestTreeModelCheckpoint:
    """TreeModel save/load round-trip tests."""

    def test_save_creates_file(self, tmp_path):
        from models.tree import TreeModel

        X, y = _make_dummy_data()
        model = TreeModel({"n_estimators": 10, "max_depth": 2})
        model.fit(X, y)

        path = str(tmp_path / "tree.joblib")
        model.save(path)
        assert os.path.exists(path)

    def test_load_restores_predictions(self, tmp_path):
        from models.tree import TreeModel

        X, y = _make_dummy_data()
        model = TreeModel({"n_estimators": 10, "max_depth": 2})
        model.fit(X, y)

        path = str(tmp_path / "tree.joblib")
        model.save(path)

        loaded = TreeModel.load(path)
        preds_original = model.predict_proba(X)
        preds_loaded = loaded.predict_proba(X)
        np.testing.assert_allclose(preds_original, preds_loaded, atol=1e-6)

    def test_load_returns_tree_model_instance(self, tmp_path):
        from models.tree import TreeModel

        X, y = _make_dummy_data()
        model = TreeModel({"n_estimators": 10, "max_depth": 2})
        model.fit(X, y)

        path = str(tmp_path / "tree.joblib")
        model.save(path)

        loaded = TreeModel.load(path)
        assert isinstance(loaded, TreeModel)

    def test_params_preserved_after_load(self, tmp_path):
        """Original params should be preserved in the loaded model."""
        from models.tree import TreeModel

        X, y = _make_dummy_data()
        params = {"n_estimators": 10, "max_depth": 2}
        model = TreeModel(params)
        model.fit(X, y)

        path = str(tmp_path / "tree.joblib")
        model.save(path)

        loaded = TreeModel.load(path)
        assert loaded.params.get("n_estimators") == 10
        assert loaded.params.get("max_depth") == 2
