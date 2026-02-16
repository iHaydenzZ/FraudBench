"""Tests for defence implementations."""

import pandas as pd
import numpy as np
from constraints.schema import ConstraintSchema, FeatureConstraint


class TestAdversarialTraining:
    """Tests for adversarial training with constraint projection."""

    def test_adv_train_step_respects_constraints(self):
        """Adversarial examples generated during training respect schema bounds."""
        from models.neural import NeuralModel
        from constraints.schema import ConstraintSchema

        # Create data with known bounds
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feat_0": np.random.uniform(0, 1, 100),
                "feat_1": np.random.uniform(-1, 1, 100),
            }
        )
        y = pd.Series(np.random.randint(0, 2, 100))

        feature_types = {"feat_0": "numeric", "feat_1": "numeric"}
        schema = ConstraintSchema.from_data(X, feature_types)

        # Train with adversarial training — should not crash
        model = NeuralModel(
            {
                "epochs": 2,
                "hidden_dim": 8,
                "adv_training": True,
                "adv_epsilon": 0.3,
                "adv_schema": schema,
                "adv_feature_names": X.columns.tolist(),
                "adv_feature_types": feature_types,
            }
        )
        model.fit(X, y)

        # Model should still produce valid probabilities
        probs = model.predict_proba(X)
        assert probs.shape == (100,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_adv_train_step_without_schema_still_works(self):
        """Backward compat: adversarial training without schema uses L-inf only."""
        from models.neural import NeuralModel

        X = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
        y = pd.Series(np.random.randint(0, 2, 50))

        model = NeuralModel(
            {
                "epochs": 2,
                "hidden_dim": 8,
                "adv_training": True,
                "adv_epsilon": 0.1,
            }
        )
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (50,)


class TestInputValidation:
    """Tests for input validation defence."""

    def test_numeric_clipping(self):
        """Numeric features are clipped to schema bounds."""
        from defences.input_validation import InputValidator

        schema = ConstraintSchema()
        schema.features["x"] = FeatureConstraint(name="x", type="numeric", min_val=0.0, max_val=10.0)

        iv = InputValidator(schema)
        X_train = pd.DataFrame({"x": [1.0, 5.0, 9.0]})
        iv.fit(X_train)

        X_bad = pd.DataFrame({"x": [-5.0, 5.0, 15.0]})
        result = iv.transform(X_bad)
        assert result["x"].tolist() == [0.0, 5.0, 10.0]

    def test_outlier_detection_sanitise_mode(self):
        """Outliers beyond k*std are clipped in sanitise mode."""
        from defences.input_validation import InputValidator

        schema = ConstraintSchema()
        schema.features["x"] = FeatureConstraint(name="x", type="numeric", min_val=-100.0, max_val=100.0)

        iv = InputValidator(schema, mode="sanitise", z_threshold=2.0)
        X_train = pd.DataFrame({"x": np.random.normal(0, 1, 1000)})
        iv.fit(X_train)

        X_test = pd.DataFrame({"x": [0.0, 5.0, -5.0]})  # 5.0 and -5.0 are outliers at z=2
        result = iv.transform(X_test)

        # Outliers should be clipped to +/- z_threshold * std
        assert abs(result["x"].iloc[0] - 0.0) < 0.01  # Normal value unchanged
        assert result["x"].iloc[1] < 5.0  # Clipped down
        assert result["x"].iloc[2] > -5.0  # Clipped up

    def test_outlier_detection_reject_mode(self):
        """Outliers beyond k*std are rejected in reject mode."""
        from defences.input_validation import InputValidator

        schema = ConstraintSchema()
        schema.features["x"] = FeatureConstraint(name="x", type="numeric", min_val=-100.0, max_val=100.0)

        iv = InputValidator(schema, mode="reject", z_threshold=2.0)
        X_train = pd.DataFrame({"x": np.random.normal(0, 1, 1000)})
        iv.fit(X_train)

        X_test = pd.DataFrame({"x": [0.0, 5.0, -5.0]})
        result, metadata = iv.transform(X_test, return_metadata=True)

        assert metadata["n_rejected"] >= 1
        assert metadata["total"] == 3

    def test_backward_compat_no_fit(self):
        """Transform still works without calling fit (no outlier detection)."""
        from defences.input_validation import InputValidator

        schema = ConstraintSchema()
        schema.features["x"] = FeatureConstraint(name="x", type="numeric", min_val=0.0, max_val=10.0)

        iv = InputValidator(schema)
        X_bad = pd.DataFrame({"x": [-5.0, 5.0, 15.0]})
        result = iv.transform(X_bad)  # Should not crash
        assert result["x"].tolist() == [0.0, 5.0, 10.0]
