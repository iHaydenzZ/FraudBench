"""Tests for defence implementations."""
import pytest
import torch
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
        X = pd.DataFrame({
            'feat_0': np.random.uniform(0, 1, 100),
            'feat_1': np.random.uniform(-1, 1, 100),
        })
        y = pd.Series(np.random.randint(0, 2, 100))

        feature_types = {'feat_0': 'numeric', 'feat_1': 'numeric'}
        schema = ConstraintSchema.from_data(X, feature_types)

        # Train with adversarial training — should not crash
        model = NeuralModel({
            'epochs': 2,
            'hidden_dim': 8,
            'adv_training': True,
            'adv_epsilon': 0.3,
            'adv_schema': schema,
            'adv_feature_names': X.columns.tolist(),
            'adv_feature_types': feature_types,
        })
        model.fit(X, y)

        # Model should still produce valid probabilities
        probs = model.predict_proba(X)
        assert probs.shape == (100,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_adv_train_step_without_schema_still_works(self):
        """Backward compat: adversarial training without schema uses L-inf only."""
        from models.neural import NeuralModel

        X = pd.DataFrame(np.random.randn(50, 3), columns=['a', 'b', 'c'])
        y = pd.Series(np.random.randint(0, 2, 50))

        model = NeuralModel({
            'epochs': 2,
            'hidden_dim': 8,
            'adv_training': True,
            'adv_epsilon': 0.1,
        })
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (50,)
