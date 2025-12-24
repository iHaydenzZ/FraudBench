"""Tests for attack functionality and projection."""
import pytest
import torch
import pandas as pd
import numpy as np
from constraints.schema import ConstraintSchema, FeatureConstraint
from attacks.capgd import project_constraints


class TestProjectConstraints:
    """Tests for constraint projection in attacks."""

    @pytest.fixture
    def simple_schema(self):
        """Create a simple schema for testing projection."""
        schema = ConstraintSchema()
        schema.features['feat_0'] = FeatureConstraint(
            name='feat_0',
            type='numeric',
            min_val=0.0,
            max_val=1.0
        )
        schema.features['feat_1'] = FeatureConstraint(
            name='feat_1',
            type='numeric',
            min_val=-1.0,
            max_val=1.0
        )
        return schema

    def test_projection_clips_to_bounds(self, simple_schema):
        """Test that projection clips values to schema bounds."""
        feature_names = ['feat_0', 'feat_1']
        feature_types = {'feat_0': 'numeric', 'feat_1': 'numeric'}

        # Create adversarial examples that exceed bounds
        x_adv = torch.tensor([[1.5, 2.0], [-0.5, -2.0]], dtype=torch.float32)
        x_orig = torch.tensor([[0.5, 0.0], [0.5, 0.0]], dtype=torch.float32)

        x_proj = project_constraints(x_adv, x_orig, simple_schema, feature_names, feature_types)

        # Check bounds are respected
        assert x_proj[0, 0].item() == 1.0  # Clipped from 1.5 to max 1.0
        assert x_proj[0, 1].item() == 1.0  # Clipped from 2.0 to max 1.0
        assert x_proj[1, 0].item() == 0.0  # Clipped from -0.5 to min 0.0
        assert x_proj[1, 1].item() == -1.0  # Clipped from -2.0 to min -1.0

    def test_projection_preserves_valid_values(self, simple_schema):
        """Test that valid values are not modified."""
        feature_names = ['feat_0', 'feat_1']
        feature_types = {'feat_0': 'numeric', 'feat_1': 'numeric'}

        x_adv = torch.tensor([[0.5, 0.0], [0.8, -0.5]], dtype=torch.float32)
        x_orig = torch.tensor([[0.5, 0.0], [0.8, -0.5]], dtype=torch.float32)

        x_proj = project_constraints(x_adv, x_orig, simple_schema, feature_names, feature_types)

        torch.testing.assert_close(x_proj, x_adv)

    def test_projection_handles_non_numeric(self):
        """Test that non-numeric features are reverted to original."""
        schema = ConstraintSchema()
        schema.features['num_feat'] = FeatureConstraint(
            name='num_feat',
            type='numeric',
            min_val=0.0,
            max_val=1.0
        )
        schema.features['cat_feat'] = FeatureConstraint(
            name='cat_feat',
            type='categorical',
            allowed_values=[0, 1, 2]
        )

        feature_names = ['num_feat', 'cat_feat']
        feature_types = {'num_feat': 'numeric', 'cat_feat': 'categorical'}

        x_adv = torch.tensor([[1.5, 0.7]], dtype=torch.float32)
        x_orig = torch.tensor([[0.5, 1.0]], dtype=torch.float32)

        x_proj = project_constraints(x_adv, x_orig, schema, feature_names, feature_types)

        assert x_proj[0, 0].item() == 1.0  # Clipped numeric
        assert x_proj[0, 1].item() == 1.0  # Reverted to original


class TestCAPGDAttack:
    """Integration tests for CAPGD attack."""

    def test_attack_returns_dataframe(self):
        """Test that attack returns a DataFrame with correct shape."""
        pytest.importorskip("torch")

        from models.neural import NeuralModel
        from attacks.capgd import capgd_attack

        # Create simple model and data
        X = pd.DataFrame(np.random.randn(50, 5), columns=[f'feat_{i}' for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 50))

        model = NeuralModel({'epochs': 1, 'hidden_dim': 8})
        model.fit(X, y)

        feature_types = {f'feat_{i}': 'numeric' for i in range(5)}
        schema = ConstraintSchema.from_data(X, feature_types)

        X_adv = capgd_attack(
            model, X, y, schema, feature_types,
            params={'epsilon': 0.1, 'steps': 2}
        )

        assert isinstance(X_adv, pd.DataFrame)
        assert X_adv.shape == X.shape
        assert list(X_adv.columns) == list(X.columns)
