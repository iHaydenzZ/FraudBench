"""Tests for constraint schema and validation."""

import pytest
import pandas as pd
import numpy as np
from constraints.schema import ConstraintSchema, FeatureConstraint
from constraints.validator import ConstraintValidator


class TestConstraintSchema:
    """Tests for ConstraintSchema functionality."""

    def test_schema_from_numeric_data(self):
        """Test schema inference from numeric data."""
        X = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0, 4.0, 5.0], "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0]})
        feature_types = {"feat_a": "numeric", "feat_b": "numeric"}

        schema = ConstraintSchema.from_data(X, feature_types)

        assert "feat_a" in schema.features
        assert "feat_b" in schema.features
        assert schema.features["feat_a"].min_val == 1.0
        assert schema.features["feat_a"].max_val == 5.0
        assert schema.features["feat_b"].min_val == 10.0
        assert schema.features["feat_b"].max_val == 50.0

    def test_schema_non_negative_detection(self):
        """Test that non-negative features are flagged."""
        X = pd.DataFrame({"positive": [0.0, 1.0, 2.0], "mixed": [-1.0, 0.0, 1.0]})
        feature_types = {"positive": "numeric", "mixed": "numeric"}

        schema = ConstraintSchema.from_data(X, feature_types)

        assert schema.features["positive"].is_non_negative is True
        assert schema.features["mixed"].is_non_negative is False

    def test_schema_categorical_values(self):
        """Test schema inference for categorical features."""
        X = pd.DataFrame({"category": ["A", "B", "C", "A", "B"]})
        feature_types = {"category": "categorical"}

        schema = ConstraintSchema.from_data(X, feature_types)

        assert schema.features["category"].type == "categorical"
        assert set(schema.features["category"].allowed_values) == {"A", "B", "C"}

    def test_schema_handles_nan_in_numeric(self):
        """Test that schema handles NaN values in numeric columns."""
        X = pd.DataFrame(
            {"with_nan": [1.0, np.nan, 3.0, np.nan, 5.0], "all_nan": [np.nan, np.nan, np.nan, np.nan, np.nan]}
        )
        feature_types = {"with_nan": "numeric", "all_nan": "numeric"}

        schema = ConstraintSchema.from_data(X, feature_types)

        # Should use non-NaN values for bounds
        assert schema.features["with_nan"].min_val == 1.0
        assert schema.features["with_nan"].max_val == 5.0
        assert schema.features["with_nan"].has_missing

        # All-NaN column should have default bounds
        assert schema.features["all_nan"].min_val == 0.0
        assert schema.features["all_nan"].max_val == 1.0
        assert schema.features["all_nan"].has_missing

    def test_schema_handles_mixed_types_categorical(self):
        """Test that schema handles mixed types in categorical columns."""
        X = pd.DataFrame({"mixed": ["A", 1, "B", 2.5, None]})
        feature_types = {"mixed": "categorical"}

        # Should not raise an error
        schema = ConstraintSchema.from_data(X, feature_types)

        assert schema.features["mixed"].type == "categorical"
        # NaN should be tracked via has_missing, not in allowed_values
        assert schema.features["mixed"].has_missing
        assert None not in schema.features["mixed"].allowed_values

    def test_schema_handles_inf_values(self):
        """Test that schema handles infinite values."""
        X = pd.DataFrame({"with_inf": [1.0, np.inf, 3.0, -np.inf, 5.0]})
        feature_types = {"with_inf": "numeric"}

        schema = ConstraintSchema.from_data(X, feature_types)

        # Inf values should be clamped
        assert schema.features["with_inf"].min_val == -1e10
        assert schema.features["with_inf"].max_val == 1e10

    def test_schema_handles_constant_column(self):
        """Test that schema handles constant columns (min == max)."""
        X = pd.DataFrame({"constant": [5.0, 5.0, 5.0, 5.0]})
        feature_types = {"constant": "numeric"}

        schema = ConstraintSchema.from_data(X, feature_types)

        # Should add buffer to avoid min == max
        assert schema.features["constant"].min_val == 5.0
        assert schema.features["constant"].max_val == 6.0


class TestConstraintValidator:
    """Tests for ConstraintValidator functionality."""

    @pytest.fixture
    def simple_schema(self):
        """Create a simple schema for testing."""
        schema = ConstraintSchema()
        schema.features["feat_a"] = FeatureConstraint(name="feat_a", type="numeric", min_val=0.0, max_val=10.0)
        schema.features["feat_b"] = FeatureConstraint(name="feat_b", type="numeric", min_val=-5.0, max_val=5.0)
        return schema

    def test_valid_samples_pass(self, simple_schema):
        """Test that valid samples pass validation."""
        validator = ConstraintValidator(simple_schema)

        X = pd.DataFrame({"feat_a": [0.0, 5.0, 10.0], "feat_b": [-5.0, 0.0, 5.0]})

        validity_rate = validator.validate(X)
        assert validity_rate == 1.0

    def test_invalid_samples_fail(self, simple_schema):
        """Test that invalid samples fail validation."""
        validator = ConstraintValidator(simple_schema)

        X = pd.DataFrame(
            {
                "feat_a": [0.0, 15.0, 5.0],  # 15.0 exceeds max
                "feat_b": [-5.0, 0.0, 5.0],
            }
        )

        validity_rate = validator.validate(X)
        assert validity_rate == pytest.approx(2 / 3, rel=0.01)

    def test_boundary_values(self, simple_schema):
        """Test that boundary values are valid."""
        validator = ConstraintValidator(simple_schema)

        X = pd.DataFrame(
            {
                "feat_a": [0.0, 10.0],  # Exact boundaries
                "feat_b": [-5.0, 5.0],
            }
        )

        validity_rate = validator.validate(X)
        assert validity_rate == 1.0

    def test_empty_dataframe(self, simple_schema):
        """Test validation of empty dataframe."""
        validator = ConstraintValidator(simple_schema)

        X = pd.DataFrame({"feat_a": [], "feat_b": []})

        validity_rate = validator.validate(X)
        assert validity_rate == 1.0

    def test_nan_categorical_with_has_missing_passes(self):
        """Test that NaN in a categorical column passes when has_missing=True."""
        schema = ConstraintSchema()
        schema.features["cat"] = FeatureConstraint(
            name="cat", type="categorical", allowed_values=["A", "B", "C"], has_missing=True
        )
        validator = ConstraintValidator(schema)

        sample = pd.Series({"cat": np.nan})
        assert validator.validate_sample(sample) is True

    def test_nan_categorical_without_has_missing_fails(self):
        """Test that NaN in a categorical column fails when has_missing=False."""
        schema = ConstraintSchema()
        schema.features["cat"] = FeatureConstraint(
            name="cat", type="categorical", allowed_values=["A", "B", "C"], has_missing=False
        )
        validator = ConstraintValidator(schema)

        sample = pd.Series({"cat": np.nan})
        assert validator.validate_sample(sample) is False

    def test_nan_numeric_with_has_missing_passes(self):
        """Test that NaN in a numeric column passes when has_missing=True."""
        schema = ConstraintSchema()
        schema.features["num"] = FeatureConstraint(
            name="num", type="numeric", min_val=0.0, max_val=10.0, has_missing=True
        )
        validator = ConstraintValidator(schema)

        sample = pd.Series({"num": np.nan})
        assert validator.validate_sample(sample) is True
