"""Tests for ART-based black-box attacks."""
import pytest
import pandas as pd
import numpy as np
from constraints.schema import ConstraintSchema, FeatureConstraint

art = pytest.importorskip("art")


class TestProjectConstraintsNP:
    """Tests for NumPy constraint projection."""

    @pytest.fixture
    def simple_schema(self):
        schema = ConstraintSchema()
        schema.features["feat_0"] = FeatureConstraint(
            name="feat_0", type="numeric", min_val=0.0, max_val=1.0
        )
        schema.features["feat_1"] = FeatureConstraint(
            name="feat_1", type="numeric", min_val=-1.0, max_val=1.0
        )
        return schema

    def test_clips_to_bounds(self, simple_schema):
        from attacks.constraints_np import project_constraints_np

        feature_names = ["feat_0", "feat_1"]
        feature_types = {"feat_0": "numeric", "feat_1": "numeric"}

        x_adv = np.array([[1.5, 2.0], [-0.5, -2.0]], dtype=np.float32)
        x_orig = np.array([[0.5, 0.0], [0.5, 0.0]], dtype=np.float32)

        x_proj = project_constraints_np(
            x_adv, x_orig, simple_schema, feature_names, feature_types
        )

        assert x_proj[0, 0] == pytest.approx(1.0)
        assert x_proj[0, 1] == pytest.approx(1.0)
        assert x_proj[1, 0] == pytest.approx(0.0)
        assert x_proj[1, 1] == pytest.approx(-1.0)

    def test_preserves_valid_values(self, simple_schema):
        from attacks.constraints_np import project_constraints_np

        feature_names = ["feat_0", "feat_1"]
        feature_types = {"feat_0": "numeric", "feat_1": "numeric"}

        x_adv = np.array([[0.5, 0.0], [0.8, -0.5]], dtype=np.float32)
        x_orig = x_adv.copy()

        x_proj = project_constraints_np(
            x_adv, x_orig, simple_schema, feature_names, feature_types
        )

        np.testing.assert_array_almost_equal(x_proj, x_adv)

    def test_reverts_non_numeric(self):
        from attacks.constraints_np import project_constraints_np

        schema = ConstraintSchema()
        schema.features["num_feat"] = FeatureConstraint(
            name="num_feat", type="numeric", min_val=0.0, max_val=1.0
        )
        schema.features["cat_feat"] = FeatureConstraint(
            name="cat_feat", type="categorical", allowed_values=[0, 1, 2]
        )

        feature_names = ["num_feat", "cat_feat"]
        feature_types = {"num_feat": "numeric", "cat_feat": "categorical"}

        x_adv = np.array([[1.5, 0.7]], dtype=np.float32)
        x_orig = np.array([[0.5, 1.0]], dtype=np.float32)

        x_proj = project_constraints_np(
            x_adv, x_orig, schema, feature_names, feature_types
        )

        assert x_proj[0, 0] == pytest.approx(1.0)  # Clipped
        assert x_proj[0, 1] == pytest.approx(1.0)  # Reverted to original


class TestComputeClipValues:
    def test_returns_correct_shape(self):
        from attacks.constraints_np import compute_clip_values

        schema = ConstraintSchema()
        schema.features["a"] = FeatureConstraint(
            name="a", type="numeric", min_val=0.0, max_val=10.0
        )
        schema.features["b"] = FeatureConstraint(
            name="b", type="numeric", min_val=-5.0, max_val=5.0
        )

        mins, maxs = compute_clip_values(schema, ["a", "b"])

        assert mins.shape == (2,)
        assert maxs.shape == (2,)
        assert mins[0] == pytest.approx(0.0)
        assert maxs[0] == pytest.approx(10.0)
        assert mins[1] == pytest.approx(-5.0)
        assert maxs[1] == pytest.approx(5.0)


class TestParseNorm:
    def test_inf_string(self):
        from attacks.constraints_np import parse_norm

        assert parse_norm("inf") == np.inf
        assert parse_norm("Inf") == np.inf

    def test_numeric(self):
        from attacks.constraints_np import parse_norm

        assert parse_norm(2) == 2.0
        assert parse_norm(2.0) == 2.0


class TestARTWrapper:
    """Tests for the ART model wrapper."""

    def test_predict_returns_two_columns(self):
        from attacks.art_wrapper import ARTModelWrapper
        from models.neural import NeuralModel

        X = pd.DataFrame(np.random.randn(30, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 30))

        model = NeuralModel({"epochs": 1, "hidden_dim": 8})
        model.fit(X, y)

        wrapper = ARTModelWrapper(
            model,
            input_shape=(5,),
            feature_names=[f"f{i}" for i in range(5)],
        )

        preds = wrapper.predict(X.values.astype(np.float32))
        assert preds.shape == (30, 2)
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(preds.sum(axis=1), np.ones(30))

    def test_works_with_tree_model(self):
        from attacks.art_wrapper import ARTModelWrapper
        from models.tree import TreeModel

        X = pd.DataFrame(np.random.randn(50, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 50))

        model = TreeModel({"n_estimators": 10, "max_depth": 3})
        model.fit(X, y)

        wrapper = ARTModelWrapper(
            model,
            input_shape=(5,),
            feature_names=[f"f{i}" for i in range(5)],
        )

        preds = wrapper.predict(X.values.astype(np.float32))
        assert preds.shape == (50, 2)
        np.testing.assert_array_almost_equal(preds.sum(axis=1), np.ones(50))


@pytest.mark.slow
class TestSquareAttackIntegration:
    """Integration test for Square Attack on tree model."""

    def test_square_attack_returns_dataframe(self):
        from attacks.square import square_attack
        from models.tree import TreeModel

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(30, 5), columns=[f"feat_{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 30))

        model = TreeModel({"n_estimators": 10, "max_depth": 3})
        model.fit(X, y)

        feature_types = {f"feat_{i}": "numeric" for i in range(5)}
        schema = ConstraintSchema.from_data(X, feature_types)

        X_adv = square_attack(
            model, X, y, schema, feature_types,
            params={"epsilon": 0.5, "max_iter": 5}
        )

        assert isinstance(X_adv, pd.DataFrame)
        assert X_adv.shape == X.shape
        assert list(X_adv.columns) == list(X.columns)


@pytest.mark.slow
class TestHopSkipJumpIntegration:
    """Integration test for HopSkipJump on tree model."""

    def test_hopskipjump_returns_dataframe(self):
        from attacks.hopskipjump import hopskipjump_attack
        from models.tree import TreeModel

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(10, 5), columns=[f"feat_{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 10))

        model = TreeModel({"n_estimators": 10, "max_depth": 3})
        model.fit(X, y)

        feature_types = {f"feat_{i}": "numeric" for i in range(5)}
        schema = ConstraintSchema.from_data(X, feature_types)

        X_adv = hopskipjump_attack(
            model, X, y, schema, feature_types,
            params={"max_iter": 2, "max_eval": 100, "init_eval": 10}
        )

        assert isinstance(X_adv, pd.DataFrame)
        assert X_adv.shape == X.shape
        assert list(X_adv.columns) == list(X.columns)
