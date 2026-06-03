"""Tests for the reusable constraint-aware CAPGD (Protocols C1/C2, spec §4).

A tiny linear torch model stands in for NeuralModel so the tests are fast and
deterministic without training. Real-data anchor validation happens in Colab.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from attacks.constrained_capgd import (
    Projection,
    capgd_attack_constrained,
    project_g1_tensor,
    project_ohe_block_tensor,
)
from constraints.schema import ConstraintSchema


class _StubModel:
    """Minimal model exposing the surface CAPGD needs: .model, .device, logits."""

    def __init__(self, n_features):
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(n_features, 1))
        self.device = torch.device("cpu")
        self._use_logits = True


def _wide_schema(columns):
    # Loose bounds so per-step schema clipping does not interfere with the test.
    ref = pd.DataFrame({c: [-10.0, 10.0] for c in columns})
    return ConstraintSchema.from_data(ref, {c: "numeric" for c in columns})


class TestProjectionOperators:
    def test_ohe_block_snaps_to_one_hot(self):
        x = torch.tensor([[0.2, 0.9, 0.1], [0.7, 0.1, 0.6]])
        out = project_ohe_block_tensor(x.clone(), [0, 1, 2])
        assert torch.allclose(out.sum(dim=1), torch.ones(2))
        assert torch.allclose(out.max(dim=1).values, torch.ones(2))
        assert out[0].argmax().item() == 1
        assert out[1].argmax().item() == 0

    def test_g1_derives_installment(self):
        # Identity scaling (mean 0, scale 1) so processed == raw.
        g1_info = {
            "idx_loan": 0,
            "mean_loan": 0.0,
            "scale_loan": 1.0,
            "idx_rate": 1,
            "mean_rate": 0.0,
            "scale_rate": 1.0,
            "idx_inst": 2,
            "mean_inst": 0.0,
            "scale_inst": 1.0,
        }
        term_info = {"indices": [3, 4], "values": [36.0, 60.0]}
        x = torch.tensor([[10000.0, 12.0, 0.0, 1.0, 0.0]])
        out = project_g1_tensor(x.clone(), g1_info, term_info)
        r = 12.0 / 12.0 / 100.0
        factor = (1.0 + r) ** 36.0
        expected = 10000.0 * r * factor / (factor - 1.0)
        assert out[0, 2].item() == pytest.approx(expected, rel=1e-4)


class TestConstrainedAttack:
    def test_mask_freezes_immutable_columns_exactly(self):
        cols = ["a", "b", "c", "d"]
        X = pd.DataFrame(np.array([[0.1, 0.2, 0.3, 0.4], [0.5, -0.5, 0.0, 0.2]]), columns=cols)
        y = pd.Series([1, 0])
        schema = _wide_schema(cols)
        mask = np.array([True, False, True, False])  # b, d immutable

        adv = capgd_attack_constrained(
            _StubModel(4),
            X,
            y,
            schema,
            {c: "numeric" for c in cols},
            projections=[],
            mutable_mask=mask,
            params={"epsilon": 0.1, "steps": 5},
        )
        # Immutable columns restored exactly from the clean float64 frame.
        assert np.array_equal(adv["b"].values, X["b"].values)
        assert np.array_equal(adv["d"].values, X["d"].values)
        # Mutable columns were perturbed.
        assert not np.allclose(adv["a"].values, X["a"].values)

    def test_projection_is_enforced_on_output(self):
        cols = ["o0", "o1", "x"]
        X = pd.DataFrame(np.array([[0.6, 0.4, 0.1], [0.3, 0.7, -0.2]]), columns=cols)
        y = pd.Series([1, 0])
        schema = _wide_schema(cols)
        snap = Projection(tensor_fn=lambda t: project_ohe_block_tensor(t, [0, 1]))

        adv = capgd_attack_constrained(
            _StubModel(3),
            X,
            y,
            schema,
            {c: "numeric" for c in cols},
            projections=[snap],
            params={"epsilon": 0.2, "steps": 5},
        )
        block = adv[["o0", "o1"]].values
        assert np.allclose(block.sum(axis=1), 1.0)
        assert np.all((np.isclose(block, 0.0)) | (np.isclose(block, 1.0)))

    def test_empty_projection_no_mask_stays_in_eps_ball(self):
        cols = ["a", "b"]
        X = pd.DataFrame(np.array([[0.0, 0.0], [1.0, -1.0]]), columns=cols)
        y = pd.Series([1, 0])
        schema = _wide_schema(cols)

        adv = capgd_attack_constrained(
            _StubModel(2),
            X,
            y,
            schema,
            {c: "numeric" for c in cols},
            params={"epsilon": 0.1, "steps": 10},
        )
        assert adv.shape == X.shape
        assert np.all(np.abs(adv.values - X.values) <= 0.1 + 1e-5)
