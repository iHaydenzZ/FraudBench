"""Tests for figure generation utilities."""

import pytest
import pandas as pd
import numpy as np


class TestLoadRegistry:
    """Tests for load_registry type conversion."""

    def test_numeric_columns_converted(self, tmp_path):
        from scripts.generate_figures import load_registry

        csv_path = tmp_path / "registry.csv"
        csv_path.write_text(
            "timestamp,experiment_name,seed,dataset,model_type,defence_type,"
            "attack_type,attack_epsilon,validity_rate,adv_validity_rate,"
            "clean_pr_auc,clean_precision,clean_recall,clean_f1,"
            "robust_pr_auc,robust_precision,robust_recall,robust_f1,"
            "clean_accuracy,robust_accuracy,train_time_sec,attack_time_sec\n"
            "2025-01-01T00:00:00,test,42,ccfd,neural,none,capgd,0.1,1.0000,0.9500,"
            "0.8500,0.8000,0.7500,0.7700,0.6500,0.6000,0.5500,0.5700,"
            "0.9000,0.8000,10.50,5.25\n"
        )

        df = load_registry(str(csv_path))
        assert df["seed"].dtype in [np.int64, np.float64]
        assert df["clean_pr_auc"].dtype == np.float64
        assert df["attack_epsilon"].dtype == np.float64
        assert len(df) == 1

    def test_handles_na_values(self, tmp_path):
        from scripts.generate_figures import load_registry

        csv_path = tmp_path / "registry.csv"
        csv_path.write_text(
            "timestamp,experiment_name,seed,dataset,model_type,defence_type,"
            "attack_type,attack_epsilon,validity_rate,adv_validity_rate,"
            "clean_pr_auc,clean_precision,clean_recall,clean_f1,"
            "robust_pr_auc,robust_precision,robust_recall,robust_f1,"
            "clean_accuracy,robust_accuracy,train_time_sec,attack_time_sec\n"
            "2025-01-01T00:00:00,test,42,ccfd,tree,none,none,0.0,1.0000,n/a,"
            "0.9000,0.8500,0.8000,0.8200,0.0000,0.0000,0.0000,0.0000,"
            "0.9500,0.0000,n/a,n/a\n"
        )

        df = load_registry(str(csv_path))
        assert pd.isna(df["adv_validity_rate"].iloc[0])
        assert pd.isna(df["train_time_sec"].iloc[0])


class TestAggregateSeeds:
    """Tests for aggregate_seeds mean/std computation."""

    def test_mean_and_std(self):
        from scripts.generate_figures import aggregate_seeds

        df = pd.DataFrame(
            {
                "dataset": ["ccfd"] * 3,
                "model_type": ["neural"] * 3,
                "defence_type": ["none"] * 3,
                "attack_type": ["capgd"] * 3,
                "attack_epsilon": [0.1] * 3,
                "clean_pr_auc": [0.80, 0.85, 0.90],
                "robust_pr_auc": [0.60, 0.65, 0.70],
            }
        )

        agg = aggregate_seeds(df)
        assert len(agg) == 1
        assert agg["clean_pr_auc_mean"].iloc[0] == pytest.approx(0.85, abs=1e-6)
        assert agg["robust_pr_auc_mean"].iloc[0] == pytest.approx(0.65, abs=1e-6)
        # Std should be > 0 for non-constant values
        assert agg["clean_pr_auc_std"].iloc[0] > 0

    def test_groups_correctly(self):
        from scripts.generate_figures import aggregate_seeds

        df = pd.DataFrame(
            {
                "dataset": ["ccfd", "ccfd", "ieee_cis", "ieee_cis"],
                "model_type": ["neural"] * 4,
                "defence_type": ["none"] * 4,
                "attack_type": ["capgd"] * 4,
                "attack_epsilon": [0.1] * 4,
                "clean_pr_auc": [0.80, 0.90, 0.70, 0.75],
                "robust_pr_auc": [0.60, 0.65, 0.50, 0.55],
            }
        )

        agg = aggregate_seeds(df)
        assert len(agg) == 2  # Two dataset groups


class TestStatisticalTests:
    """Tests for pairwise defence statistical comparisons."""

    def _make_registry(
        self,
        seeds=(42, 123, 456),
        robust_none=(0.60, 0.65, 0.70),
        robust_adv=(0.80, 0.82, 0.84),
        robust_iv=(0.55, 0.58, 0.53),
        robust_ens=(0.75, 0.77, 0.79),
    ):
        """Build a minimal registry DataFrame for testing."""
        rows = []
        for defence, values in [
            ("none", robust_none),
            ("adversarial_training", robust_adv),
            ("input_validation", robust_iv),
            ("ensemble", robust_ens),
        ]:
            for seed, val in zip(seeds, values):
                rows.append(
                    {
                        "timestamp": "2025-01-01T00:00:00",
                        "experiment_name": "test",
                        "seed": seed,
                        "dataset": "ccfd",
                        "model_type": "neural",
                        "defence_type": defence,
                        "attack_type": "capgd",
                        "attack_epsilon": 0.1,
                        "robust_pr_auc": val,
                    }
                )
        return pd.DataFrame(rows)

    def test_pairwise_comparison_significant(self):
        """Test that a known large difference is detected as significant."""
        from scripts.statistical_tests import pairwise_defence_tests

        df = self._make_registry(
            robust_none=(0.60, 0.65, 0.70),
            robust_adv=(0.90, 0.92, 0.94),
            robust_iv=(0.55, 0.58, 0.53),
        )
        results = pairwise_defence_tests(df)
        assert len(results) == 6  # six pairwise comparisons

        # none vs adversarial_training should be significant (large gap)
        row_na = results[(results["defence_a"] == "none") & (results["defence_b"] == "adversarial_training")].iloc[0]
        assert row_na["significant"] == True  # noqa: E712
        assert row_na["p_value"] < 0.05
        assert row_na["mean_diff"] < 0  # none < adv_training
        assert row_na["cohens_d"] != 0.0

    def test_pairwise_comparison_not_significant(self):
        """Test that nearly identical values are NOT significant."""
        from scripts.statistical_tests import pairwise_defence_tests

        df = self._make_registry(
            robust_none=(0.700, 0.701, 0.699),
            robust_adv=(0.701, 0.700, 0.700),
            robust_iv=(0.699, 0.700, 0.701),
            robust_ens=(0.700, 0.701, 0.700),
        )
        results = pairwise_defence_tests(df)
        # No comparison should be significant for near-identical values
        for _, row in results.iterrows():
            if row["note"] == "":
                assert row["significant"] == False  # noqa: E712

    def test_identical_values_handled(self):
        """Test that identical robust_pr_auc across defences is handled."""
        from scripts.statistical_tests import pairwise_defence_tests

        df = self._make_registry(
            robust_none=(0.80, 0.80, 0.80),
            robust_adv=(0.80, 0.80, 0.80),
            robust_iv=(0.80, 0.80, 0.80),
            robust_ens=(0.80, 0.80, 0.80),
        )
        results = pairwise_defence_tests(df)
        for _, row in results.iterrows():
            assert row["significant"] == False  # noqa: E712
            assert row["note"] == "identical values across seeds"

    def test_missing_defence_skipped(self):
        """Test graceful handling when one defence has no data."""
        from scripts.statistical_tests import pairwise_defence_tests

        # Only 'none' and 'input_validation', no 'adversarial_training'
        rows = []
        for defence, values in [("none", (0.60, 0.65, 0.70)), ("input_validation", (0.55, 0.58, 0.53))]:
            for seed, val in zip((42, 123, 456), values):
                rows.append(
                    {
                        "seed": seed,
                        "dataset": "ccfd",
                        "model_type": "tree",
                        "defence_type": defence,
                        "robust_pr_auc": val,
                    }
                )
        df = pd.DataFrame(rows)
        results = pairwise_defence_tests(df)

        # Comparisons involving adversarial_training or ensemble should be skipped
        missing_rows = results[
            (results["defence_a"].isin(["adversarial_training", "ensemble"]))
            | (results["defence_b"].isin(["adversarial_training", "ensemble"]))
        ]
        for _, row in missing_rows.iterrows():
            assert "insufficient" in row["note"]

    def test_cohens_d_computation(self):
        """Test Cohen's d with known values."""
        from scripts.statistical_tests import compute_cohens_d

        # Two groups with mean diff = 1 and pooled std = 1 -> d = 1.0
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([0.0, 1.0, 2.0])
        d = compute_cohens_d(g1, g2)
        assert d == pytest.approx(1.0, abs=1e-6)

        # Identical groups -> d = 0
        d_zero = compute_cohens_d(g1, g1)
        assert d_zero == 0.0

    def test_output_csv(self, tmp_path):
        """Test that results CSV is written correctly."""
        from scripts.statistical_tests import pairwise_defence_tests

        df = self._make_registry()
        results = pairwise_defence_tests(df)
        csv_path = tmp_path / "statistical_tests.csv"
        results.to_csv(str(csv_path), index=False)

        loaded = pd.read_csv(str(csv_path))
        assert len(loaded) == len(results)
        assert "p_value" in loaded.columns
        assert "cohens_d" in loaded.columns
        assert "significant" in loaded.columns
        assert "w_statistic" in loaded.columns
        assert "w_p_value" in loaded.columns
