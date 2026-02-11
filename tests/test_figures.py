"""Tests for figure generation utilities."""
import pytest
import pandas as pd
import numpy as np
import os


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

        df = pd.DataFrame({
            "dataset": ["ccfd"] * 3,
            "model_type": ["neural"] * 3,
            "defence_type": ["none"] * 3,
            "attack_type": ["capgd"] * 3,
            "attack_epsilon": [0.1] * 3,
            "clean_pr_auc": [0.80, 0.85, 0.90],
            "robust_pr_auc": [0.60, 0.65, 0.70],
        })

        agg = aggregate_seeds(df)
        assert len(agg) == 1
        assert agg["clean_pr_auc_mean"].iloc[0] == pytest.approx(0.85, abs=1e-6)
        assert agg["robust_pr_auc_mean"].iloc[0] == pytest.approx(0.65, abs=1e-6)
        # Std should be > 0 for non-constant values
        assert agg["clean_pr_auc_std"].iloc[0] > 0

    def test_groups_correctly(self):
        from scripts.generate_figures import aggregate_seeds

        df = pd.DataFrame({
            "dataset": ["ccfd", "ccfd", "ieee_cis", "ieee_cis"],
            "model_type": ["neural"] * 4,
            "defence_type": ["none"] * 4,
            "attack_type": ["capgd"] * 4,
            "attack_epsilon": [0.1] * 4,
            "clean_pr_auc": [0.80, 0.90, 0.70, 0.75],
            "robust_pr_auc": [0.60, 0.65, 0.50, 0.55],
        })

        agg = aggregate_seeds(df)
        assert len(agg) == 2  # Two dataset groups
