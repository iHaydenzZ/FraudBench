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

    def test_wilcoxon_computed_with_six_seeds(self):
        """Wilcoxon signed-rank test is computed when n_paired >= 6."""
        from scripts.statistical_tests import pairwise_defence_tests

        seeds = (1, 2, 3, 4, 5, 6)
        df = self._make_registry(
            seeds=seeds,
            robust_none=(0.50, 0.52, 0.54, 0.56, 0.58, 0.60),
            robust_adv=(0.80, 0.82, 0.84, 0.86, 0.88, 0.90),
            robust_iv=(0.45, 0.47, 0.49, 0.51, 0.53, 0.55),
            robust_ens=(0.70, 0.72, 0.74, 0.76, 0.78, 0.80),
        )
        results = pairwise_defence_tests(df)

        # none vs adversarial_training should have Wilcoxon results
        row_na = results[(results["defence_a"] == "none") & (results["defence_b"] == "adversarial_training")].iloc[0]
        assert not np.isnan(row_na["w_statistic"])
        assert not np.isnan(row_na["w_p_value"])
        assert row_na["w_p_value"] < 0.05  # large gap → significant

    def test_wilcoxon_nan_with_few_seeds(self):
        """Wilcoxon is NaN when n_paired < 6."""
        from scripts.statistical_tests import pairwise_defence_tests

        df = self._make_registry(
            robust_none=(0.50, 0.55, 0.60),
            robust_adv=(0.80, 0.85, 0.90),
        )
        results = pairwise_defence_tests(df)
        row_na = results[(results["defence_a"] == "none") & (results["defence_b"] == "adversarial_training")].iloc[0]
        assert np.isnan(row_na["w_statistic"])
        assert np.isnan(row_na["w_p_value"])
        # t-test should still work
        assert row_na["significant"] == True  # noqa: E712

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


class TestPlotRobustnessCurves:
    """Tests for robustness curves figure generation."""

    def _make_multi_defence_data(self):
        """Registry with baseline at multiple epsilons + single-epsilon defences."""
        rows = []
        # Baseline (none) with multi-epsilon sweep
        for eps in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]:
            for seed in [42, 123, 456]:
                rows.append({
                    "dataset": "ccfd", "model_type": "neural",
                    "defence_type": "none", "attack_type": "capgd",
                    "attack_epsilon": eps, "seed": seed,
                    "robust_pr_auc": max(0.01, 0.85 - eps * 2),
                    "clean_pr_auc": 0.90,
                })
        # adversarial_training at eps=0.1 only
        for seed in [42, 123, 456]:
            rows.append({
                "dataset": "ccfd", "model_type": "neural",
                "defence_type": "adversarial_training", "attack_type": "capgd",
                "attack_epsilon": 0.1, "seed": seed,
                "robust_pr_auc": 0.75,
                "clean_pr_auc": 0.88,
            })
        # input_validation at eps=0.1 only
        for seed in [42, 123, 456]:
            rows.append({
                "dataset": "ccfd", "model_type": "neural",
                "defence_type": "input_validation", "attack_type": "capgd",
                "attack_epsilon": 0.1, "seed": seed,
                "robust_pr_auc": 0.55,
                "clean_pr_auc": 0.89,
            })
        return pd.DataFrame(rows)

    def test_single_epsilon_defences_included(self, tmp_path, monkeypatch):
        """Defences with only 1 epsilon value should appear as scatter markers."""
        import matplotlib.pyplot as plt
        from scripts.generate_figures import plot_robustness_curves

        df = self._make_multi_defence_data()

        # Prevent plt.close so we can inspect the figure
        monkeypatch.setattr(plt, "close", lambda *a, **kw: None)
        plot_robustness_curves(df, str(tmp_path))
        assert (tmp_path / "robustness_curves.png").exists()

        fig = plt.gcf()
        ax = fig.axes[0]

        # Collect legend labels — should include all 3 defences
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("none" in l for l in legend_labels), f"Missing 'none' in {legend_labels}"
        assert any("adversarial_training" in l for l in legend_labels), f"Missing 'adversarial_training' in {legend_labels}"
        assert any("input_validation" in l for l in legend_labels), f"Missing 'input_validation' in {legend_labels}"

        # Should have at least 1 line (none baseline) and 2 scatter collections
        lines = ax.get_lines()
        collections = [c for c in ax.collections if hasattr(c, 'get_offsets') and len(c.get_offsets()) > 0]
        assert len(lines) >= 1, "Expected at least 1 line for multi-epsilon defence"
        assert len(collections) >= 2, f"Expected at least 2 scatter collections, got {len(collections)}"
        plt.close("all")

    def test_neural_only_filter(self, tmp_path, monkeypatch):
        """Robustness curves should filter to neural models only (CAPGD irrelevant for trees)."""
        import matplotlib.pyplot as plt
        from scripts.generate_figures import plot_robustness_curves

        df = self._make_multi_defence_data()
        for eps in [0.01, 0.05, 0.1]:
            for seed in [42, 123, 456]:
                df = pd.concat([df, pd.DataFrame([{
                    "dataset": "ccfd", "model_type": "tree",
                    "defence_type": "none", "attack_type": "capgd",
                    "attack_epsilon": eps, "seed": seed,
                    "robust_pr_auc": 0.86,
                    "clean_pr_auc": 0.86,
                }])], ignore_index=True)

        monkeypatch.setattr(plt, "close", lambda *a, **kw: None)
        plot_robustness_curves(df, str(tmp_path))
        assert (tmp_path / "robustness_curves.png").exists()

        fig = plt.gcf()
        assert "Neural" in fig.texts[0].get_text()
        plt.close("all")


class TestPlotRobustnessBars:
    """Tests for robustness bars figure generation."""

    def test_filters_to_canonical_epsilon(self, tmp_path, monkeypatch):
        """Bar chart should only show ε=0.1 data, not other epsilons."""
        import matplotlib.pyplot as plt
        from scripts.generate_figures import plot_robustness_bars

        rows = []
        for eps in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]:
            for seed in [42, 123, 456]:
                rows.append({
                    "dataset": "ccfd", "model_type": "neural",
                    "defence_type": "none", "attack_type": "capgd",
                    "attack_epsilon": eps, "seed": seed,
                    "robust_pr_auc": 0.65, "clean_pr_auc": 0.85,
                })
        df = pd.DataFrame(rows)

        monkeypatch.setattr(plt, "close", lambda *a, **kw: None)
        plot_robustness_bars(df, str(tmp_path))
        assert (tmp_path / "robustness_bars.png").exists()

        fig = plt.gcf()
        ax = fig.axes[0]
        # 2 bars per group (clean + robust), 1 group → 2 bars total
        bars = [p for p in ax.patches if hasattr(p, 'get_height') and p.get_height() > 0]
        assert len(bars) == 2, f"Expected 2 bars (1 group × 2 metrics), got {len(bars)}"
        plt.close("all")

    def test_bar_labels_include_model_and_attack(self, tmp_path, monkeypatch):
        """Bar labels should include model_type and attack_type for disambiguation."""
        import matplotlib.pyplot as plt
        from scripts.generate_figures import plot_robustness_bars

        rows = []
        for model in ["neural", "tree"]:
            for defence in ["none", "adversarial_training"]:
                for seed in [42, 123, 456]:
                    rows.append({
                        "dataset": "ccfd", "model_type": model,
                        "defence_type": defence, "attack_type": "capgd",
                        "attack_epsilon": 0.1, "seed": seed,
                        "robust_pr_auc": 0.65, "clean_pr_auc": 0.85,
                    })
        df = pd.DataFrame(rows)

        monkeypatch.setattr(plt, "close", lambda *a, **kw: None)
        plot_robustness_bars(df, str(tmp_path))
        assert (tmp_path / "robustness_bars.png").exists()

        fig = plt.gcf()
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_xticklabels()]
        # Each label should contain model_type
        for label in labels:
            assert "neural" in label or "tree" in label, f"Label missing model_type: {label}"
        # 4 groups (2 models × 2 defences) → 4 labels
        assert len(labels) == 4, f"Expected 4 bar groups, got {len(labels)}"
        plt.close("all")


class TestPlotDefenceHeatmap:
    """Tests for defence heatmap figure generation."""

    def _make_heatmap_data(self):
        """Registry with neural baseline, adversarial_training, and ensemble."""
        rows = []
        # neural baseline (none)
        for seed in [42, 123, 456]:
            rows.append({
                "dataset": "ccfd", "model_type": "neural",
                "defence_type": "none", "attack_type": "capgd",
                "attack_epsilon": 0.1, "seed": seed,
                "robust_pr_auc": 0.65, "clean_pr_auc": 0.85,
            })
        # adversarial_training (neural)
        for seed in [42, 123, 456]:
            rows.append({
                "dataset": "ccfd", "model_type": "neural",
                "defence_type": "adversarial_training", "attack_type": "capgd",
                "attack_epsilon": 0.1, "seed": seed,
                "robust_pr_auc": 0.75, "clean_pr_auc": 0.83,
            })
        # ensemble (model_type=ensemble, defence_type=ensemble)
        for seed in [42, 123, 456]:
            rows.append({
                "dataset": "ccfd", "model_type": "ensemble",
                "defence_type": "ensemble", "attack_type": "capgd",
                "attack_epsilon": 0.1, "seed": seed,
                "robust_pr_auc": 0.72, "clean_pr_auc": 0.87,
            })
        return pd.DataFrame(rows)

    def test_ensemble_column_present(self, tmp_path):
        """Heatmap should include ensemble defence column with correct delta."""
        from scripts.generate_figures import aggregate_seeds

        df = self._make_heatmap_data()
        agg = aggregate_seeds(df)

        # Reproduce the heatmap merge logic to validate pivot content
        baseline = agg[agg["defence_type"] == "none"].copy()
        baseline = baseline.rename(columns={"robust_pr_auc_mean": "baseline_robust"})
        merge_on = ["dataset", "model_type", "attack_type", "attack_epsilon"]
        merged = agg.merge(baseline[merge_on + ["baseline_robust"]], on=merge_on, how="left")

        # Ensemble has no same-model baseline — should be NaN before fallback
        ens_rows = merged[merged["model_type"] == "ensemble"]
        assert ens_rows["baseline_robust"].isna().all(), "Ensemble should have no same-model baseline"

        # Now test the full function produces a file with ensemble
        from scripts.generate_figures import plot_defence_heatmap
        plot_defence_heatmap(df, str(tmp_path))
        assert (tmp_path / "defence_heatmap.png").exists()

    def test_ensemble_delta_correct(self, tmp_path, monkeypatch):
        """Ensemble delta should be computed against neural none baseline."""
        from scripts.generate_figures import plot_defence_heatmap
        import matplotlib.pyplot as plt

        df = self._make_heatmap_data()

        monkeypatch.setattr(plt, "close", lambda *a, **kw: None)
        plot_defence_heatmap(df, str(tmp_path))

        fig = plt.gcf()
        ax = fig.axes[0]
        # Heatmap should have 2 columns: adversarial_training and ensemble
        col_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "adversarial_training" in col_labels, f"Missing adversarial_training in {col_labels}"
        assert "ensemble" in col_labels, f"Missing ensemble in {col_labels}"
        plt.close("all")

    def test_multi_epsilon_baseline_no_duplication(self, tmp_path):
        """Multi-epsilon baseline should not produce many-to-many join."""
        from scripts.generate_figures import aggregate_seeds, plot_defence_heatmap
        import matplotlib.pyplot as plt

        rows = []
        # neural baseline at multiple epsilons
        for eps in [0.05, 0.1, 0.2]:
            for seed in [42, 123, 456]:
                rows.append({
                    "dataset": "ccfd", "model_type": "neural",
                    "defence_type": "none", "attack_type": "capgd",
                    "attack_epsilon": eps, "seed": seed,
                    "robust_pr_auc": 0.80 - eps, "clean_pr_auc": 0.90,
                })
        # adversarial_training at eps=0.1 only
        for seed in [42, 123, 456]:
            rows.append({
                "dataset": "ccfd", "model_type": "neural",
                "defence_type": "adversarial_training", "attack_type": "capgd",
                "attack_epsilon": 0.1, "seed": seed,
                "robust_pr_auc": 0.75, "clean_pr_auc": 0.88,
            })

        df = pd.DataFrame(rows)
        agg = aggregate_seeds(df)

        # With attack_epsilon in merge, adv_training at 0.1 should only join
        # with baseline at 0.1, not with 0.05 or 0.2
        baseline = agg[agg["defence_type"] == "none"].copy()
        baseline = baseline.rename(columns={"robust_pr_auc_mean": "baseline_robust"})
        merge_on = ["dataset", "model_type", "attack_type", "attack_epsilon"]
        merged = agg.merge(baseline[merge_on + ["baseline_robust"]], on=merge_on, how="left")
        defended = merged[merged["defence_type"] == "adversarial_training"]
        # Should be exactly 1 row (at eps=0.1), not 3 from many-to-many
        assert len(defended) == 1, f"Expected 1 row, got {len(defended)} (many-to-many join?)"
        plt.close("all")
