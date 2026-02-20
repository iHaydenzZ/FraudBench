"""Statistical significance tests for pairwise defence comparisons."""

import argparse
import os
import pandas as pd
import numpy as np
from scipy import stats


def load_registry(path: str = "results/registry.csv") -> pd.DataFrame:
    """Load registry via generate_figures to avoid duplication."""
    from scripts.generate_figures import load_registry as _load

    return _load(path)


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d for paired samples.

    Uses the pooled standard deviation as the denominator.
    Returns 0.0 if both groups have zero variance.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def pairwise_defence_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run pairwise paired t-tests between defence types.

    For each (dataset, model_type) combination, compares robust_pr_auc
    across seeds for every pair of defence types.

    Returns a DataFrame with one row per comparison.
    """
    pairs = [
        ("none", "adversarial_training"),
        ("none", "input_validation"),
        ("none", "ensemble"),
        ("adversarial_training", "input_validation"),
        ("adversarial_training", "ensemble"),
        ("input_validation", "ensemble"),
    ]

    rows = []

    for (dataset, model_type), grp in df.groupby(["dataset", "model_type"]):
        for def_a, def_b in pairs:
            df_a = grp[grp["defence_type"] == def_a][["seed", "robust_pr_auc"]].dropna()
            df_b = grp[grp["defence_type"] == def_b][["seed", "robust_pr_auc"]].dropna()

            # Inner join on seed to ensure correct pairing
            paired = df_a.merge(df_b, on="seed", suffixes=("_a", "_b"))

            row = {
                "dataset": dataset,
                "model_type": model_type,
                "defence_a": def_a,
                "defence_b": def_b,
                "n_a": len(df_a),
                "n_b": len(df_b),
            }

            # Need at least 3 paired observations for a meaningful t-test
            n_paired = len(paired)
            if n_paired < 3:
                row.update(
                    {
                        "mean_a": float(df_a["robust_pr_auc"].mean()) if len(df_a) > 0 else np.nan,
                        "mean_b": float(df_b["robust_pr_auc"].mean()) if len(df_b) > 0 else np.nan,
                        "mean_diff": np.nan,
                        "t_statistic": np.nan,
                        "p_value": np.nan,
                        "w_statistic": np.nan,
                        "w_p_value": np.nan,
                        "cohens_d": np.nan,
                        "significant": False,
                        "note": f"insufficient paired data (n={n_paired})",
                    }
                )
                rows.append(row)
                continue

            a = paired["robust_pr_auc_a"].values
            b = paired["robust_pr_auc_b"].values

            mean_diff = float(np.mean(a) - np.mean(b))

            # Check if values are identical (zero variance in differences)
            if np.allclose(a, b):
                row.update(
                    {
                        "mean_a": float(np.mean(a)),
                        "mean_b": float(np.mean(b)),
                        "mean_diff": mean_diff,
                        "t_statistic": 0.0,
                        "p_value": 1.0,
                        "w_statistic": np.nan,
                        "w_p_value": np.nan,
                        "cohens_d": 0.0,
                        "significant": False,
                        "note": "identical values across seeds",
                    }
                )
                rows.append(row)
                continue

            t_stat, p_val = stats.ttest_rel(a, b)
            d = compute_cohens_d(a, b)

            # Wilcoxon signed-rank (requires n >= 6 for meaningful results)
            if n_paired >= 6:
                w_stat, w_pval = stats.wilcoxon(a, b)
            else:
                w_stat, w_pval = np.nan, np.nan

            row.update(
                {
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "mean_diff": mean_diff,
                    "t_statistic": float(t_stat),
                    "p_value": float(p_val),
                    "w_statistic": float(w_stat) if not np.isnan(w_stat) else np.nan,
                    "w_p_value": float(w_pval) if not np.isnan(w_pval) else np.nan,
                    "cohens_d": float(d),
                    "significant": bool(p_val < 0.05),
                    "note": "",
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame) -> None:
    """Print a human-readable summary of statistical test results."""
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS — Pairwise Defence Comparisons")
    print("=" * 80)

    for (dataset, model_type), grp in results.groupby(["dataset", "model_type"]):
        print(f"\n--- {dataset} / {model_type} ---")
        for _, row in grp.iterrows():
            label = f"  {row['defence_a']} vs {row['defence_b']}"
            if row["note"]:
                print(f"{label}: SKIPPED ({row['note']})")
                continue

            sig = "*" if row["significant"] else " "
            print(
                f"{label}: "
                f"mean_diff={row['mean_diff']:+.4f}  "
                f"t={row['t_statistic']:.3f}  "
                f"p={row['p_value']:.4f} {sig} "
                f"d={row['cohens_d']:.3f}"
            )

    # Summary counts
    testable = results[results["note"] == ""]
    n_sig = testable["significant"].sum()
    n_total = len(testable)
    print(f"\n{n_sig}/{n_total} comparisons significant at p < 0.05")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests for pairwise defence comparisons")
    parser.add_argument(
        "--registry",
        default="results/registry.csv",
        help="Path to registry CSV (default: results/registry.csv)",
    )
    parser.add_argument(
        "--output",
        default="results/figures",
        help="Output directory for results CSV (default: results/figures)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading registry from {args.registry}...")
    df = load_registry(args.registry)
    print(f"  {len(df)} experiment rows loaded.")

    results = pairwise_defence_tests(df)

    csv_path = os.path.join(args.output, "statistical_tests.csv")
    results.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
