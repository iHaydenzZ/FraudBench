"""Analyse adversarial training trade-offs across datasets.

Computes clean accuracy cost, correlates effectiveness with dataset
characteristics, and documents the tree model limitation.
"""

import argparse
import os

import pandas as pd
from scipy import stats


def load_registry(path: str) -> pd.DataFrame:
    from scripts.generate_figures import load_registry as _load

    return _load(path)


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    from scripts.generate_figures import aggregate_seeds as _agg

    return _agg(df)


# Dataset characteristics (from docs/Context.md and dataset cards)
DATASET_CHARS = {
    "ccfd": {"fraud_rate": 0.00173, "n_features": 30, "n_samples": 284807},
    "ieee_cis": {"fraud_rate": 0.03499, "n_features": 394, "n_samples": 590540},
    "lcld": {"fraud_rate": 0.11300, "n_features": 57, "n_samples": 100653},
    "sparkov": {"fraud_rate": 0.00579, "n_features": 22, "n_samples": 1296675},
}


def compute_tradeoffs(agg: pd.DataFrame) -> pd.DataFrame:
    """Compute clean accuracy cost and robustness gain from adv training."""
    merge_keys = ["dataset", "model_type", "attack_type", "attack_epsilon"]

    baseline = agg[agg["defence_type"] == "none"].copy()
    adv_train = agg[agg["defence_type"] == "adversarial_training"].copy()

    if baseline.empty or adv_train.empty:
        print("ERROR: need both 'none' and 'adversarial_training' rows.")
        return pd.DataFrame()

    merged = adv_train.merge(
        baseline[merge_keys + ["clean_pr_auc_mean", "robust_pr_auc_mean"]],
        on=merge_keys,
        suffixes=("_at", "_base"),
    )

    merged["clean_cost"] = merged["clean_pr_auc_mean_at"] - merged["clean_pr_auc_mean_base"]
    merged["robust_gain"] = merged["robust_pr_auc_mean_at"] - merged["robust_pr_auc_mean_base"]

    # Add dataset characteristics
    for dataset, chars in DATASET_CHARS.items():
        mask = merged["dataset"] == dataset
        for key, val in chars.items():
            merged.loc[mask, key] = val

    return merged


def main():
    parser = argparse.ArgumentParser(description="Adversarial training trade-off analysis")
    parser.add_argument("--registry", default="results/registry_clean.csv")
    parser.add_argument("--output", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = load_registry(args.registry)
    print(f"Loaded {len(df)} rows")

    agg = aggregate_seeds(df)
    tradeoffs = compute_tradeoffs(agg)

    if tradeoffs.empty:
        print("No adversarial training data found.")
        return

    print("\n" + "=" * 80)
    print("ADVERSARIAL TRAINING TRADE-OFF ANALYSIS")
    print("=" * 80)

    for _, row in tradeoffs.iterrows():
        label = f"{row['dataset'].upper()} / {row['model_type']}"
        print(f"\n  {label}")
        print(f"    Clean PR-AUC cost:  {row['clean_cost']:+.4f}")
        print(f"    Robust PR-AUC gain: {row['robust_gain']:+.4f}")
        if "fraud_rate" in row:
            print(f"    Dataset fraud rate:  {row['fraud_rate']:.5f}")

    # Correlation analysis: dataset characteristics vs robustness gain
    corr_vars = ["fraud_rate", "n_features", "n_samples"]
    available = [v for v in corr_vars if v in tradeoffs.columns]
    if len(tradeoffs) >= 3 and available:
        print("\n" + "-" * 80)
        print("CORRELATION: Dataset Characteristics vs Robust PR-AUC Gain")
        print("-" * 80)
        for var in available:
            col = tradeoffs[var].astype(float)
            gain = tradeoffs["robust_gain"].astype(float)
            if col.nunique() < 2 or gain.nunique() < 2:
                print(f"  {var}: insufficient variance for correlation")
                continue
            rho, p = stats.spearmanr(col, gain)
            print(f"  {var:>12s}:  Spearman rho={rho:+.3f}  p={p:.4f}")

    # Save CSV
    csv_path = os.path.join(args.output, "adv_training_tradeoffs.csv")
    tradeoffs.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    # Note about tree models
    print("\n" + "-" * 80)
    print("TREE MODEL LIMITATION:")
    print("  Adversarial Training requires gradient computation (backpropagation).")
    print("  XGBoost tree models are non-differentiable, making Adversarial Training")
    print("  architecturally inapplicable. This is a model-family constraint, not a bug.")
    print("  The runner raises ValueError if defence=adversarial_training with model=tree.")
    print("-" * 80)


if __name__ == "__main__":
    main()
