"""Generate thesis-ready figures from the experiment registry."""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


NUMERIC_COLS = [
    "seed",
    "attack_epsilon",
    "validity_rate",
    "adv_validity_rate",
    "clean_pr_auc",
    "clean_precision",
    "clean_recall",
    "clean_f1",
    "robust_pr_auc",
    "robust_precision",
    "robust_recall",
    "robust_f1",
    "clean_accuracy",
    "robust_accuracy",
    "train_time_sec",
    "attack_time_sec",
]


def load_registry(path: str = "results/registry.csv") -> pd.DataFrame:
    """Load and type-convert the experiment registry."""
    df = pd.read_csv(path)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-seed runs: compute mean and std per experiment config."""
    group_cols = ["dataset", "model_type", "defence_type", "attack_type", "attack_epsilon"]
    metric_cols = [
        "clean_pr_auc",
        "clean_precision",
        "clean_recall",
        "clean_f1",
        "robust_pr_auc",
        "robust_precision",
        "robust_recall",
        "robust_f1",
        "clean_accuracy",
        "robust_accuracy",
        "train_time_sec",
        "attack_time_sec",
    ]
    # Only include columns that exist
    existing_metrics = [c for c in metric_cols if c in df.columns]
    existing_groups = [c for c in group_cols if c in df.columns]

    agg = df.groupby(existing_groups, dropna=False)[existing_metrics].agg(["mean", "std"])
    # Flatten multi-level columns
    agg.columns = ["_".join(col).strip() for col in agg.columns]
    return agg.reset_index()


def plot_robustness_bars(df: pd.DataFrame, output_dir: str):
    """
    Figure 1: Clean vs Robust PR-AUC per defence, per dataset (2x2 subplots).

    Filters to ε=0.1 only (canonical epsilon). Groups bars by model_type
    using colour coding.
    """
    agg = aggregate_seeds(df)
    # Filter to rows that have robust metrics
    agg = agg[agg["robust_pr_auc_mean"].notna() & (agg["robust_pr_auc_mean"] > 0)]
    # Filter to canonical epsilon only
    agg = agg[np.isclose(agg["attack_epsilon"], 0.1)]

    datasets = sorted(agg["dataset"].unique())
    n = len(datasets)
    if n == 0:
        print("  Skipping robustness_bars: no robust data available at ε=0.1.")
        return

    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        sub = agg[agg["dataset"] == dataset].copy()
        sub["label"] = sub["model_type"] + " / " + sub["defence_type"] + " / " + sub["attack_type"]

        x = np.arange(len(sub))
        width = 0.35
        ax.bar(
            x - width / 2,
            sub["clean_pr_auc_mean"],
            width,
            label="Clean PR-AUC",
            yerr=sub.get("clean_pr_auc_std", 0),
            capsize=3,
        )
        ax.bar(
            x + width / 2,
            sub["robust_pr_auc_mean"],
            width,
            label="Robust PR-AUC",
            yerr=sub.get("robust_pr_auc_std", 0),
            capsize=3,
        )
        ax.set_ylabel("PR-AUC")
        ax.set_title(f"{dataset.upper()} (ε=0.1)")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Robustness Degradation: Clean vs Robust PR-AUC (ε=0.1)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "robustness_bars.png"), dpi=150)
    plt.close(fig)
    print("  Saved robustness_bars.png")


def plot_attack_comparison(df: pd.DataFrame, output_dir: str):
    """
    Figure 2: Robust PR-AUC by attack type and model type (grouped bars).
    """
    agg = aggregate_seeds(df)
    agg = agg[agg["robust_pr_auc_mean"].notna() & (agg["robust_pr_auc_mean"] > 0)]

    if agg.empty:
        print("  Skipping attack_comparison: no robust data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=agg, x="attack_type", y="robust_pr_auc_mean", hue="model_type", ax=ax, capsize=0.05)
    ax.set_ylabel("Robust PR-AUC (mean across seeds)")
    ax.set_xlabel("Attack Type")
    ax.set_title("Attack Comparison: Robust PR-AUC by Attack and Model Type")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "attack_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved attack_comparison.png")


def plot_summary_table(df: pd.DataFrame, output_dir: str):
    """
    Figure 3: Cross-dataset summary table — mean +/- std, saved as CSV + PNG.
    """
    agg = aggregate_seeds(df)
    summary_cols = ["dataset", "model_type", "defence_type", "attack_type", "attack_epsilon"]
    metric_pairs = [
        ("clean_pr_auc_mean", "clean_pr_auc_std"),
        ("robust_pr_auc_mean", "robust_pr_auc_std"),
        ("clean_f1_mean", "clean_f1_std"),
        ("robust_f1_mean", "robust_f1_std"),
    ]

    rows = []
    for _, row in agg.iterrows():
        r = {c: row[c] for c in summary_cols if c in row.index}
        for mean_col, std_col in metric_pairs:
            m = row.get(mean_col, np.nan)
            s = row.get(std_col, np.nan)
            label = mean_col.replace("_mean", "")
            if pd.notna(m):
                r[label] = f"{m:.4f} +/- {s:.4f}" if pd.notna(s) else f"{m:.4f}"
            else:
                r[label] = "n/a"
        rows.append(r)

    # Annotate tree + CAPGD rows where robust == clean (attack is a no-op)
    for r in rows:
        if r.get("model_type") == "tree" and r.get("attack_type") == "capgd":
            if "robust_pr_auc" in r and r["robust_pr_auc"] != "n/a":
                r["robust_pr_auc"] = r["robust_pr_auc"] + " †"

    summary_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "summary_table.csv")
    summary_df.to_csv(csv_path, index=False)
    print("  Saved summary_table.csv")

    # Render as PNG table
    fig, ax = plt.subplots(figsize=(max(14, len(summary_df.columns) * 2), max(4, len(summary_df) * 0.4 + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)
    fig.suptitle("Cross-Dataset Summary\n† CAPGD is inapplicable to tree models (gradient-based attack)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved summary_table.png")


def plot_defence_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Figure 4: Defence effectiveness heatmap — delta robust PR-AUC vs no-defence.

    Ensemble defence is compared against the neural 'none' baseline, since
    the ensemble model includes a neural sub-model.
    """
    agg = aggregate_seeds(df)
    agg = agg[agg["robust_pr_auc_mean"].notna() & (agg["robust_pr_auc_mean"] > 0)]

    if agg.empty:
        print("  Skipping defence_heatmap: no robust data available.")
        return

    # Get baseline (defence_type == 'none') per dataset+model+attack+epsilon
    baseline = agg[agg["defence_type"] == "none"].copy()
    baseline = baseline.rename(columns={"robust_pr_auc_mean": "baseline_robust"})

    # Include attack_epsilon in merge to avoid many-to-many joins
    merge_on = ["dataset", "model_type", "attack_type", "attack_epsilon"]
    merged = agg.merge(baseline[merge_on + ["baseline_robust"]], on=merge_on, how="left")

    # For ensemble: use neural 'none' baseline (include epsilon to stay 1:1)
    neural_baseline = baseline[baseline["model_type"] == "neural"][
        ["dataset", "attack_type", "attack_epsilon", "baseline_robust"]
    ].copy()
    neural_baseline = neural_baseline.rename(columns={"baseline_robust": "neural_baseline_robust"})

    merged = merged.merge(
        neural_baseline, on=["dataset", "attack_type", "attack_epsilon"], how="left"
    )

    # Fill in ensemble rows: use neural baseline where same-model baseline is missing
    is_ensemble = merged["model_type"] == "ensemble"
    merged.loc[is_ensemble, "baseline_robust"] = merged.loc[is_ensemble, "neural_baseline_robust"]
    merged = merged.drop(columns=["neural_baseline_robust"])

    merged["delta"] = merged["robust_pr_auc_mean"] - merged["baseline_robust"]

    # Filter to defences only (exclude 'none')
    defended = merged[merged["defence_type"] != "none"]
    if defended.empty:
        print("  Skipping defence_heatmap: no defence experiments found.")
        return

    pivot = defended.pivot_table(values="delta", index=["dataset", "attack_type"], columns="defence_type")

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.5 + 1)))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Defence Effectiveness: Delta Robust PR-AUC vs No Defence\n(ensemble compared vs neural baseline)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "defence_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  Saved defence_heatmap.png")


def plot_training_time(df: pd.DataFrame, output_dir: str):
    """
    Figure 5: Training time comparison — bar chart by model and defence.
    """
    agg = aggregate_seeds(df)
    if "train_time_sec_mean" not in agg.columns or agg["train_time_sec_mean"].isna().all():
        print("  Skipping training_time: no timing data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=agg, x="model_type", y="train_time_sec_mean", hue="defence_type", ax=ax, capsize=0.05)
    ax.set_ylabel("Training Time (seconds, mean)")
    ax.set_xlabel("Model Type")
    ax.set_title("Training Time by Model and Defence")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_time.png"), dpi=150)
    plt.close(fig)
    print("  Saved training_time.png")


def plot_robustness_curves(df: pd.DataFrame, output_dir: str):
    """
    Figure 6: Robustness curves — Robust PR-AUC vs epsilon per dataset.

    Shows multi-epsilon sweeps as lines and single-epsilon defences as scatter
    markers.  Filters to neural models only (CAPGD is a no-op on trees).
    """
    agg = aggregate_seeds(df)
    agg = agg[agg["robust_pr_auc_mean"].notna() & (agg["robust_pr_auc_mean"] > 0)]

    if agg.empty:
        print("  Skipping robustness_curves: no robust data available.")
        return

    # Filter to neural models only — CAPGD has no effect on tree models
    agg = agg[agg["model_type"] == "neural"]
    if agg.empty:
        print("  Skipping robustness_curves: no neural model data available.")
        return

    # Classify each (dataset, defence, attack) group as multi-epsilon or single
    group_cols = ["dataset", "model_type", "defence_type", "attack_type"]
    counts = agg.groupby(group_cols)["attack_epsilon"].nunique().reset_index(name="n_eps")
    multi_eps = counts[counts["n_eps"] > 1]

    # Need at least one multi-epsilon group to anchor the x-axis
    if multi_eps.empty:
        print("  Skipping robustness_curves: no multi-epsilon groups found.")
        return

    datasets = sorted(agg["dataset"].unique())
    n = len(datasets)
    if n == 0:
        print("  Skipping robustness_curves: no datasets with data.")
        return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        sub = agg[agg["dataset"] == dataset]

        # Iterate by (defence, attack) pairs to avoid collapsing different attacks
        series_keys = sorted(sub[["defence_type", "attack_type"]].drop_duplicates().itertuples(index=False))
        for defence, attack in series_keys:
            d = sub[(sub["defence_type"] == defence) & (sub["attack_type"] == attack)].sort_values("attack_epsilon")
            yerr = d["robust_pr_auc_std"].fillna(0)
            series_label = f"{defence} ({attack})" if sub["attack_type"].nunique() > 1 else defence

            if len(d) >= 2:
                # Line plot for multi-epsilon data
                ax.errorbar(
                    d["attack_epsilon"],
                    d["robust_pr_auc_mean"],
                    yerr=yerr,
                    marker="o",
                    capsize=3,
                    label=series_label,
                )
            elif len(d) == 1:
                # Single-point marker for defences with only one epsilon
                ax.scatter(
                    d["attack_epsilon"].values,
                    d["robust_pr_auc_mean"].values,
                    marker="^",
                    s=100,
                    zorder=5,
                    label=f"{series_label} (ε={d['attack_epsilon'].values[0]})",
                )

        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Robust PR-AUC")
        ax.set_title(dataset.upper())
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Robustness Curves: Robust PR-AUC vs Epsilon (Neural Models)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "robustness_curves.png"), dpi=150)
    plt.close(fig)
    print("  Saved robustness_curves.png")


FIGURE_REGISTRY = {
    "robustness_bars": plot_robustness_bars,
    "attack_comparison": plot_attack_comparison,
    "summary_table": plot_summary_table,
    "defence_heatmap": plot_defence_heatmap,
    "training_time": plot_training_time,
    "robustness_curves": plot_robustness_curves,
}


def main():
    parser = argparse.ArgumentParser(description="Generate thesis-ready figures from registry")
    parser.add_argument("--registry", default="results/registry.csv", help="Path to registry CSV")
    parser.add_argument("--output", default="results/figures", help="Output directory for figures")
    parser.add_argument("--figures", default=None, help="Comma-separated figure names to generate (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading registry from {args.registry}...")
    df = load_registry(args.registry)
    print(f"  {len(df)} experiment rows loaded.")

    if args.figures:
        names = [n.strip() for n in args.figures.split(",")]
    else:
        names = list(FIGURE_REGISTRY.keys())

    for name in names:
        if name not in FIGURE_REGISTRY:
            print(f"  Unknown figure: {name}. Available: {list(FIGURE_REGISTRY.keys())}")
            continue
        print(f"\nGenerating {name}...")
        FIGURE_REGISTRY[name](df, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
