"""Analyse how z_threshold values affect input validation defence effectiveness.

Compares z_threshold={3.0, 5.0, 10.0} against baseline (no defence) for CCFD
and Sparkov datasets, both neural and tree models.
"""

import argparse
import os
import re

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_registry(path: str) -> pd.DataFrame:
    from scripts.generate_figures import load_registry as _load
    return _load(path)


def extract_z_threshold(experiment_name: str):
    """Extract z_threshold from experiment name, e.g. 'ccfd_neural_input_val_z5' -> 5.0"""
    match = re.search(r"_z(\d+(?:\.\d+)?)$", experiment_name)
    if match:
        return float(match.group(1))
    # Default z_threshold for standard input_val configs
    if "input_val" in experiment_name:
        return 3.0
    return None


def main():
    parser = argparse.ArgumentParser(description="Z-threshold sweep analysis")
    parser.add_argument("--registry", default="results/registry.csv")
    parser.add_argument("--output", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = load_registry(args.registry)
    print(f"Loaded {len(df)} rows from {args.registry}")

    # Filter to relevant experiments
    relevant = df[
        (df["dataset"].isin(["ccfd", "sparkov"]))
        & (df["defence_type"].isin(["none", "input_validation"]))
        & (df["seed"] == 42)
    ].copy()

    # Extract z_threshold
    relevant["z_threshold"] = relevant["experiment_name"].apply(extract_z_threshold)
    relevant.loc[relevant["defence_type"] == "none", "z_threshold"] = "none"

    print("\nZ-threshold comparison:")
    cols = ["dataset", "model_type", "defence_type", "z_threshold",
            "clean_pr_auc", "robust_pr_auc"]
    print(relevant[cols].sort_values(["dataset", "model_type", "z_threshold"]).to_string(index=False))

    # Save CSV
    csv_path = os.path.join(args.output, "z_threshold_analysis.csv")
    relevant[cols].to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
