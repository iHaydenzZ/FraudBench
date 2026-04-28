"""Phase A.3 micro-diagnostic: locate the source of the 0.890 g3 pass rate.

Phase A (OHE all-zero) and Phase A.2 (raw data quality) hypotheses both falsified.
New hypothesis: the 0.890 number comes from preprocessor inverse_transform
floating-point drift on `pub_rec_bankruptcies` and `pub_rec`, combined with
strict `<=` check (no tolerance). Most LCLD rows have both fields = 0; tiny
float drift in either direction produces False violations.

This script tests 3 g3-check variants on LCLD test set per seed:
  V1: raw X_test           (control — should match Phase A.2 result of ~0.999)
  V2: inverse(transform(X_test))  (round-trip — tests inverse-transform drift)
  V3: V2 with tolerance 1e-6      (drift + tolerance — should recover near 1.000)

If V2 << V1 and V3 ≈ V1, the inverse-transform-drift hypothesis is confirmed
and the fix is a 1-character notebook change: `tol=1e-6` in check_g3.

Run on Colab: python scratch/diagnose_g3_pipeline.py
"""

import pandas as pd

from datasets.loader import load_dataset
from datasets.splitter import split_dataset
from preprocessing.processor import DataPreprocessor


def check_g3_strict(df):
    """Mirror notebook's check_g3_bankruptcies — strict <= comparison."""
    return df["pub_rec_bankruptcies"].astype(float) <= df["pub_rec"].astype(float)


def check_g3_tolerant(df, tol=1e-6):
    """g3 with absolute tolerance — should be robust to inverse-transform drift."""
    return df["pub_rec_bankruptcies"].astype(float) <= df["pub_rec"].astype(float) + tol


def round_trip(pp: DataPreprocessor, X: pd.DataFrame) -> pd.DataFrame:
    """Apply transform then inverse_transform via the underlying ColumnTransformer."""
    X_proc = pp.pipeline.transform(X)
    # Use the named numeric pipeline's inverse_transform on the relevant columns
    # The ColumnTransformer doesn't natively support inverse_transform across all
    # branches, so we manually invert the scaler on numeric columns we care about.
    # Cheaper approach: reconstruct only pub_rec and pub_rec_bankruptcies.
    num_pipeline = pp.pipeline.named_transformers_["num"]
    num_cols = list(pp.pipeline.transformers_[0][2])  # numeric columns in fit order

    # Find the slices in X_proc that correspond to the numeric block
    # (numeric block comes first in transformers list since it's appended first)
    num_block_width = len(num_cols)
    X_num_proc = X_proc[:, :num_block_width]

    # Reverse: scaler.inverse_transform → impute round-trip is identity for non-NaN
    scaler = num_pipeline.named_steps["scaler"]
    X_num_raw = scaler.inverse_transform(X_num_proc)

    df = pd.DataFrame(X_num_raw, columns=num_cols, index=X.index)
    return df


def diagnose_g3_pipeline(seeds: list):
    sample_frac = 0.1
    print(f"\n{'=' * 90}")
    print(f"=== LCLD g3 pass rate under 3 pipeline variants (sample_frac={sample_frac}) ===")
    print(f"{'=' * 90}")
    print(
        f"{'seed':<6}{'V1_raw':<14}{'V2_round_trip':<18}{'V3_round_trip+tol':<22}{'V2-V1 delta':<14}{'V3-V1 delta':<14}"
    )

    drift_examples = None
    for seed in seeds:
        ds = load_dataset("lcld", {"sample_frac": sample_frac})
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(ds, random_state=seed)
        pp = DataPreprocessor(ds.feature_types).fit(X_train)

        # V1: raw
        v1 = check_g3_strict(X_test).mean()

        # V2: round-trip through scaler/imputer/inverse on numeric columns
        try:
            X_round_trip = round_trip(pp, X_test)
            # Need pub_rec and pub_rec_bankruptcies columns
            v2_df = pd.DataFrame(
                {
                    "pub_rec": X_round_trip["pub_rec"],
                    "pub_rec_bankruptcies": X_round_trip["pub_rec_bankruptcies"],
                }
            )
            v2 = check_g3_strict(v2_df).mean()
            v3 = check_g3_tolerant(v2_df, tol=1e-6).mean()
        except KeyError as e:
            print(f"  seed={seed}: round_trip failed — column missing: {e}")
            continue

        print(f"{seed:<6}{v1:<14.6f}{v2:<18.6f}{v3:<22.6f}{v2 - v1:<+14.6f}{v3 - v1:<+14.6f}")

        # Capture seed-42 drift examples for inspection
        if seed == 42 and drift_examples is None:
            v1_mask = check_g3_strict(X_test)
            v2_mask = check_g3_strict(v2_df)
            # Rows that pass V1 but fail V2 — these are the drift-induced violations
            drifted = v1_mask & ~v2_mask
            drift_examples = pd.DataFrame(
                {
                    "pub_rec_raw": X_test["pub_rec"].astype(float),
                    "pub_rec_bankrupt_raw": X_test["pub_rec_bankruptcies"].astype(float),
                    "pub_rec_round_trip": v2_df["pub_rec"],
                    "pub_rec_bankrupt_round_trip": v2_df["pub_rec_bankruptcies"],
                    "drifted": drifted,
                }
            )

    if drift_examples is not None:
        print("\n--- Seed-42 inverse-transform drift inspection ---")
        n_drifted = int(drift_examples["drifted"].sum())
        print(f"Rows that pass V1 (raw) but fail V2 (round-trip): {n_drifted}")
        if n_drifted > 0:
            sample = drift_examples[drift_examples["drifted"]].head(8)
            print("\nFirst 8 drifted rows (raw vs round-trip values):")
            print(sample.drop(columns=["drifted"]).to_string())

            # Show drift magnitude
            diff_bankrupt = sample["pub_rec_bankrupt_round_trip"] - sample["pub_rec_bankrupt_raw"]
            diff_pubrec = sample["pub_rec_round_trip"] - sample["pub_rec_raw"]
            print("\nDrift magnitude on these 8 rows:")
            print(
                f"  pub_rec_bankruptcies: min={diff_bankrupt.min():.2e}, "
                f"max={diff_bankrupt.max():.2e}, mean={diff_bankrupt.mean():.2e}"
            )
            print(
                f"  pub_rec:              min={diff_pubrec.min():.2e}, "
                f"max={diff_pubrec.max():.2e}, mean={diff_pubrec.mean():.2e}"
            )


def diagnose_value_distribution(seed: int):
    """How many test rows have (pub_rec=0, pub_rec_bankruptcies=0)?"""
    ds = load_dataset("lcld", {"sample_frac": 0.1})
    _, _, X_test, _, _, _ = split_dataset(ds, random_state=seed)
    pubrec = X_test["pub_rec"].astype(float)
    bankrupt = X_test["pub_rec_bankruptcies"].astype(float)

    print(f"\n--- Distribution of (pub_rec, pub_rec_bankruptcies) on seed={seed} test set ---")
    print(f"Total test rows: {len(X_test)}")
    print(f"  pub_rec == 0:                    {int((pubrec == 0).sum())} ({(pubrec == 0).mean() * 100:.1f}%)")
    print(f"  pub_rec_bankruptcies == 0:       {int((bankrupt == 0).sum())} ({(bankrupt == 0).mean() * 100:.1f}%)")
    print(
        f"  both == 0:                       {int(((pubrec == 0) & (bankrupt == 0)).sum())} "
        f"({((pubrec == 0) & (bankrupt == 0)).mean() * 100:.1f}%)"
    )
    print(f"  pub_rec_bankruptcies is NaN:     {int(bankrupt.isna().sum())}")


def main():
    SEEDS = [42, 123, 456]
    diagnose_value_distribution(seed=42)
    diagnose_g3_pipeline(seeds=SEEDS)
    print("\n" + "=" * 90)
    print("Phase A.3 diagnostic complete.")
    print("=" * 90)
    print("\nInterpretation guide:")
    print("  - V1 ~ V2 (delta near 0):       inverse-transform drift is NOT the cause")
    print("                                   → 0.890 must come from attack pipeline,")
    print("                                     not inverse-transform")
    print("  - V2 << V1 (delta -10pp+):       inverse-transform drift IS the cause")
    print("                                   → fix is `check_g3 + tol=1e-6` or similar")
    print("  - V2 ≈ V1 but V3 = V1 :         drift exists but is sub-threshold; check still")
    print("                                   needs tolerance for safety")


if __name__ == "__main__":
    main()
