"""Phase A.4: M1 mask leak diagnostic.

Tests the float32-downcast hypothesis for the seed-42 g3 ~ 0.89 figure.

Notebook chain (g1_projection_attack.ipynb Cell 15):
  1. capgd_attack_m1_g1_projected returns x_adv_df built from a torch float32
     tensor.
  2. Immutable cols restored via `x_adv_df[immutable_cols] = X[immutable_cols].values`.
     X here is X_test_p (sklearn → float64). pandas downcasts the assigned
     values to match the existing column dtype (float32).
  3. Result: even though M1 conceptually freezes pub_rec / pub_rec_bankruptcies,
     the restored values carry ~1e-7 float32 noise. After inverse_transform,
     this becomes raw-space drift ~5e-8 — enough to flip strict <= comparisons
     on integer-valued boundary rows where pub_rec_bankruptcies == pub_rec.

This script loads the saved X_adv parquets and diffs them against a freshly
rebuilt X_test_p, in both processed and raw space, on the two target columns.

Also re-runs g3 with strict and tolerant comparisons to cross-check Phase A.3.

Run on Colab:
    python scratch/diagnose_m1_leak.py

Interpretation key (printed at end):
  - max |diff| processed > 1e-10           → leak in processed space (assignment bug)
  - processed = 0 but raw drift > 1e-10    → float32 dtype downcast on .values assign
  - both = 0 and g3_adv ≈ g3_test          → M1 not the cause; investigate elsewhere
"""

import os

import numpy as np
import pandas as pd

from datasets.loader import load_dataset
from datasets.splitter import split_dataset
from preprocessing.processor import DataPreprocessor


# Notebook writes locally then backs up to Drive. After a fresh clone the
# local dir is empty, so fall back to the Drive backup location.
OUT_DIR_CANDIDATES = [
    "results/adv_examples/g1_projection",
    "/content/drive/MyDrive/FraudBench/results/adv_examples/g1_projection",
]
SEEDS = [42, 123, 456]
SAMPLE_FRAC = 0.1
TARGET_COLS = ["pub_rec", "pub_rec_bankruptcies"]


def find_parquet(seed: int) -> str | None:
    fname = f"lcld_neural_m1_g1proj_seed{seed}.parquet"
    for d in OUT_DIR_CANDIDATES:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None


def get_scaler_and_num_names(pp):
    num_pipeline = pp.pipeline.named_transformers_["num"]
    num_cols = list(pp.pipeline.transformers_[0][2])
    return num_pipeline.named_steps["scaler"], num_cols


def inverse_one(X_p: pd.DataFrame, col: str, scaler, num_names) -> np.ndarray:
    """Inverse one numeric column from processed back to raw."""
    idx = num_names.index(col)
    return X_p[col].astype(float).values * scaler.scale_[idx] + scaler.mean_[idx]


def diagnose_seed(seed: int):
    print(f"\n{'=' * 90}")
    print(f"=== seed={seed} M1 mask leak inspection ===")
    print(f"{'=' * 90}")

    parq = find_parquet(seed)
    if parq is None:
        print(f"  MISSING parquet for seed={seed} in any of:")
        for d in OUT_DIR_CANDIDATES:
            print(f"    - {d}/lcld_neural_m1_g1proj_seed{seed}.parquet")
        print("  → run cells 14-16 of notebooks/g1_projection_attack.ipynb first")
        return
    print(f"  loaded: {parq}")

    ds = load_dataset("lcld", {"sample_frac": SAMPLE_FRAC})
    X_train, _, X_test, _, _, _ = split_dataset(ds, random_state=seed)
    pp = DataPreprocessor(ds.feature_types).fit(X_train)
    X_test_p = pp.transform(X_test)
    scaler, num_names = get_scaler_and_num_names(pp)

    X_adv_p = pd.read_parquet(parq)
    print(f"  X_adv_p shape={X_adv_p.shape}  dtypes={X_adv_p.dtypes.value_counts().to_dict()}")
    print(f"  X_test_p shape={X_test_p.shape}  dtypes={X_test_p.dtypes.value_counts().to_dict()}")

    # Sanity check: indices and column order
    if not X_adv_p.index.equals(X_test_p.index):
        print("  WARN: indices differ (parquet may be from a different split run)")
    common_cols = [c for c in TARGET_COLS if c in X_adv_p.columns and c in X_test_p.columns]
    if not common_cols:
        print("  ERROR: no target columns in both DataFrames")
        return

    for col in common_cols:
        a_p = X_adv_p[col].astype(float).values
        t_p = X_test_p[col].astype(float).values
        diff_p = a_p - t_p
        n_p = int((np.abs(diff_p) > 1e-10).sum())
        print(f"\n  --- {col} (processed space) ---")
        print(f"    max |X_adv_p - X_test_p|: {np.abs(diff_p).max():.3e}")
        print(f"    median |diff|:            {np.median(np.abs(diff_p)):.3e}")
        print(f"    rows with |diff| > 1e-10: {n_p} ({n_p / len(diff_p) * 100:.2f}%)")

        a_r = inverse_one(X_adv_p, col, scaler, num_names)
        t_r = inverse_one(X_test_p, col, scaler, num_names)
        diff_r = a_r - t_r
        n_r = int((np.abs(diff_r) > 1e-10).sum())
        print(f"  --- {col} (raw space) ---")
        print(f"    max |X_adv_raw - X_test_raw|: {np.abs(diff_r).max():.3e}")
        print(f"    median |diff|:                {np.median(np.abs(diff_r)):.3e}")
        print(f"    rows with |diff| > 1e-10:     {n_r} ({n_r / len(diff_r) * 100:.2f}%)")

    if all(c in common_cols for c in TARGET_COLS):
        p_a = inverse_one(X_adv_p, "pub_rec", scaler, num_names)
        b_a = inverse_one(X_adv_p, "pub_rec_bankruptcies", scaler, num_names)
        p_t = inverse_one(X_test_p, "pub_rec", scaler, num_names)
        b_t = inverse_one(X_test_p, "pub_rec_bankruptcies", scaler, num_names)

        g3_adv_strict = (b_a <= p_a).mean()
        g3_test_strict = (b_t <= p_t).mean()
        g3_adv_tol = (b_a <= p_a + 1e-6).mean()
        print("\n  g3 (full test set, raw):")
        print(f"    on X_test_p inverse:                 {g3_test_strict:.4f}  (~Phase A.3 V2)")
        print(f"    on X_adv_p inverse (strict):         {g3_adv_strict:.4f}  (notebook number)")
        print(f"    on X_adv_p inverse (tol=1e-6):       {g3_adv_tol:.4f}")

        flipped = (b_t <= p_t) & ~(b_a <= p_a)
        n_flip = int(flipped.sum())
        print(f"\n  Rows where M1-frozen but g3 still flipped pass→fail: {n_flip}")
        if n_flip > 0:
            idx = np.where(flipped)[0][:8]
            sample = pd.DataFrame(
                {
                    "pub_rec_test": p_t[idx],
                    "pub_rec_adv": p_a[idx],
                    "bank_test": b_t[idx],
                    "bank_adv": b_a[idx],
                    "diff_pubrec": p_a[idx] - p_t[idx],
                    "diff_bank": b_a[idx] - b_t[idx],
                }
            )
            print("\n  First 8 flipped rows (raw values, post-inverse):")
            print(sample.to_string())


def main():
    for seed in SEEDS:
        diagnose_seed(seed)
    print("\n" + "=" * 90)
    print("Phase A.4 diagnostic complete.")
    print("=" * 90)
    print("""
Interpretation:
  1. max |diff| processed > 1e-10
       → M1 leak in processed space (assignment bug; immutable_cols logic broken)
  2. processed diff = 0 but raw |diff| ~ 1e-7 ish, AND tol=1e-6 recovers g3 ≈ 1.0
       → float32 dtype downcast on `.values` assign — fix is to upcast the
         x_adv_df immediately after construction:
           x_adv_df = pd.DataFrame(... ).astype(np.float64)
         (or restore via numeric column slicing into a fresh float64 array)
  3. processed diff = 0 AND raw |diff| ~ 4e-16 only AND g3 ~ 0.99
       → not the M1 leak; the 0.890 number must come from an earlier
         saved parquet (stale). Re-run notebook before further investigation.
""")


if __name__ == "__main__":
    main()
