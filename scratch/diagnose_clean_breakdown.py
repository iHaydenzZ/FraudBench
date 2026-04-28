"""Phase A.5: per-constraint breakdown of clean_feasibility on round-tripped data.

Phase A.3 measured g3 round-trip pass rate at 0.9943 on seed-42, but the
notebook's lcld_feasibility(X_test_raw, X_test_p) returns clean_agg = 0.888 on
seed-42 (and the M1+g1 attack output mirrors it at 0.890). The notebook only
logs the aggregate, so we don't know which constraint is the bottleneck.

This diagnostic recomputes lcld_feasibility per seed using the notebook's exact
code path (verbatim copies of inverse_transform_numeric, check_g1/g2/g3/g4,
lcld_feasibility) and prints per-constraint pass rates plus violator inspection
for the bottleneck constraint.

Run on Colab:
    python scratch/diagnose_clean_breakdown.py
"""

import re

import numpy as np
import pandas as pd

from datasets.loader import load_dataset
from datasets.splitter import split_dataset
from preprocessing.processor import DataPreprocessor


# Verbatim from notebooks/g1_projection_attack.ipynb Cells 6-8
G1_TOL = 0.10
TOL_OHE = 0.01


def _to_float(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def get_scaler_and_num_names(pp):
    num_pipeline = pp.pipeline.named_transformers_["num"]
    num_cols = list(pp.pipeline.transformers_[0][2])
    return num_pipeline.named_steps["scaler"], num_cols


def inverse_transform_numeric(X_proc, num_feature_names, scaler):
    sanitize = lambda c: re.sub(r"[\[\]<>]", "_", c)
    sanitized_num = [sanitize(c) for c in num_feature_names]
    proc_cols = X_proc.columns.tolist()
    matched = [(raw, san) for raw, san in zip(num_feature_names, sanitized_num) if san in proc_cols]
    raw_names = [m[0] for m in matched]
    san_names = [m[1] for m in matched]
    idx_in_scaler = [num_feature_names.index(r) for r in raw_names]
    X_scaled = X_proc[san_names].values
    means = scaler.mean_[idx_in_scaler]
    scales = scaler.scale_[idx_in_scaler]
    return pd.DataFrame(X_scaled * scales + means, columns=raw_names, index=X_proc.index)


def reconstruct_term_from_ohe(X_proc):
    term_cols = [c for c in X_proc.columns if c.startswith("term_")]
    if not term_cols:
        return None
    term_vals = {}
    for col in term_cols:
        v = pd.to_numeric(col.replace("term_", "").replace("months", "").strip(), errors="coerce")
        if not np.isnan(v):
            term_vals[col] = v
    if not term_vals:
        return None
    term_df = X_proc[list(term_vals.keys())]
    return term_df.idxmax(axis=1).map(term_vals)


def check_g1_installment(df, tol=G1_TOL):
    loan = _to_float(df["loan_amnt"])
    rate = _to_float(df["int_rate"])
    term = _to_float(df["term"])
    inst = _to_float(df["installment"])
    r = rate / 12.0 / 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = loan * r * (1 + r) ** term / ((1 + r) ** term - 1)
    return (inst - expected).abs() < tol


def check_g2_open_total(df):
    return _to_float(df["open_acc"]) <= _to_float(df["total_acc"])


def check_g3_bankruptcies(df):
    return _to_float(df["pub_rec_bankruptcies"]) <= _to_float(df["pub_rec"])


def check_g4_term_ohe(X_proc, tol=TOL_OHE):
    cols = [c for c in X_proc.columns if c.startswith("term_")]
    if not cols:
        return None
    ohe = X_proc[cols].values
    valid = (np.abs(ohe.sum(axis=1) - 1.0) < tol) & (np.abs(ohe.max(axis=1) - 1.0) < tol)
    return pd.Series(valid, index=X_proc.index)


def _pass(s):
    return float(s.fillna(True).mean()) if s is not None else float("nan")


def lcld_feasibility_with_masks(X_raw, X_proc):
    X_raw = X_raw.copy()
    t = reconstruct_term_from_ohe(X_proc)
    if t is not None:
        X_raw["term"] = t.values
    g1 = check_g1_installment(X_raw)
    g2 = check_g2_open_total(X_raw)
    g3 = check_g3_bankruptcies(X_raw)
    g4 = check_g4_term_ohe(X_proc)
    per = {
        "g1_installment": _pass(g1),
        "g2_open_total": _pass(g2),
        "g3_bankruptcy": _pass(g3),
        "g4_term_ohe": _pass(g4),
    }

    def _b(s):
        return s.fillna(True).astype(bool) if s is not None else pd.Series(True, index=X_raw.index)

    agg = float((_b(g1) & _b(g2) & _b(g3) & _b(g4)).mean())
    return per, agg, {"g1": g1, "g2": g2, "g3": g3, "g4": g4}, X_raw


def diagnose(seed: int):
    print(f"\n{'=' * 90}")
    print(f"=== seed={seed} clean per-constraint breakdown (notebook code path) ===")
    print(f"{'=' * 90}")

    ds = load_dataset("lcld", {"sample_frac": 0.1})
    X_train, _, X_test, _, _, _ = split_dataset(ds, random_state=seed)
    pp = DataPreprocessor(ds.feature_types).fit(X_train)
    X_test_p = pp.transform(X_test)
    scaler, num_names = get_scaler_and_num_names(pp)
    X_test_raw = inverse_transform_numeric(X_test_p, num_names, scaler)

    per, agg, masks, X_raw_with_term = lcld_feasibility_with_masks(X_test_raw, X_test_p)
    print("  per-constraint pass rates:")
    for k, v in per.items():
        print(f"    {k:<22} {v:.4f}")
    print(f"  aggregate (g1 & g2 & g3 & g4): {agg:.4f}")

    # If anything is < 0.99, drill down
    bottleneck = min(per.items(), key=lambda kv: kv[1])
    if bottleneck[1] < 0.99:
        name, val = bottleneck
        constraint_id = name.split("_")[0]
        mask = masks[constraint_id]
        if mask is None:
            return
        viol_mask = ~mask.fillna(True).astype(bool)
        n_fail = int(viol_mask.sum())
        n_nan = int(mask.isna().sum())
        print(f"\n  Bottleneck: {name} = {val:.4f}")
        print(f"  Violators (non-NaN, fails check): {n_fail} ({n_fail / len(mask) * 100:.2f}%)")
        print(f"  NaN rows (counted as pass via fillna(True)): {n_nan}")

        if n_fail > 0:
            viol_idx = mask.index[viol_mask][:10]
            print(f"\n  First 10 {constraint_id} violators (raw values + post-inverse values):")

            if constraint_id == "g3":
                # raw vs post-inverse pub_rec, pub_rec_bankruptcies
                sample = pd.DataFrame(
                    {
                        "raw_pubrec": _to_float(X_test["pub_rec"]).loc[viol_idx].values,
                        "raw_bank": _to_float(X_test["pub_rec_bankruptcies"]).loc[viol_idx].values,
                        "inv_pubrec": X_test_raw["pub_rec"].loc[viol_idx].values,
                        "inv_bank": X_test_raw["pub_rec_bankruptcies"].loc[viol_idx].values,
                    }
                )
                sample["diff_bank_minus_pubrec_inv"] = sample["inv_bank"] - sample["inv_pubrec"]
                print(sample.to_string())
            elif constraint_id == "g2":
                sample = pd.DataFrame(
                    {
                        "raw_open": _to_float(X_test["open_acc"]).loc[viol_idx].values,
                        "raw_total": _to_float(X_test["total_acc"]).loc[viol_idx].values,
                        "inv_open": X_test_raw["open_acc"].loc[viol_idx].values,
                        "inv_total": X_test_raw["total_acc"].loc[viol_idx].values,
                    }
                )
                print(sample.to_string())
            elif constraint_id == "g1":
                sample = pd.DataFrame(
                    {
                        "raw_loan": _to_float(X_test["loan_amnt"]).loc[viol_idx].values,
                        "raw_rate": _to_float(X_test["int_rate"]).loc[viol_idx].values,
                        "raw_term": _to_float(X_test["term"]).loc[viol_idx].values,
                        "raw_inst": _to_float(X_test["installment"]).loc[viol_idx].values,
                        "inv_loan": X_test_raw["loan_amnt"].loc[viol_idx].values,
                        "inv_rate": X_test_raw["int_rate"].loc[viol_idx].values,
                        "term_recon": X_raw_with_term["term"].loc[viol_idx].values,
                        "inv_inst": X_test_raw["installment"].loc[viol_idx].values,
                    }
                )
                print(sample.to_string())
            elif constraint_id == "g4":
                term_cols = [c for c in X_test_p.columns if c.startswith("term_")]
                sample = X_test_p.loc[viol_idx, term_cols]
                sample["sum"] = sample.sum(axis=1)
                print(sample.to_string())


def main():
    for seed in [42, 123, 456]:
        diagnose(seed)
    print("\n" + "=" * 90)
    print("Phase A.5 diagnostic complete.")
    print("=" * 90)
    print("""
Interpretation:
  - If seed-42 g3 ≈ 0.89 here (matching notebook): differs from Phase A.3 V2's 0.9943.
    Difference must come from inverse_transform_numeric vs scaler.inverse_transform.
    Most likely cause: scaler.inverse_transform expects ALL numeric columns in
    fit order (188 of them), but inverse_transform_numeric uses sanitize+match
    which can change column order/selection. Check raw_pubrec vs inv_pubrec.
  - If seed-42 g3 ≈ 0.99 here: notebook's clean_agg = 0.888 was computed on a
    different X (e.g., earlier code revision); current rerun would give ~0.99.
    Doc's '0.888 clean_feasibility' figure is stale — re-run notebook to
    refresh.
  - If a different constraint (g4 or g1) is the bottleneck: that's the actual
    cause; 5 findings docs misattributed it to g3/OHE-categorical.
""")


if __name__ == "__main__":
    main()
