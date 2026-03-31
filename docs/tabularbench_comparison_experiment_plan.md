# TabularBench Comparison Experiment Plan
## Direction 3: ADV vs ADV+CTR Gap & Direction 7: Ranking Metric Sensitivity

**Author:** Hayden  
**Date:** 2026-03-31  
**Target venue:** ICAIF 2026  
**Execution environment:** Google Colab (GPU, FraudBench repo cloned)

---

## 0. Context and Objective

### What we already have

From `tabularbench_comparison.ipynb` (seed=42, LCLD neural baseline, ε=0.1):

| Data artifact | Variable / file | Shape |
|---|---|---|
| Raw test set | `lcld_Xtest_raw_seed42.parquet` | (26820, 63) |
| Processed test set | `lcld_Xtest_processed_seed42.parquet` | (26820, 188) |
| True labels | `lcld_ytest_seed42.parquet` | (26820, 1) |
| Adversarial (unmasked) | `lcld_neural_unmasked_seed42.parquet` | (26820, 188) |
| Adversarial (masked) | `lcld_neural_masked_seed42.parquet` | (26820, 188) |
| Derived features | `lcld_derived_features_sampled.parquet` | for constraint check |
| Comparison CSVs | `comparison_unmasked.csv`, `comparison_masked.csv` | 3-seed metrics |
| Constraint functions | `check_g1_installment`, `check_g2_open_total`, `check_g3_bankruptcies`, `check_g4_processed`, `check_g4_term`, `check_g5_ratio_loan_inc`, `check_g6_ratio_open_total` | defined in Cell 12 |
| Constraint globals | `G1_TOL = 0.1`, `TOLERANCE = 0.01` | defined in Cell 12 |
| Aggregate feasibility | `compute_aggregate_feasibility()` | returns (per_sample_bool, rate, n_constraints) |

Parquet files exist in two locations (Cell 15 copies local → Drive):
- **Local (during Colab execution):** `results/adv_examples/` (relative to repo root `/content/Capstone_FraudBench`)
- **Google Drive (persistent backup):** `/content/drive/MyDrive/FraudBench/results/adv_examples/`

The original notebook's Cell 11 (inverse-transform) reads from the **local** path via `ADV_SAVE_DIR = "results/adv_examples"`. The new notebook should load from Google Drive (since it runs in a fresh Colab session without the local artifacts).

**Dataset setup (from notebook Cells 2-5):**
- Google Drive is mounted at `/content/drive`
- `DRIVE_ROOT = "/content/drive/MyDrive/FraudBench"` — top-level Drive folder
- `DRIVE_DATA = "/content/drive/MyDrive/FraudBench/data"` — raw datasets on Drive
- Datasets are symlinked: `DRIVE_DATA/{LCLD,CCFD,...}` → `/content/Capstone_FraudBench/datasets/{LCLD,CCFD,...}`
- Repo is cloned to `/content/Capstone_FraudBench` and `pip install -e .` makes project imports work

### What the new notebook produces

1. **Direction 3:** A table mapping FraudBench results to TabularBench's ADV / ADV+CTR framework, with per-sample joint analysis.
2. **Direction 7:** A re-ranking analysis of TabularBench's LCLD leaderboard under alternative metrics (F1, MCC, hypothetical PR-AUC), with degenerate model identification and rank correlation statistics.

### Paper positioning

These two analyses together form a "Cross-Benchmark Comparison" subsection. Direction 7 goes into Motivation / Related Work (why PR-AUC matters). Direction 3 goes into Experiments (what happens when we apply constraint filtering to FraudBench attacks).

---

## 1. Notebook Structure

```
Notebook: tabularbench_metric_analysis.ipynb

Cell 0:  [Markdown] Title + overview
Cell 1:  [Code]     Setup — mount Drive, clone repo, install deps (then restart runtime)
Cell 2:  [Code]     Symlink datasets, load Drive artifacts, re-train model, get predictions
Cell 3:  [Code]     Inverse-transform + derived features + constraint functions (Direction 3 prerequisite)
Cell 4:  [Code]     Compute per-sample ADV and ADV+CTR (Direction 3 core)
Cell 5:  [Code]     Cross-benchmark comparison table (Direction 3 output)
Cell 6:  [Code]     Build TabularBench LCLD leaderboard DataFrame (Direction 7 input)
Cell 7:  [Code]     Degenerate model identification (Direction 7 analysis A)
Cell 8:  [Code]     Re-ranking under alternative metrics (Direction 7 analysis B)
Cell 9:  [Code]     Rank correlation analysis (Direction 7 analysis C)
Cell 10: [Code]     Visualization: rank shift scatter plot (Direction 7 figure)
Cell 11: [Code]     Combined summary narrative for paper
Cell 12: [Code]     Save all outputs to Google Drive
```

---

## 2. Direction 3: ADV vs ADV+CTR Gap — Detailed Specification

### 2.1 Goal

Map FraudBench's LCLD results into TabularBench's evaluation framework by computing:

- **ADV** (unconstrained adversarial recall): among positive-class (fraud) test samples, what fraction still gets correctly predicted as fraud after the attack? (Lower = stronger attack)
- **ADV+CTR** (constrained adversarial recall): same as ADV, but adversarial examples that violate domain constraints are "rejected" — the original (clean) prediction is restored for those samples.

### 2.2 What the original notebook saved vs what's missing

The original notebook saved parquet artifacts to Drive (see Section 0), but did **not** save:
- **Trained model weights** — `NeuralModel` is never serialized
- **Per-sample predictions** (`y_probs_clean`, `y_probs_unmasked`, `y_probs_masked`)
- **Inverse-transformed DataFrames** (`X_adv_un_inv`, `X_adv_mk_inv`) — raw-space versions needed by constraint functions
- **Fitted preprocessor** — saved only locally at `results/preprocessor_lcld_n<N>_seed42.joblib`, not on Drive

**To make the new notebook self-contained**, Cell 2 must re-run the FraudBench pipeline (load dataset → split → preprocess → train). This is ~30s on A100 and produces:
1. A fitted `preprocessor` (needed for inverse-transform in Cell 3)
2. A trained `model` (needed for predictions in Cell 2)
3. Per-sample predictions (the primary goal)

**Note on reproducibility:** GPU training may not be perfectly deterministic across runs. Verify output metrics against the original notebook's 3-seed averages (see Section 4.4). If values deviate beyond tolerance, the re-trained model is still valid for this analysis — we only need consistent predictions across clean/adversarial inputs from the *same* model, not exact reproduction of the original model.

### 2.3 Cell 1: Setup (self-contained)

This cell mirrors Cells 1-5 of `tabularbench_comparison.ipynb`:

```python
# === STEP 1: GPU check ===
import torch
assert torch.cuda.is_available(), "GPU required for model training"

# === STEP 2: Mount Google Drive ===
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_ROOT = "/content/drive/MyDrive/FraudBench"
DRIVE_ADV  = os.path.join(DRIVE_ROOT, "results/adv_examples")
DRIVE_DATA = os.path.join(DRIVE_ROOT, "data")

# === STEP 3: Clone repo ===
REPO_DIR = "/content/Capstone_FraudBench"
if not os.path.exists(os.path.join(REPO_DIR, ".git")):
    !git clone https://github.com/iHaydenzZ/Capstone_FraudBench.git {REPO_DIR}
os.chdir(REPO_DIR)

# === STEP 4: Install deps ===
# After this cell, restart runtime, then skip to Cell 2
!pip install "numpy<2.1" "scipy>=1.14,<1.15" "scikit-learn>=1.5" -q
!pip install -e . --no-deps -q
!pip install "numba>=0.61" xgboost torch art pyyaml joblib pandas -q
print(">>> RESTART RUNTIME NOW, then run from Cell 2 <<<")
```

### 2.4 Cell 2: Load artifacts, re-train model, get predictions

```python
# === After runtime restart — re-establish environment ===
import os, pandas as pd, numpy as np

# Drive mount persists across Colab runtime restarts, but verify:
if not os.path.exists("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount('/content/drive')

os.chdir("/content/Capstone_FraudBench")

DRIVE_ROOT = "/content/drive/MyDrive/FraudBench"
DRIVE_ADV  = os.path.join(DRIVE_ROOT, "results/adv_examples")
DRIVE_DATA = os.path.join(DRIVE_ROOT, "data")

# === Symlink datasets (needed for load_dataset + preprocessor fitting) ===
DATASETS_DIR = "/content/Capstone_FraudBench/datasets"
for d in ["CCFD", "ieee-fraud-detection", "LCLD", "Sparkov"]:
    src, dst = os.path.join(DRIVE_DATA, d), os.path.join(DATASETS_DIR, d)
    if os.path.islink(dst): os.unlink(dst)
    if os.path.exists(src): os.symlink(src, dst)

# === Load saved artifacts from Google Drive ===
X_test_raw    = pd.read_parquet(os.path.join(DRIVE_ADV, "lcld_Xtest_raw_seed42.parquet"))
X_test_p      = pd.read_parquet(os.path.join(DRIVE_ADV, "lcld_Xtest_processed_seed42.parquet"))
y_test        = pd.read_parquet(os.path.join(DRIVE_ADV, "lcld_ytest_seed42.parquet")).squeeze()
X_adv_un_proc = pd.read_parquet(os.path.join(DRIVE_ADV, "lcld_neural_unmasked_seed42.parquet"))
X_adv_mk_proc = pd.read_parquet(os.path.join(DRIVE_ADV, "lcld_neural_masked_seed42.parquet"))
df_derived    = pd.read_parquet(os.path.join(DRIVE_ADV, "lcld_derived_features_sampled.parquet"))

# Shapes:
# X_test_p:       (26820, 188) processed test features
# X_adv_un_proc:  (26820, 188) unmasked adversarial (processed space)
# X_adv_mk_proc:  (26820, 188) masked adversarial (processed space)
# y_test:         (26820,) true binary labels
# df_derived:     derived features for full sampled set (for constraint check)

# === Re-train model (same config as original Cell 8, seed=42) ===
# This also fits the preprocessor, which we need for inverse-transform in Cell 3.
from datasets.loader import load_dataset
from datasets.splitter import split_dataset
from preprocessing.processor import DataPreprocessor, get_preprocessor_path
from models.neural import NeuralModel
from evaluation.metrics import compute_metrics

SEED = 42
SAMPLE_FRAC = 0.1
dataset = load_dataset("lcld", config={"sample_frac": SAMPLE_FRAC})
X_train, X_val, X_test_split, y_train, y_val, y_test_split = split_dataset(
    dataset, test_size=0.2, val_size=0.2, random_state=SEED,
)

pp_path = get_preprocessor_path("lcld", SEED, len(dataset.X))
if os.path.exists(pp_path):
    preprocessor = DataPreprocessor.load(pp_path)
    X_train_p = preprocessor.transform(X_train)
else:
    preprocessor = DataPreprocessor(dataset.feature_types)
    X_train_p = preprocessor.fit_transform(X_train)
    preprocessor.save(pp_path)

model_params = {"epochs": 20, "hidden_dim": 128, "batch_size": 256, "lr": 0.001}
model = NeuralModel(model_params)
model.fit(X_train_p, y_train)

# === Get predictions on saved test/adversarial data ===
y_probs_clean    = model.predict_proba(X_test_p)
y_probs_unmasked = model.predict_proba(X_adv_un_proc)
y_probs_masked   = model.predict_proba(X_adv_mk_proc)

y_pred_clean    = (y_probs_clean >= 0.5).astype(int)
y_pred_unmasked = (y_probs_unmasked >= 0.5).astype(int)
y_pred_masked   = (y_probs_masked >= 0.5).astype(int)

# === Sanity check ===
m_clean = compute_metrics(y_test, y_probs_clean)
print(f"clean_pr_auc = {m_clean['pr_auc']:.4f} (expect ~0.2995-0.3015)")
print(f"clean_accuracy = {m_clean['accuracy']:.4f} (expect ~0.6420-0.6480)")
# Outputs: preprocessor (for Cell 3), model, y_pred_* (for Cell 4)
```

**Important note on threshold:** TabularBench's ADV is based on hard predictions (the class with highest probability). FraudBench's `compute_metrics` also uses threshold=0.5 for accuracy/recall. Stick with 0.5.

### 2.5 Cell 3: Inverse-transform + constraint functions

The constraint functions operate on **raw-space** features (e.g., `loan_amnt`, `int_rate`), not processed-space. This cell inverse-transforms the adversarial parquets and defines all constraint functions.

```python
# === Inverse-transform: processed space → raw feature space ===
# Replicates logic from original notebook Cell 11.
# The preprocessor (from Cell 2) contains the fitted StandardScaler.
import re

ct = preprocessor.pipeline
num_feature_names = []
for name, transformer, columns in ct.transformers_:
    if name == "num":
        scaler = transformer.named_steps["scaler"]
        num_feature_names = list(columns)
        break

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

X_test_inv   = inverse_transform_numeric(X_test_p, num_feature_names, scaler)
X_adv_un_inv = inverse_transform_numeric(X_adv_un_proc, num_feature_names, scaler)
X_adv_mk_inv = inverse_transform_numeric(X_adv_mk_proc, num_feature_names, scaler)

# === Attach derived features (for g5/g6, and term from OHE) ===
# Replicates original notebook Cell 12 (add_derived_from_base).
# df_derived was loaded in Cell 2 from Drive.
derived_for_test = df_derived.iloc[X_test_raw.index].reset_index(drop=True)

def add_derived_from_base(df_inv, derived_template):
    """Recompute derived ratios from perturbed base features."""
    df = df_inv.copy()
    df["ratio_loan_amnt_annual_inc"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
    df["ratio_open_acc_total_acc"] = df["open_acc"] / df["total_acc"].replace(0, np.nan)
    # Copy month_since_earliest_cr_line from template (date-derived, not perturbed)
    if "month_since_earliest_cr_line" in derived_template.columns:
        df["month_since_earliest_cr_line"] = derived_template["month_since_earliest_cr_line"].values
    return df

df_test_raw  = add_derived_from_base(X_test_inv, derived_for_test)
df_adv_un    = add_derived_from_base(X_adv_un_inv, derived_for_test)
df_adv_mk    = add_derived_from_base(X_adv_mk_inv, derived_for_test)

# === Reconstruct term from OHE (for g1 installment check) ===
# Replicates original notebook Cell 12: reconstruct_term_from_ohe()
def reconstruct_term_from_ohe(X_proc):
    term_cols = [c for c in X_proc.columns if c.startswith("term_")]
    if not term_cols: return None
    term_vals = {}
    for col in term_cols:
        val = pd.to_numeric(col.replace("term_", "").replace("months", "").strip(), errors="coerce")
        if not np.isnan(val): term_vals[col] = val
    if not term_vals: return None
    return X_proc[list(term_vals.keys())].idxmax(axis=1).map(term_vals)

for label, df, X_proc in [("test", df_test_raw, X_test_p),
                           ("adv_un", df_adv_un, X_adv_un_proc),
                           ("adv_mk", df_adv_mk, X_adv_mk_proc)]:
    term = reconstruct_term_from_ohe(X_proc)
    if term is not None:
        df["term"] = term.values

# === Define constraint functions ===
# Copied from original notebook Cell 12.
TOLERANCE = 0.01
G1_TOL = 0.1

def _to_float(series):
    if pd.api.types.is_numeric_dtype(series): return series.astype(float)
    return pd.to_numeric(series.astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")

def check_g1_installment(df, tol=G1_TOL):
    # ... (copy from original Cell 12 — see Section 4.3)
    pass

def check_g2_open_total(df):
    # ... (copy from original Cell 12)
    pass

def check_g3_bankruptcies(df):
    # ... (copy from original Cell 12)
    pass

def check_g4_processed(X_proc):
    # ... (copy from original Cell 12)
    pass

def check_g4_term(df):
    # ... (copy from original Cell 12)
    pass

def check_g5_ratio_loan_inc(df):
    # ... (copy from original Cell 12)
    pass

def check_g6_ratio_open_total(df):
    # ... (copy from original Cell 12)
    pass

def compute_aggregate_feasibility(df, X_proc=None):
    # ... (copy from original Cell 13)
    pass

# Outputs: df_adv_un, df_adv_mk (raw-space with term + derived features),
#          all constraint functions, G1_TOL
#
# NOTE: Function bodies above are shown as `pass` placeholders.
# At implementation time, copy the full implementations from the original
# notebook's Cells 12-13. See Section 4.3 for the complete list.
```

### 2.6 Cell 4: Compute per-sample ADV and ADV+CTR

This is the core computation. All inputs are now available from Cells 2-3.

```python
# === STEP 1: Identify positive-class test samples ===
pos_mask = (y_test == 1)  # boolean, shape (26820,)
n_pos = pos_mask.sum()     # expect ~5200 for LCLD at ~20% positive rate
# VERIFY: print n_pos, should be roughly 26820 * 0.2009 ≈ 5388
# (LCLD positive class rate is ~20.09%, not ~11.3%)

# === STEP 2: Compute ADV (unconstrained adversarial recall) ===
# Among positive samples, how many are still predicted as positive after attack?
# Note: "attack succeeds" = prediction flips from 1 to 0
# ADV = fraction of positive samples still correctly predicted as 1
adv_unmasked = y_pred_unmasked[pos_mask].mean()  # = robust recall on pos class
adv_masked   = y_pred_masked[pos_mask].mean()

# Cross-check: adv_unmasked should ≈ robust_recall from original notebook Cell 9 output (0.0004)
# This is very low because CAPGD is very effective at flipping predictions

# === STEP 3: Compute per-sample feasibility ===
# df_adv_un, df_adv_mk, and all constraint functions are from Cell 3.
feasible_un, _, _ = compute_aggregate_feasibility(df_adv_un, X_proc=X_adv_un_proc)
feasible_mk, _, _ = compute_aggregate_feasibility(df_adv_mk, X_proc=X_adv_mk_proc)

# === STEP 4: Compute ADV+CTR (constrained adversarial recall) ===
# Logic: for each positive test sample:
#   if adversarial example is feasible → use adversarial prediction
#   if adversarial example is NOT feasible → constraint system rejects it,
#     restore clean prediction
y_pred_constrained_un = np.where(
    feasible_un.values,       # condition: constraint satisfied
    y_pred_unmasked,          # if True: use adversarial prediction
    y_pred_clean              # if False: restore clean prediction
)

y_pred_constrained_mk = np.where(
    feasible_mk.values,
    y_pred_masked,
    y_pred_clean
)

# ADV+CTR = recall on positive class using constrained predictions
adv_ctr_unmasked = y_pred_constrained_un[pos_mask].mean()
adv_ctr_masked   = y_pred_constrained_mk[pos_mask].mean()

# === STEP 5: Also compute clean recall for reference ===
clean_recall = y_pred_clean[pos_mask].mean()  # = ID in TabularBench terms
```

### 2.7 Expected output values (sanity check)

Based on existing data:

| Metric | Expected value | Reasoning |
|---|---|---|
| n_pos | ~5388 | LCLD positive class rate ~20.09% × 26820 |
| clean_recall (ID) | ~0.557 | from original notebook Cell 9: clean_recall = 0.5573 (3-seed avg) |
| adv_unmasked (ADV) | ~0.0004 | from original notebook Cell 9: robust_recall = 0.0004 (3-seed avg) |
| adv_masked (ADV) | ~0.0017 | from original notebook Cell 9: robust_recall = 0.0017 (3-seed avg) |
| feasible rate (unmasked) | 0.0014 | from original notebook Cell 13 output: 37/26820 |
| adv_ctr_unmasked (ADV+CTR) | ≈ clean_recall ≈ 0.557 | almost all attacks are infeasible → restored to clean |
| adv_ctr_masked (ADV+CTR) | ≈ clean_recall ≈ 0.557 | same reasoning |

**Key insight to verify:** ADV+CTR should be very close to clean_recall because >99.8% of adversarial examples fail feasibility → their predictions are restored to clean. Only 37 out of 26820 unmasked samples pass feasibility; of those 37, some might be positive-class samples whose prediction was flipped. But even in the worst case (all 37 are positive and all flipped), the impact is 37/5388 ≈ 0.7% — negligible.

### 2.8 Cell 5: Cross-benchmark comparison table

Produce a publication-ready table:

```
=======================================================================
  CROSS-BENCHMARK COMPARISON: FraudBench vs TabularBench (LCLD)
=======================================================================

Framework        Model             Training  ID(Recall)  ADV     ADV+CTR   Gap(ADV+CTR−ADV)
─────────────────────────────────────────────────────────────────────────────────────────────
TabularBench     STG               Adv+CTGAN   83.28%   82.00%   81.16%     -0.84pp
TabularBench     TabTr             Adv+CTGAN   80.27%   79.50%   78.50%     -1.00pp
TabularBench     TabTr             Adv+CutMix  72.33%   72.50%   70.96%     -1.54pp
TabularBench     RLN               Adv+None    69.28%   69.48%   63.04%     -6.44pp
TabularBench     STG               Std+None    65.99%   66.40%   53.60%    -12.80pp
TabularBench     RLN               Std+None    68.51%   68.30%    0.02%    -68.28pp
TabularBench     VIME              Std+None    67.13%   67.00%    2.38%    -64.62pp
─────────────────────────────────────────────────────────────────────────────────────────────
FraudBench       MLP (neural)      Std+None    55.73%    0.04%   ~55.7%    +55.7pp
FraudBench       MLP (neural)      Std+Mask    55.73%    0.17%   ~55.7%    +55.5pp
─────────────────────────────────────────────────────────────────────────────────────────────

Notes:
  - TabularBench data from LCLD leaderboard (CAA attack with constraint repair)
  - FraudBench data from tabularbench_comparison.ipynb (CAPGD without constraint repair)
  - Gap = ADV+CTR - ADV (how much constraint filtering changes adversarial recall)
  - TabularBench (positive gap): CAA generates constraint-aware attacks, so ADV+CTR ≈ ADV
    with small drop from residual constraint violations
  - FraudBench (large positive gap): CAPGD ignores constraints, so >99.8% of attacks
    are infeasible and rejected → predictions restored to clean, giving ADV+CTR ≈ clean_recall
  - This asymmetry is the key finding: without constraint-aware attacks, post-hoc
    constraint filtering renders the attack ineffective
```

Also compute and display:
- **Feasibility-filtered attack success rate:** among positive samples where the attack *flipped* the prediction from 1→0, what fraction of those flips survive constraint filtering?
  - `attack_flipped_pos = (y_pred_clean == 1) & (y_pred_unmasked == 0) & pos_mask`
  - `n_successful_attacks_pos = attack_flipped_pos.sum()`
  - `n_feasible_successful = (attack_flipped_pos & feasible_un).sum()`
  - `filtered_success_rate = n_feasible_successful / n_successful_attacks_pos`
  - Expected: very close to 0% — almost no successful attack is also feasible
  - Note: using `y_pred_clean == 1` ensures we only count actual flips, not samples that were already misclassified on clean data (clean recall is only ~55.7%)

### 2.9 Additional analysis: per-constraint contribution to filtering

Break down which constraint is responsible for filtering:

```python
# For each positive sample where attack flipped the prediction (1→0):
attack_succeeded = (y_pred_clean == 1) & (y_pred_unmasked == 0) & pos_mask  # boolean

# Check each constraint individually on this subset
g1_pass = check_g1_installment(df_adv_un, tol=G1_TOL)[attack_succeeded]
g2_pass = check_g2_open_total(df_adv_un)[attack_succeeded]
g3_pass = check_g3_bankruptcies(df_adv_un)[attack_succeeded]
g4_pass = check_g4_processed(X_adv_un_proc)[attack_succeeded]

# Report: "Of N successful attacks on positive samples, X% fail g1, Y% fail g2, ..."
```

This tells us which constraint is doing the heavy lifting — we already know g1 (installment formula) is the dominant killer constraint.

---

## 3. Direction 7: Ranking Metric Sensitivity — Detailed Specification

### 3.1 Goal

Demonstrate that TabularBench's accuracy-based ranking on LCLD produces misleading results for fraud detection, and that alternative metrics (F1, MCC, hypothetical PR-AUC) lead to substantially different rankings.

### 3.2 Cell 6: Build leaderboard DataFrame

Hardcode the LCLD leaderboard data (70 rows from the website) into a DataFrame:

```python
# Columns to include from the leaderboard:
# Rank, Architecture, Training, Augmentation, ID(%), ADV+CTR(%), ADV(%),
# AUC, Accuracy(%), Precision(%), Recall(%), MCC

# CRITICAL: Use the exact values from the leaderboard webpage.
# ID column = clean recall on positive class
# AUC = ROC-AUC (NOT PR-AUC)

lcld_leaderboard = pd.DataFrame([
    {"rank": 1, "arch": "STG", "training": "Adv", "aug": "CT-GAN",
     "id_recall": 82.00, "adv_ctr": 81.16, "adv": 82.00,
     "auc": 0.7002, "accuracy": 50.50, "precision": 26.61,
     "recall": 83.28, "mcc": 0.2118},
    # ... all 70 rows ...
])

# Compute derived metrics
lcld_leaderboard["f1"] = 2 * (
    lcld_leaderboard["precision"]/100 * lcld_leaderboard["recall"]/100
) / (
    lcld_leaderboard["precision"]/100 + lcld_leaderboard["recall"]/100 + 1e-10
)

# TabularBench's ranking formula
lcld_leaderboard["tb_score"] = (
    lcld_leaderboard["accuracy"] + lcld_leaderboard["adv_ctr"]
) / 2
```

**Data entry note:** There are 70 rows. Recommend copy-pasting from the scraped HTML and parsing programmatically rather than typing manually.

### 3.3 Cell 7: Degenerate model identification

```python
# Definition of "degenerate" model:
# A model that achieves high ADV+CTR by predicting all-positive (or near-all-positive),
# evidenced by:
#   - Accuracy < 25% (on a ~20% positive-rate dataset, random = 50%, all-positive ≈ 20%)
#   - OR Precision ≈ positive class rate (≈ 20.09%)
#   - OR MCC ≈ 0

degenerate_mask = (
    (lcld_leaderboard["accuracy"] < 25) |
    (lcld_leaderboard["mcc"].abs() < 0.01)
)

print(f"Degenerate models: {degenerate_mask.sum()} / {len(lcld_leaderboard)}")
print(lcld_leaderboard[degenerate_mask][
    ["rank", "arch", "training", "aug", "accuracy", "adv_ctr", "mcc"]
])

# Also flag "near-degenerate" models where clean performance is very poor
near_degenerate_mask = (lcld_leaderboard["accuracy"] < 40)
```

**Expected finding:** At least 1 clearly degenerate model (TabNet Adv GOGGLE: accuracy=20.09%, ADV+CTR=100%, MCC=0). Several near-degenerate models in the bottom ranks.

### 3.4 Cell 8: Re-ranking under alternative metrics

Define multiple ranking formulas and compute new ranks:

```python
# Formula 1: Original TabularBench (Accuracy + ADV+CTR) / 2
lcld_leaderboard["score_original"] = (
    lcld_leaderboard["accuracy"] + lcld_leaderboard["adv_ctr"]
) / 2

# Formula 2: F1-based = (F1 + ADV+CTR) / 2
lcld_leaderboard["score_f1"] = (
    lcld_leaderboard["f1"] * 100 + lcld_leaderboard["adv_ctr"]
) / 2

# Formula 3: MCC-based = (MCC_normalized + ADV+CTR) / 2
# Normalize MCC from [-1,1] to [0,100] for comparable scale
lcld_leaderboard["mcc_normalized"] = (lcld_leaderboard["mcc"] + 1) / 2 * 100
lcld_leaderboard["score_mcc"] = (
    lcld_leaderboard["mcc_normalized"] + lcld_leaderboard["adv_ctr"]
) / 2

# Formula 4: AUC-based = (AUC*100 + ADV+CTR) / 2
lcld_leaderboard["score_auc"] = (
    lcld_leaderboard["auc"] * 100 + lcld_leaderboard["adv_ctr"]
) / 2

# Formula 5 (ambitious): harmonic mean of F1 and ADV+CTR
# This penalizes more aggressively when either metric is low
lcld_leaderboard["score_harmonic"] = 2 * (
    lcld_leaderboard["f1"] * 100 * lcld_leaderboard["adv_ctr"]
) / (
    lcld_leaderboard["f1"] * 100 + lcld_leaderboard["adv_ctr"] + 1e-10
)

# Compute ranks for each formula
for col in ["score_original", "score_f1", "score_mcc", "score_auc", "score_harmonic"]:
    lcld_leaderboard[f"rank_{col}"] = lcld_leaderboard[col].rank(
        ascending=False, method="min"
    ).astype(int)

# Display top-10 comparison
comparison_cols = ["arch", "training", "aug",
                   "rank_score_original", "rank_score_f1",
                   "rank_score_mcc", "rank_score_harmonic"]
print(lcld_leaderboard.sort_values("rank_score_original").head(10)[comparison_cols])
```

### 3.5 Cell 9: Rank correlation analysis

```python
from scipy.stats import kendalltau, spearmanr

ranking_pairs = [
    ("Original vs F1", "rank_score_original", "rank_score_f1"),
    ("Original vs MCC", "rank_score_original", "rank_score_mcc"),
    ("Original vs AUC", "rank_score_original", "rank_score_auc"),
    ("Original vs Harmonic", "rank_score_original", "rank_score_harmonic"),
]

print("=" * 60)
print("  RANK CORRELATION ANALYSIS")
print("=" * 60)

for name, col_a, col_b in ranking_pairs:
    tau, p_tau = kendalltau(lcld_leaderboard[col_a], lcld_leaderboard[col_b])
    rho, p_rho = spearmanr(lcld_leaderboard[col_a], lcld_leaderboard[col_b])
    print(f"\n  {name}:")
    print(f"    Kendall's tau = {tau:.4f} (p = {p_tau:.4e})")
    print(f"    Spearman's rho = {rho:.4f} (p = {p_rho:.4e})")

# Interpretation guide:
# tau/rho ≈ 1.0 → rankings are nearly identical (metric choice doesn't matter)
# tau/rho < 0.8 → rankings differ substantially (metric choice matters a lot)
# tau/rho < 0.6 → rankings are quite different
```

**Expected finding:** Kendall's tau between Original and MCC-based rankings should be significantly below 1.0, indicating that metric choice substantially changes the ranking — especially for models in the "middle tier" where clean performance and robustness trade off.

### 3.6 Cell 10: Visualization

Produce a scatter plot for the paper:

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metric_pairs = [
    ("rank_score_original", "rank_score_f1", "F1-based"),
    ("rank_score_original", "rank_score_mcc", "MCC-based"),
    ("rank_score_original", "rank_score_harmonic", "Harmonic(F1, ADV+CTR)"),
]

for ax, (col_x, col_y, ylabel) in zip(axes, metric_pairs):
    x = lcld_leaderboard[col_x].values
    y = lcld_leaderboard[col_y].values

    # Color: red for degenerate, blue for normal
    colors = ["red" if d else "steelblue" for d in degenerate_mask]

    ax.scatter(x, y, c=colors, alpha=0.6, s=30)
    ax.plot([0, 70], [0, 70], "k--", alpha=0.3, label="y=x (no change)")
    ax.set_xlabel("Original Rank (Accuracy-based)")
    ax.set_ylabel(f"Re-Rank ({ylabel})")
    ax.set_title(ylabel)
    ax.set_xlim(0, 72)
    ax.set_ylim(0, 72)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend(fontsize=8)

    # Annotate degenerate models
    for idx in lcld_leaderboard[degenerate_mask].index:
        row = lcld_leaderboard.loc[idx]
        ax.annotate(
            f"{row['arch']}\n{row['aug']}",
            (row[col_x], row[col_y]),
            fontsize=6, color="red", alpha=0.8
        )

plt.tight_layout()
plt.savefig("rank_sensitivity_lcld.png", dpi=150, bbox_inches="tight")
plt.savefig("rank_sensitivity_lcld.pdf", bbox_inches="tight")
print("Saved: rank_sensitivity_lcld.png/pdf")
```

**What the figure shows:** Points on the diagonal = ranking unchanged. Points far from diagonal = ranking shifted. Red dots = degenerate models. The displacement pattern reveals how much the ranking depends on the clean performance metric.

### 3.7 Cell 10 (continued): Additional figure — ADV vs ADV+CTR gap by training type

```python
fig, ax = plt.subplots(figsize=(8, 6))

# Color by training type
color_map = {"Adv": "orangered", "Std": "steelblue"}
for training_type in ["Adv", "Std"]:
    subset = lcld_leaderboard[lcld_leaderboard["training"] == training_type]
    ax.scatter(
        subset["adv"], subset["adv_ctr"],
        c=color_map[training_type], alpha=0.6, s=40,
        label=f"{training_type} training"
    )

ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="ADV = ADV+CTR")
ax.set_xlabel("ADV (Unconstrained Adversarial Recall %)")
ax.set_ylabel("ADV+CTR (Constrained Adversarial Recall %)")
ax.set_title("LCLD: Effect of Constraint Filtering on Attack Success")
ax.legend()
plt.savefig("adv_vs_advctr_lcld.png", dpi=150, bbox_inches="tight")
```

**What this shows:** Std-trained models cluster near the x-axis (high ADV but near-zero ADV+CTR → constraints reject most attacks). Adv-trained models cluster near the diagonal (ADV ≈ ADV+CTR → attacks are already constraint-aware). This is a powerful visual for the paper.

---

## 4. Implementation Notes

### 4.1 Notebook cell dependency graph

```
Cell 1 (Setup + install)                ← Run once, then restart runtime
  └─→ Cell 2 (Load + re-train + predict) ← Needs GPU, ~30s; symlinks datasets
        └─→ Cell 3 (Inverse-transform + constraints)  ← Needs preprocessor from Cell 2
              └─→ Cell 4 (ADV / ADV+CTR)      ← Core Direction 3
                    └─→ Cell 5 (Comparison table)
Cell 6 (Leaderboard data)               ← No dependency on Cells 1-5
  ├─→ Cell 7 (Degenerate ID)
  └─→ Cell 8 (Re-ranking)
        ├─→ Cell 9 (Rank correlation)
        └─→ Cell 10 (Visualization)     ← Also needs Cell 7 (degenerate_mask)
Cell 11 (Summary)                        ← Depends on all above
Cell 12 (Save)                           ← Depends on all above
```

Direction 3 (Cells 1-5) and Direction 7 (Cells 6-10) are **independent** — they can be developed and tested in parallel. Cells 11-12 combine both.

### 4.2 Data entry strategy for leaderboard (Cell 6)

The LCLD leaderboard has 70 rows. Recommend:

```python
# Option: parse from the scraped HTML string (already fetched)
# The leaderboard table has consistent format:
# | Rank | Architecture | Training | Augmentation | ID | ADV+CTR | ADV | AUC | Accuracy | Precision | Recall | MCC |

# Hardcode as list of dicts for reliability:
lcld_data = [
    # Top entries (most important for analysis)
    (1, "STG", "Adv", "CT-GAN", 82.00, 81.16, 82.00, 0.7002, 50.50, 26.61, 83.28, 0.2118),
    (2, "TabTransformer", "Adv", "CT-GAN", 79.50, 78.50, 79.50, 0.6945, 52.63, 27.09, 80.27, 0.2119),
    # ... continue for all 70 rows ...
]

columns = ["rank", "arch", "training", "aug", "id_recall", "adv_ctr", "adv",
           "auc", "accuracy", "precision", "recall", "mcc"]
lcld_leaderboard = pd.DataFrame(lcld_data, columns=columns)
```

### 4.3 Functions to copy into Cell 3

All of the following must be defined in Cell 3 of the new notebook. Copy their implementations from `tabularbench_comparison.ipynb` Cells 11-13:

**From original Cell 11 (inverse-transform):**
- `inverse_transform_numeric(X_proc, num_feature_names, scaler)` — recovers raw-space numeric values

**From original Cell 12 (derived features + constraint checks):**
- `add_derived_from_base(df_inv, derived_template)` — recomputes ratio features from perturbed base
- `reconstruct_term_from_ohe(X_proc)` — reconstructs term from one-hot columns
- `_to_float(series)` — coerces to float
- `check_g1_installment(df, tol)` — installment formula constraint
- `check_g2_open_total(df)` — open_acc <= total_acc
- `check_g3_bankruptcies(df)` — bankruptcies <= pub_rec
- `check_g4_processed(X_proc)` — OHE validity in processed space
- `check_g4_term(df)` — raw-space term in {36, 60}
- `check_g5_ratio_loan_inc(df)` — derived ratio constraint
- `check_g6_ratio_open_total(df)` — derived ratio constraint

**From original Cell 13 (aggregate):**
- `compute_aggregate_feasibility(df, X_proc=None)` — runs g1-g4 jointly

**Globals:**
- `G1_TOL = 0.1` — installment tolerance (10x TabularBench's 0.01)
- `TOLERANCE = 0.01` — general constraint tolerance

See Section 2.5 for the complete Cell 3 pseudocode showing how these fit together.

### 4.4 Key variables to verify

After Cell 2 (re-training), verify clean metrics match original output:

| Variable | Expected (3-seed avg) | Tolerance |
|---|---|---|
| clean_pr_auc | 0.3015 | ±0.01 (training variance) |
| clean_accuracy | 0.6419 | ±0.01 |

**Note:** We cannot verify robust_pr_auc or robust_accuracy because the new notebook does not re-run the attack — it loads pre-saved adversarial parquets. The re-trained model will produce *different* predictions on the adversarial examples than the original model did, which is expected and acceptable. The ADV/ADV+CTR analysis only needs internally consistent predictions from the *same* model across clean and adversarial inputs.

If clean metrics deviate wildly (e.g., clean_pr_auc < 0.20), investigate whether the dataset symlinks and preprocessing are correct.

### 4.5 Threshold sensitivity check (optional but recommended)

TabularBench uses argmax (for multiclass) or 0.5 threshold (for binary). FraudBench also uses 0.5. But fraud detection in practice often uses optimized thresholds. Add a brief sensitivity check:

```python
for threshold in [0.3, 0.4, 0.5, 0.6]:
    y_pred = (y_probs_unmasked >= threshold).astype(int)
    recall_pos = y_pred[pos_mask].mean()
    print(f"  threshold={threshold}: ADV (unconstrained recall) = {recall_pos:.4f}")
```

This shows that the extreme ADV value (~0.04%) is not an artifact of threshold choice.

---

## 5. Expected Outputs and Deliverables

### 5.1 Tables for paper

1. **Table: Cross-Benchmark ADV/ADV+CTR Comparison** (Direction 3, Cell 5)
2. **Table: Ranking Sensitivity to Metric Choice** (Direction 7, top-10 under each formula)
3. **Table: Rank Correlation Coefficients** (Direction 7, Cell 9)

### 5.2 Figures for paper

1. **Figure: Rank Shift Scatter Plot** (Direction 7, Cell 10) — 3-panel, original rank vs re-rank
2. **Figure: ADV vs ADV+CTR by Training Type** (Direction 7, Cell 10) — scatter with diagonal

### 5.3 Key numbers for text

- FraudBench ADV on positive class: ~0.04% (attack nearly 100% successful)
- FraudBench ADV+CTR: ~55.7% (constraint filtering restores nearly all predictions)
- Gap magnitude: ~55.7 percentage points — largest among all LCLD entries
- Feasibility-filtered attack success rate: ~0% of successful attacks survive constraints
- Rank correlation (Original vs MCC): expected τ < 0.85
- Number of degenerate models on LCLD: at least 1 (TabNet Adv GOGGLE)

### 5.4 Files saved to Google Drive

```
/content/drive/MyDrive/FraudBench/results/metric_analysis/
  ├── adv_advctr_comparison.csv          # Direction 3 results
  ├── lcld_leaderboard_reranked.csv      # Direction 7 full table
  ├── rank_correlation_results.csv       # Direction 7 statistics
  ├── rank_sensitivity_lcld.png          # Direction 7 figure
  ├── rank_sensitivity_lcld.pdf          # Direction 7 figure (vector)
  ├── adv_vs_advctr_lcld.png             # Direction 7 figure
  └── metric_analysis_summary.txt        # Combined narrative
```

---

## 6. Risk Mitigation

| Risk | Mitigation |
|---|---|
| Model re-training gives different results | Load saved adversarial parquets + re-predict only; OR use saved metric CSVs for aggregate-level analysis. GPU non-determinism means exact metric reproduction is unlikely — verify within tolerance (see Section 4.4), and note that ADV/ADV+CTR analysis only requires consistent predictions from the *same* re-trained model |
| n_pos too small for meaningful ADV+CTR | LCLD has ~20% positive rate × 26820 ≈ 5388 positives — sufficient |
| Leaderboard data entry errors | Cross-check totals (70 rows) and spot-check rank ordering. Consider parsing from scraped HTML programmatically rather than manual entry |
| Reviewer questions "why not run TabularBench's CAA directly?" | Address in paper: Python version incompatibility (TB requires 3.8.10); direct metric comparison is valid without identical attack because we compare the GAP pattern, not absolute values |
| Reviewer says "this is just re-analyzing someone else's leaderboard" | Frame as: "metric sensitivity analysis to justify FraudBench's design choice of PR-AUC" — the analysis serves OUR benchmark's positioning |

---

## 7. Timeline

| Task | Estimated time | Dependency |
|---|---|---|
| Cell 1-3: Setup + re-train + inverse-transform | 45 min | GPU access |
| Cell 4-5: ADV/ADV+CTR computation + comparison table | 1.5 hours | Cell 3 |
| Cell 6: Leaderboard data entry | 45 min | None |
| Cell 7-9: Degenerate + re-ranking + correlation | 2 hours | Cell 6 |
| Cell 10: Visualization | 1 hour | Cells 7 + 8 |
| Cell 11-12: Summary + save | 30 min | All above |
| Review + polish | 1 hour | All above |
| **Total** | **~7.5 hours** | |

Cells 1-5 and Cells 6-10 can be parallelized (Direction 3 needs GPU; Direction 7 is pure analysis).
