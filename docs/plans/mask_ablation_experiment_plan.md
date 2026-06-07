# Mask Ablation Experiment Plan

**Objective**: Explore how different perturbation masks affect adversarial robustness and domain feasibility on LCLD, producing results for the next advisor meeting.

**Notebook to create**: `notebooks/mask_ablation.ipynb`
**Reference notebook**: `notebooks/tabularbench_comparison.ipynb` (copy its setup cells 1–5 verbatim)

---

## 1. Environment & Infrastructure

### 1.1 Colab Setup (identical to `tabularbench_comparison.ipynb`)

- **Runtime**: GPU (A100 preferred)
- **Google Drive mount**: `/content/drive/MyDrive/FraudBench/`
- **Repo clone**: `https://github.com/iHaydenzZ/FraudBench.git` → `/content/FraudBench`
- **Dataset symlink**: `datasets/LCLD` → Google Drive `data/LCLD/`
- **Dependencies**: Same `pip install` block as reference notebook Cell 4, restart runtime after install
- Copy Cells 1–5 from `tabularbench_comparison.ipynb` exactly. No changes needed.

### 1.2 Shared Constants

```
DATASET       = "lcld"
SAMPLE_FRAC   = 0.1
SEEDS         = [42, 123, 456]
EPSILON       = 0.1
STEPS         = 10
MODEL_PARAMS  = {"epochs": 20, "hidden_dim": 128, "batch_size": 256, "lr": 0.001}
G1_TOL        = 0.10
```

### 1.3 Output Directory

All adversarial example parquets → `results/adv_examples/mask_ablation/`
All summary CSVs → same directory
Backup to Google Drive at end: `/content/drive/MyDrive/FraudBench/results/mask_ablation/`

---

## 2. Experiment Variants

### 2.0 M0 — No Mask (baseline, no rerun needed)

**Source**: Already completed in `tabularbench_comparison.ipynb`.
**Action**: Load existing results from `results/adv_examples/comparison_unmasked.csv` and the saved parquets (`lcld_neural_unmasked_seed{42,123,456}.parquet`). These parquets are also backed up on Google Drive at `/content/drive/MyDrive/FraudBench/results/adv_examples/`.

### 2.1 M1 — Binary Mutability Mask (baseline, no rerun needed)

**Source**: Already completed in `tabularbench_comparison.ipynb`.
**Action**: Load existing results from `results/adv_examples/comparison_masked.csv` and the saved parquets (`lcld_neural_masked_seed{42,123,456}.parquet`).

### 2.2 M3 — M1 + Derived Feature Freeze (dti/dti_joint)

**Background**: The original M3 idea was to freeze `installment`, but checking the reference notebook Cell 6 shows `installment` is already in `LCLD_IMMUTABLE_RAW`. So freezing it again has zero effect. Instead, M3 freezes `dti` and `dti_joint` — features that are currently classified as mutable but are in fact server-computed (DTI = total monthly debt payments / gross monthly income, calculated by the platform).

**Immutable set modification**:
```
LCLD_IMMUTABLE_M3 = LCLD_IMMUTABLE_RAW | {"dti", "dti_joint"}
```

**Attack function**: Use `capgd_attack_masked()` from the reference notebook. Only the immutable set changes.

**File naming**: `lcld_neural_M3_seed{seed}.parquet`

### 2.3 M4 — M1 + Categorical Freeze (Term OHE)

**What changes**: Freeze the term OHE columns in processed space. Rationale: CAPGD operates in continuous space and pushes OHE columns to non-binary values (~80% of adversarial examples have invalid term encoding in existing results). Since CAPGD fundamentally cannot produce valid discrete OHE perturbations, freezing term is the honest evaluation.

**Immutable set modification**:
```
LCLD_IMMUTABLE_M4 = LCLD_IMMUTABLE_RAW | {"term"}
```

This works because `build_processed_mutable_mask()` matches the raw feature name `term` to all OHE columns prefixed with `term_` in processed space.

**File naming**: `lcld_neural_M4_seed{seed}.parquet`

### 2.4 M5 — M1 + M3 + M4 Combined

**What changes**: Freeze `dti`, `dti_joint`, and `term` simultaneously.
```
LCLD_IMMUTABLE_M5 = LCLD_IMMUTABLE_RAW | {"dti", "dti_joint", "term"}
```

**Significance**: This is the "most realistic" perturbation space achievable without modifying the attack algorithm. It addresses the dominant feasibility failures: g4 OHE invalidity (~80% failure) is eliminated by freezing term, and dti is no longer independently perturbed.

If aggregate feasibility jumps significantly under M5, the conclusion is that mask-level fixes suffice. If feasibility remains near zero (because g1 installment formula still couples loan_amnt and int_rate), the conclusion is that constraint-aware attacks are necessary — pointing toward TabularBench integration.

**File naming**: `lcld_neural_M5_seed{seed}.parquet`

### 2.5 M2 — M1 + Directionality Mask

**What changes**: For mutable features with monotonic real-world constraints, enforce one-sided perturbation.

**Directional constraints on LCLD mutable features**:

| Raw Feature       | Direction      | Rationale                                            |
|-------------------|----------------|------------------------------------------------------|
| `loan_amnt`       | bidirectional  | Borrower freely chooses amount                       |
| `term`            | categorical    | Not applicable to directionality                     |
| `purpose`         | categorical    | Not applicable                                       |
| `emp_length`      | increase-only  | Employment length cannot shrink at application time  |
| `annual_inc`      | bidirectional  | Could report higher or lower                         |
| `annual_inc_joint`| bidirectional  | Same reasoning                                       |
| `home_ownership`  | categorical    | Not applicable                                       |
| `dti`             | bidirectional  | Changes with income/debt                             |
| `dti_joint`       | bidirectional  | Same reasoning                                       |
| `application_type`| categorical    | Not applicable                                       |
| `addr_state`      | categorical    | Not applicable                                       |

**Only `emp_length` has a clear increase-only constraint** among the 11 mutable features. M2's incremental effect is expected to be small. This is itself a reportable finding.

**Implementation**: Modify `capgd_attack_masked()` to add a directional clip after the standard epsilon-ball projection. For `emp_length` columns in processed space, enforce `x_adv[:, i] >= x_orig[:, i]` (perturbation delta ≥ 0 in processed space).

**Caveat about processed space**: StandardScaler transforms as `x_scaled = (x - mean) / std`. Since std > 0, increasing in raw space = increasing in scaled space. The clip `x_adv[:, i] = max(x_adv[:, i], x_orig[:, i])` in processed space is correct for increase-only.

**File naming**: `lcld_neural_M2_seed{seed}.parquet`

### 2.6 M6 — Tiered Actionability (Strict vs Relaxed)

**What changes**: Define two attacker profiles by varying which features are mutable.

**Strict profile** (low-capability attacker — can only set fields directly on the application form at submission time):
```
MUTABLE_STRICT = {"loan_amnt", "purpose", "home_ownership", "application_type", "addr_state"}
```
Excludes: `annual_inc` (verification possible), `emp_length` (verifiable), `dti`/`dti_joint` (server-computed), `term` (OHE issue), `annual_inc_joint`.

**Relaxed profile** (medium-capability attacker — can also self-report unverified fields):
```
MUTABLE_RELAXED = {"loan_amnt", "purpose", "home_ownership", "application_type",
                   "addr_state", "annual_inc", "annual_inc_joint", "emp_length"}
```
Adds self-reported income and employment length, but still excludes `dti` (server-computed) and `term` (OHE issue).

**Computing immutable sets**: For each profile, `IMMUTABLE = ALL_RAW_FEATURES - MUTABLE_PROFILE`. To get `ALL_RAW_FEATURES`, collect all column names from the raw dataset before preprocessing.

**Expected mutable feature counts in processed space** (approximate, depends on OHE expansion):
- Strict: ~30–40 (mostly categorical OHE columns from purpose, home_ownership, addr_state, application_type + loan_amnt)
- Relaxed: ~35–45 (adds annual_inc, annual_inc_joint, emp_length)
- M1: ~123 (current baseline)

**File naming**: `lcld_neural_M6strict_seed{seed}.parquet`, `lcld_neural_M6relaxed_seed{seed}.parquet`

### 2.7 E1 — Cost-Weighted Evaluation (no rerun, analysis only)

**What changes**: No new attack runs. Re-analyze existing M0 and M1 adversarial examples by assigning per-feature modification costs.

**Cost assignments** (normalized units, based on domain reasoning):

| Raw Feature          | Cost  | Rationale                                           |
|----------------------|-------|-----------------------------------------------------|
| `loan_amnt`          | 1.0   | Freely chosen by borrower                           |
| `purpose`            | 0.5   | Trivial to change on application                    |
| `home_ownership`     | 3.0   | Requires actual change or fraud                     |
| `addr_state`         | 2.0   | Requires relocation or address fraud                |
| `application_type`   | 1.0   | Choice between individual/joint                     |
| `annual_inc`         | 8.0   | Requires fake income documentation                  |
| `annual_inc_joint`   | 8.0   | Same                                                |
| `emp_length`         | 5.0   | Accumulated over time, hard to fake                 |
| `dti`                | 7.0   | Derived from financials, hard to control            |
| `dti_joint`          | 7.0   | Same                                                |
| `term`               | 1.0   | Borrower's choice                                   |

**Computation** (on inverse-transformed adversarial examples, raw space):
```
For each adversarial example:
    total_cost = sum over mutable features f:
        COST[f] * |x_adv_raw[f] - x_orig_raw[f]| / feature_range[f]
```
where `feature_range[f] = max(f) - min(f)` in the training set, to normalize deltas across features with different scales.

**Outputs**:
1. Histogram: distribution of total_cost across all adversarial examples (separate curves for M0 and M1)
2. "Affordable attack" curve: for each cost budget B on x-axis, what fraction of adversarial examples have total_cost ≤ B on y-axis?
3. Table: mean / median / p95 total cost for M0 vs M1
4. Cost sensitivity check: repeat with costs ×2 and ×0.5 to demonstrate conclusion stability

---

## 3. Pipeline for Each Attack Variant (M2–M6)

Each variant follows the same loop as reference notebook Cell 8. The **only change** between variants is the immutable set (and for M2, the directional clip).

### Optimization: share model training across variants

All variants use the same model per seed (the mask affects only the attack, not training). Train the model once per seed, then run all attack variants on it:

```
for seed in SEEDS:
    load data, preprocess, train model (once)
    evaluate clean metrics (once)
    
    for variant_name, immutable_set in ALL_VARIANTS.items():
        build mask from immutable_set
        run CAPGD with mask
        evaluate robust metrics
        save adversarial parquet
```

This saves ~5× training time.

### Critical: reuse existing preprocessor

Load the preprocessor from `results/preprocessor_lcld_n134097_seed{seed}.joblib` (already saved by the reference notebook). Do NOT refit. This ensures feature scaling is identical across all variants and consistent with M0/M1 baseline.

---

## 4. Feasibility Audit (for all variants)

After generating adversarial examples for all variants, run the full feasibility audit on each.

### What to check:

1. **Inverse-transform** adversarial examples to raw space using the saved preprocessor's StandardScaler parameters
2. **Reconstruct term** from OHE via argmax (same as reference notebook)
3. **Per-constraint pass rates**: g1 (installment formula, tol=0.10), g2 (open_acc ≤ total_acc), g3 (bankruptcies ≤ pub_rec), g4 (OHE validity check in processed space)
4. **Aggregate feasibility rate**: fraction passing ALL of g1–g4 simultaneously
5. **Perturbation statistics**: mean |delta|, max |delta|, % changed for key features

### Functions to reuse from reference notebook:

- `inverse_transform_numeric()` — Cell 11
- `check_g1_installment()`, `check_g2_open_total()`, `check_g3_bankruptcies()`, `check_g4_term()`, `check_g4_processed()` — Cell 12
- `compute_aggregate_feasibility()` — Cell 13
- `reconstruct_term_from_ohe()` — Cell 12

Copy these functions into the new notebook (or factor into a shared utility if preferred).

### Feasibility audit applies only to seed=42

To keep runtime manageable, run the full feasibility audit (inverse transform + constraint checks) on seed=42 only. Robust metrics (PR-AUC, accuracy) are computed for all 3 seeds.

---

## 5. Output Specification

### 5.1 Primary Results Table

One CSV (`mask_ablation_results.csv`) with columns:

```
variant, seed, n_mutable, n_immutable,
clean_pr_auc, clean_accuracy, clean_recall, clean_f1,
robust_pr_auc, robust_accuracy, robust_recall, robust_f1
```

### 5.2 Summary Table (mean ± std over 3 seeds)

Printed at end of notebook and saved as `mask_ablation_summary.csv`:

| Variant | Mutable | Robust PR-AUC | Robust Acc | Feasibility | g1 Pass | g4 Pass |
|---------|---------|---------------|------------|-------------|---------|---------|
| M0 | all | 0.1051±0.0001 | 0.0545±0.019 | 0.0014 | 0.020 | 0.193 |
| M1 | ~123 | 0.1051±0.0001 | 0.1721±0.029 | 0.0022 | 0.009 | 0.222 |
| M2 | ~122 | ? | ? | ? | ? | ? |
| M3 | ~121 | ? | ? | ? | ? | ? |
| M4 | ~121 | ? | ? | ? | ? | ? |
| M5 | ~119 | ? | ? | ? | ? | ? |
| M6-strict | ~35 | ? | ? | ? | ? | ? |
| M6-relaxed | ~43 | ? | ? | ? | ? | ? |

(Exact mutable counts depend on OHE expansion per seed.)

### 5.3 Feasibility Detail Table (seed=42 only)

One row per variant, columns for each constraint's pass rate:

| Variant | g1 Pass | g2 Pass | g3 Pass | g4 OHE Pass | Aggregate |
|---------|---------|---------|---------|-------------|-----------|

### 5.4 Perturbation Statistics Table (seed=42 only)

For each variant, report per-feature: mean |delta|, % changed.
Key features: `loan_amnt`, `annual_inc`, `dti`, `emp_length`, `int_rate`, `installment`, plus term OHE max delta.

### 5.5 E1 Cost-Weighted Outputs

- `e1_cost_distribution.png` — histogram
- `e1_affordable_curve.png` — budget vs fraction affordable
- `e1_cost_summary.csv` — mean, median, p95

### 5.6 Backup to Google Drive

At end of notebook, copy everything in `results/adv_examples/mask_ablation/` to:
```
/content/drive/MyDrive/FraudBench/results/mask_ablation/
```

---

## 6. Notebook Cell Structure

```
Cell 1:   Verify GPU                                    [copy from ref]
Cell 2:   Mount Google Drive                            [copy from ref]
Cell 3:   Clone/update repo                             [copy from ref]
Cell 4:   Install dependencies                          [copy from ref]
Cell 5:   Symlink datasets                              [copy from ref]
Cell 6:   Define all mask variant configurations        [NEW]
            - LCLD_IMMUTABLE_RAW (from ref Cell 6)
            - LCLD_IMMUTABLE_M3 = LCLD_IMMUTABLE_RAW | {"dti", "dti_joint"}
            - LCLD_IMMUTABLE_M4 = LCLD_IMMUTABLE_RAW | {"term"}
            - LCLD_IMMUTABLE_M5 = LCLD_IMMUTABLE_RAW | {"dti", "dti_joint", "term"}
            - MUTABLE_STRICT, MUTABLE_RELAXED sets
            - DIRECTION_CONSTRAINTS = {"emp_length": "increase"}
            - FEATURE_COSTS dict for E1
            - VARIANTS dict mapping variant name → immutable set
Cell 7:   Define utility functions                      [adapt from ref Cell 7]
            - build_processed_mutable_mask()
            - capgd_attack_masked()
            - capgd_attack_masked_directional() for M2
Cell 8:   Main experiment loop                          [NEW]
            - Outer: seeds
            - Train model once per seed
            - Inner: variants
            - Save parquets + collect metrics
Cell 9:   Print robust metric comparison table          [NEW]
Cell 10:  Feasibility audit functions                   [copy from ref Cells 10-12]
            - inverse_transform_numeric()
            - check_g1_installment(), check_g2, check_g3, check_g4
            - reconstruct_term_from_ohe()
            - compute_aggregate_feasibility()
Cell 11:  Run feasibility audit on all variants         [NEW, seed=42 only]
Cell 12:  Perturbation statistics                       [NEW]
Cell 13:  E1 Cost-weighted evaluation                   [NEW]
Cell 14:  Final summary + save to Google Drive          [adapt from ref Cell 15]
```

---

## 7. Expected Runtime

| Step | Per-seed | Total (3 seeds) |
|------|----------|-----------------|
| Data load + preprocess | ~10s | ~30s |
| Model training (once per seed) | ~25s | ~75s |
| CAPGD per variant | ~0.2s | ~0.6s |
| 6 new variants × CAPGD per seed | ~1.2s | ~3.6s |
| Feasibility audit (all variants, seed=42 only) | ~60s | ~60s |
| E1 cost analysis | — | ~10s |
| **Total** | — | **~4 min** |

---

## 8. Known Issues & Mitigations

### 8.1 `installment` is already immutable in M1

Verified from reference notebook Cell 6. Original M3 idea (freeze installment) would have zero delta. Redefined M3 to freeze `dti`/`dti_joint` instead.

### 8.2 M2 directional effect likely negligible

Only `emp_length` is increase-only among 11 mutable features. If M2 ≈ M1, report: "Directionality constraints have minimal impact on LCLD because most directionally-constrained features are already classified as immutable."

### 8.3 M6-strict may render CAPGD near-powerless

With only ~5 raw mutable features, CAPGD may not find effective perturbations. Robust metrics may approach clean metrics. This is a valid finding: a realistic low-capability attacker has limited evasion ability on LCLD.

### 8.4 Processed feature count varies by seed

Reference notebook shows 188 features for seed 42 but 187 for seed 123. `build_processed_mutable_mask()` handles this by name matching. Report per-seed mutable/immutable counts.

### 8.5 M6 immutable set construction

For M6 variants, the immutable set is ALL features minus the mutable set. To get ALL features, use the raw feature names from `dataset.X.columns` (before preprocessing). Then:
```
IMMUTABLE_M6_STRICT = set(dataset.X.columns) - MUTABLE_STRICT
IMMUTABLE_M6_RELAXED = set(dataset.X.columns) - MUTABLE_RELAXED
```

---

## 9. Meeting Discussion Points

1. **Which mask layer has the biggest marginal effect on robust accuracy?** Compare M1 vs M4 vs M5 vs M6-strict.
2. **Does freezing term (M4) significantly improve g4 feasibility?** Expected: g4 pass rate → 100%. Does aggregate feasibility also jump?
3. **How sensitive is the threat assessment to attacker capability?** M6-strict vs M6-relaxed vs M1 shows the spectrum.
4. **What does cost-weighted analysis (E1) reveal?** Are most "successful" attacks cheap or expensive for the attacker?
5. **What should the next experiment round focus on?** Constraint-aware attacks? Cross-dataset? More mask granularity?
