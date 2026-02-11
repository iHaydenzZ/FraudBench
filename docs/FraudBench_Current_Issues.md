# FraudBench: Current Issues & Gap Analysis

> **Date:** 2026-02-12 (updated)
> **Registry snapshot:** 60 rows x 22 columns, 20/28 unique configs executed, 4 datasets, 3 seeds each
> **Purpose:** Identify remaining work to complete FraudBench — a benchmark for comparing the effectiveness of adversarial training and defence methods on financial fraud detection models.

---

## 1. Current State Summary

FraudBench compares the effectiveness of adversarial training and input validation defences against adversarial attacks on fraud detection models across 4 financial datasets (CCFD, IEEE-CIS, LCLD, Sparkov), 2 model families (Neural MLP, XGBoost), and 3 random seeds (42, 123, 456).

The registry contains 60 experiment rows from CAPGD (white-box) attacks. Phase 4 added HopSkipJump and Square Attack (black-box) implementations and 8 config files, but these experiments have **not yet been run** — no black-box results exist in the registry. Phase 5 added automated figure generation (`scripts/generate_figures.py`).

**Key research findings so far:**
- Adversarial training significantly improves robustness for Neural models across all datasets
- XGBoost is immune to gradient-based CAPGD (expected — tree models lack gradients)
- Input validation consistently degrades robustness (analysed below — likely a genuine finding)

**Critical remaining gap:** Tree models have never been successfully attacked. Until the black-box experiments (HopSkipJump, Square) are run, we cannot compare defence effectiveness on XGBoost.

---

## 2. Critical Bugs (P0) — Must Fix Before Any Submission

### 2.1 IEEE-CIS Constraint Validity Rate = 0.0000

**Impact:** Undermines the credibility of the constraint validation mechanism for 30/60 registry rows.

All 30 IEEE-CIS experiment rows report `validity_rate = 0.0000`. Every clean sample fails constraint validation, which is incorrect — clean data should trivially satisfy domain constraints.

**Root cause (confirmed):** The bug is in `constraints/validator.py:validate_sample()`. For categorical features, the validator checks `if val not in constraint.allowed_values`. When `val` is NaN (which is common in IEEE-CIS — many features have high missing rates), this check **always fails** because `NaN != NaN` in Python, so `NaN not in [...]` is always `True`.

The `has_missing` flag is correctly tracked in `ConstraintSchema` but **never used** during validation. A single NaN in any categorical column causes the entire sample to fail.

**Evidence:** IEEE-CIS has missing values across many feature groups (D1-D15, M1-M9, V1-V339, id_* columns). With 10% sampling, virtually every test sample has at least one NaN in a categorical column.

**Fix:** In `ConstraintValidator.validate_sample()`, skip NaN values when the feature's `has_missing` flag is `True`:

```python
if pd.isna(val):
    if constraint.has_missing:
        continue  # NaN is acceptable for features with known missing values
    else:
        return False
```

**Definition of Done:** All IEEE-CIS clean samples achieve `validity_rate >= 0.95`. Re-run all IEEE-CIS experiments.

---

### 2.2 Input Validation Consistently Degrades Robustness

**Impact:** This is a central finding for the thesis — it directly answers the research question about defence effectiveness. Must be correctly analysed and documented.

Input validation makes robustness **worse** in every configuration tested:

| Dataset | Model | Baseline Robust PR-AUC | Input Val Robust PR-AUC | Change |
|---|---|---|---|---|
| CCFD | Neural | 0.5961 | 0.1545 | **-74.1%** |
| CCFD | Tree | 0.8511 | 0.7047 | -17.2% |
| IEEE-CIS | Neural | 0.0776 | 0.0185 | **-76.2%** |
| IEEE-CIS | Tree | 0.5660 | 0.5378 | -5.0% |
| LCLD | Neural | 0.1051 | 0.1051 | 0.0% |
| LCLD | Tree | 0.3683 | 0.3669 | -0.4% |
| Sparkov | Neural | 0.0056 | 0.0053 | -5.4% |
| Sparkov | Tree | 0.7467 | 0.2315 | **-69.0%** |

**Analysis (from code review of `defences/input_validation.py`):**

The `transform()` method applies two layers of clipping to **all** numeric features in the processed space:
1. **Bound clipping** to `[min_val, max_val]` from the constraint schema
2. **Z-score outlier clipping** to `[mean - 3*std, mean + 3*std]` after `fit()` on training data

This is **likely a genuine finding, not a bug**, for two reasons:

1. **CAPGD already respects domain constraints.** The adversarial examples are "valid" within schema bounds, so bound clipping has no effect. The z-score clipping is what causes damage — it clips adversarial perturbations that happen to fall outside the 3-sigma range, but in doing so destroys discriminative signal that the model relies on.

2. **Neural models are affected far more than tree models.** Neural decision boundaries depend on precise feature values, so clipping distorts predictions. Tree models use discrete split thresholds and are more tolerant of clipping. Exception: Sparkov Tree shows -69% — this may indicate the z-score bounds are too tight for this dataset's distribution.

**Conclusion:** Input validation as implemented is an **ineffective defence** against constrained adversarial attacks. This is a valid research finding that contrasts with adversarial training's effectiveness and should be analysed in the thesis Discussion section.

**Remaining action:** Verify this interpretation by running a diagnostic — compute the fraction of features that get clipped by the z-score bounds on adversarial vs clean examples. If the clipped fraction is similar for both, the degradation is from information loss, not a bug.

**Definition of Done:** Document the finding with evidence in the thesis. Optionally test with different `z_threshold` values (e.g., 5.0, 10.0) to show the effect of tighter vs looser clipping.

---

## 3. Benchmark Gaps (P1) — Required for "Benchmark" Status

### 3.1 Run Black-Box Attack Experiments (24 new experiments)

**Impact:** Without these results, we cannot compare defence effectiveness on tree models — a critical gap in the benchmark's coverage.

Phase 4 implemented HopSkipJump and Square Attack code + 8 config files, but **no experiments have been run**. All tree model rows in the registry show identical clean/robust metrics because CAPGD was skipped.

**Required action:**
1. Run all 8 black-box configs x 3 seeds = 24 experiments via `scripts/run_all_seeds.py`
2. Verify tree models now show actual robustness degradation (robust != clean)
3. Generate updated figures with `scripts/generate_figures.py`

**Definition of Done:** Registry contains 84 rows (60 existing + 24 new). Tree models show measurable robustness degradation under black-box attacks.

---

### 3.2 Epsilon Sweeps for Robustness Curves

**Impact:** A benchmark with a single perturbation budget (epsilon=0.1) is incomplete. Robustness curves are the standard output of adversarial robustness benchmarks.

The runner supports `epsilon_values` lists, and `configs/ccfd_eps_sweep.yaml` exists as a template, but only CCFD has been swept and those results may not be in the main registry.

**Required action:**
1. Create epsilon sweep configs for all 4 datasets (at minimum: neural + no defence)
2. Use `epsilon_values: [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]`
3. Add a robustness curve plot to `scripts/generate_figures.py` (Robust PR-AUC vs epsilon, one line per defence)

**Definition of Done:** Registry contains multi-epsilon results for at least 2 datasets; robustness curve figure is auto-generated.

---

### 3.3 Reproducibility Documentation

**Impact:** Reviewers and other researchers must be able to reproduce all results from the README alone.

**Required:**
1. **Dataset download instructions** — exact URLs and placement for all 4 datasets
2. **Full reproduction command** — document `scripts/run_all_seeds.py` in README
3. **Config documentation** — explain all YAML fields with examples, including new attack types (`hopskipjump`, `square`)
4. **README update** — add HopSkipJump, Square Attack, LCLD, and Sparkov to the README (currently only documents CCFD and IEEE-CIS)

**Definition of Done:** A new user can clone the repo and reproduce all registry results following only the README.

---

### 3.4 Tree + Adversarial Training Gap (4 missing configs)

**Impact:** 4 out of 28 possible configurations cannot be run — XGBoost + adversarial training across all 4 datasets.

**Root cause:** Adversarial training uses CAPGD's inner loop for PGD augmentation, which requires gradients. XGBoost does not expose gradients. The runner already raises `ValueError` for this combination.

**Recommendation:** Document as a finding — gradient-based adversarial training is **architecturally inapplicable** to tree models. This is itself a meaningful result: it shows that the most effective defence (adversarial training) is model-family-dependent, which has practical implications for fraud detection system design.

Mark these 4 cells as "N/A" in the results table with justification.

**Definition of Done:** Benchmark documentation explicitly addresses this gap with justification.

---

## 4. Enhancements (P2) — Strengthen the Benchmark

### 4.1 Cross-Model Transferability Experiments

Test whether adversarial examples generated on Neural MLP can fool XGBoost (and vice versa with black-box attacks) on the same dataset. This addresses a key question: **do defence strategies need to be attack-type-specific?**

Directly relevant to the thesis: if adversarial training on Neural models does not protect against transferred attacks on Tree models, the defence comparison is more nuanced.

**Estimated effort:** 3-5 hours.

---

### 4.2 Statistical Significance Testing

The 3-seed setup enables paired t-tests or Wilcoxon signed-rank tests for pairwise defence comparisons. This would strengthen claims like "adversarial training significantly outperforms input validation" with p-values.

**Estimated effort:** 1-2 hours.

---

### 4.3 Pre-Trained Model Zoo

Save and share trained model weights (both baseline and adversarially trained) so other researchers can evaluate new attacks or defences without retraining.

**Estimated effort:** 2-4 hours.

---

## 5. Resolved Items (from previous version)

| Item | Status | Resolution |
|---|---|---|
| Black-box attack code (HopSkipJump + Square) | **Resolved** | `attacks/hopskipjump.py`, `attacks/square.py`, 8 config files created. Experiments pending (see 3.1). |
| Automated figure generation | **Resolved** | `scripts/generate_figures.py` — 5 figures: robustness bars, attack comparison, summary table, defence heatmap, training time. |
| Dataset cards | **Resolved** | All 4 cards exist in `datasets/cards/` (ccfd.md, ieee_cis.md, lcld.md, sparkov.md). |

---

## 6. Priority Summary

| Priority | Task | Est. Hours | Impact |
|---|---|---|---|
| P0 | Fix IEEE-CIS validity_rate = 0 (NaN handling in validator) | 1h | Constraint mechanism credibility |
| P0 | Analyse Input Validation degradation (document as finding) | 2h | Core thesis finding |
| P1 | Run 24 black-box experiments (HSJ + Square on tree models) | 2-4h | Enables tree model defence comparison |
| P1 | Epsilon sweeps for robustness curves | 2-3h | Standard benchmark output |
| P1 | README + reproduction documentation | 2-3h | Reproducibility |
| P1 | Document tree + adv_training gap | 0.5h | Coverage completeness |
| P2 | Transferability experiments | 3-5h | Research finding |
| P2 | Statistical significance tests | 1-2h | Academic rigour |
| P2 | Pre-trained model zoo | 2-4h | Community reusability |
| | **Total (P0 + P1)** | **~10-14h** | **Minimum for submission** |
| | **Total (all)** | **~16-24h** | **Full benchmark suite** |

---

## 7. Current Registry Statistics

- **Total rows:** 60
- **Datasets:** CCFD, IEEE-CIS, LCLD, Sparkov
- **Models:** Neural MLP, XGBoost
- **Attacks executed:** CAPGD only (HopSkipJump + Square code ready but not yet run)
- **Defences:** none, adversarial_training, input_validation
- **Seeds:** 42, 123, 456
- **Configs executed:** 20 / 28 (71%) — excludes 4 tree+adv_train (N/A) and 4 tree+black-box (pending)
- **Key findings:**
  - CAPGD degrades neural models: CCFD -18.5%, IEEE-CIS -82.7%, LCLD -65.4%, Sparkov -99.1%
  - XGBoost immune to CAPGD (gradient-free — clean = robust)
  - Adversarial training effective for neural: CCFD +7%, IEEE-CIS +254%, LCLD +184%, Sparkov +3742%
  - Input validation ineffective: never improves robustness (worst: -76%), likely a genuine finding
  - IEEE-CIS validity_rate = 0 across all experiments (NaN handling bug)
  - LCLD and Sparkov neural clean PR-AUC are modest (~0.30 and ~0.62)
