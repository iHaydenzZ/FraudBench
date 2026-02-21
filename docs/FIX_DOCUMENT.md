# FraudBench — Figure & Analysis Fix Document

> **Date:** 2026-02-21  
> **Registry:** `registry_clean.csv` (182 rows, 35 configurations)  
> **Status:** MVP data complete; visualisation and analysis artefacts require fixes before supervisor presentation.

---

## Executive Summary

Six figures and three analysis CSVs have been generated from the experiment registry. While the underlying data is sound (182 runs, no validity issues, all 4 datasets covered), several artefacts have quality issues that would undermine the benchmark narrative if presented as-is. This document catalogues each issue, explains its impact, and provides actionable fix instructions.

**Estimated total fix effort:** 3–4 hours

---

## Issue 1: Robustness Curves Show Only `none` Defence

**Priority:** P0 — Critical  
**File:** `robustness_curves.png`  
**Script:** `scripts/generate_figures.py → plot_robustness_curves()`

### Problem

The epsilon sweep figure displays a single blue line (`none`) per dataset. The `adversarial_training` and `input_validation` lines are missing, even though the registry contains epsilon sweep data for these defences at ε ∈ {0.01, 0.05, 0.1, 0.15, 0.2, 0.3}.

### Impact

This is the **most important figure for the defence comparison narrative**. Without overlaid defence curves, the reader cannot see how defences perform across perturbation budgets — the core research question. The figure currently adds no value beyond showing baseline degradation.

### Root Cause

The `plot_robustness_curves()` function filters for configurations that have data at multiple epsilon values. Adversarial training and input validation experiments only exist at ε = 0.1 for most dataset–model combinations, so they get excluded by the multi-epsilon filter.

### Fix

**Option A (Recommended):** Relax the filter — plot any defence that has ≥ 2 epsilon data points, and also include single-point defences as scatter markers on the same axes. This lets the reader compare the defence's ε = 0.1 value against the baseline curve.

**Option B:** Run epsilon sweep experiments for adversarial training and input validation (ε = {0.01, 0.05, 0.1, 0.15, 0.2, 0.3}). This requires additional GPU time but produces the cleanest figure.

Modify `plot_robustness_curves()`:

```python
# Current: only plots defences with multi-epsilon data
# Fix: overlay single-epsilon defence results as markers

for defence in defences:
    d = sub[sub["defence_type"] == defence].sort_values("attack_epsilon")
    if len(d) >= 2:
        # Line plot for multi-epsilon data
        ax.errorbar(d["attack_epsilon"], d["robust_pr_auc_mean"],
                     yerr=d["robust_pr_auc_std"].fillna(0),
                     marker="o", capsize=3, label=defence)
    elif len(d) == 1:
        # Single-point marker for defences with only ε=0.1
        ax.scatter(d["attack_epsilon"], d["robust_pr_auc_mean"],
                   marker="^", s=100, zorder=5, label=f"{defence} (ε=0.1)")
```

Additionally, filter to **neural model only** for this figure, since CAPGD has no effect on tree models and would produce flat curves.

### Acceptance Criteria

- [ ] Figure shows ≥ 2 curves/markers per dataset subplot (baseline + at least one defence)
- [ ] Legend clearly distinguishes `none`, `adversarial_training`, and `input_validation`
- [ ] Y-axis range is consistent across subplots (0.0–1.0)

---

## Issue 2: Defence Heatmap Missing Ensemble Column

**Priority:** P1 — High  
**File:** `defence_heatmap.png`  
**Script:** `scripts/generate_figures.py → plot_defence_heatmap()`

### Problem

The heatmap displays only two columns: `adversarial_training` and `input_validation`. The `ensemble` defence (24 runs in the registry) is absent, making the defence comparison incomplete.

### Impact

The thesis title promises a "Comparative Study of Defence Mechanisms." Presenting a 2-column heatmap when 3 defence methods were evaluated weakens the comparative claim.

### Root Cause

The heatmap is computed as `delta = defence_robust_pr_auc − baseline_robust_pr_auc`, where the baseline is `defence_type == "none"` for the **same model_type**. Since the ensemble model has `model_type = "ensemble"` and `defence_type = "ensemble"`, there is no matching `none` baseline for model_type `"ensemble"`.

### Fix

The ensemble defence should be compared against the **neural `none` baseline**, as the ensemble model includes a neural sub-model and represents an alternative architecture-level defence. Modify the delta computation:

```python
# For ensemble defence: compare against neural+none baseline
ensemble_rows = agg[(agg["model_type"] == "ensemble") & (agg["defence_type"] == "ensemble")]
neural_baseline = agg[(agg["model_type"] == "neural") & (agg["defence_type"] == "none")]

# Merge on dataset + attack_type + attack_epsilon to compute delta
```

Alternatively, compute two deltas for ensemble: one against neural baseline and one against tree baseline, and show both (or the more conservative one).

### Acceptance Criteria

- [ ] Heatmap shows 3 columns: `adversarial_training`, `input_validation`, `ensemble`
- [ ] Ensemble delta is clearly labelled with its comparison baseline (e.g., "vs neural baseline")
- [ ] Colour scale accommodates the new column's value range

---

## Issue 3: Statistical Tests — Ensemble Comparisons All Fail

**Priority:** P1 — High  
**File:** `statistical_tests.csv`  
**Script:** `scripts/statistical_tests.py`

### Problem

Every row involving `ensemble` as either `defence_a` or `defence_b` shows `significant = False` with the note `"insufficient paired data (n=0)"`. This means 0 out of 24 ensemble comparisons produced valid statistical results.

### Impact

Without statistical backing, the ensemble defence results cannot be claimed as significant improvements or degradations. This is a gap in the benchmark's rigour.

### Root Cause

The script performs **paired comparisons within the same model_type**. Since `ensemble` is its own model_type and has no `defence_type = "none"` rows, the pairing logic finds zero matched pairs.

### Fix

Modify the pairing logic in `scripts/statistical_tests.py` to support **cross-model comparisons** for the ensemble defence. The ensemble should be compared against:

1. `model_type = "neural"`, `defence_type = "none"` (same attack, same dataset, same epsilon)
2. `model_type = "tree"`, `defence_type = "none"` (same attack, same dataset, same epsilon)

```python
# Add cross-model pairs for ensemble
if model_type == "ensemble":
    # Compare ensemble against neural baseline
    neural_none = df[(df["model_type"] == "neural") & (df["defence_type"] == "none")]
    # Pair by (dataset, attack_type, attack_epsilon, seed)
    paired = ensemble_data.merge(neural_none,
                                  on=["dataset", "attack_type", "attack_epsilon", "seed"],
                                  suffixes=("_ens", "_base"))
```

Also ensure that the Wilcoxon test minimum sample size requirement (n ≥ 6) is met. With 3 seeds × 1 epsilon = 3 pairs per dataset, the paired t-test is applicable but Wilcoxon may lack power. Document this limitation.

### Acceptance Criteria

- [ ] Ensemble vs neural-baseline comparisons produce valid p-values for all 4 datasets
- [ ] Results table clearly indicates cross-model comparison (not within-model)
- [ ] A note column explains when Wilcoxon is inapplicable due to small sample size

---

## Issue 4: Robustness Bars — X-axis Label Overcrowding

**Priority:** P2 — Medium  
**File:** `robustness_bars.png`  
**Script:** `scripts/generate_figures.py → plot_robustness_bars()`

### Problem

Each of the 4 dataset subplots contains ~12 bars with labels like `none / capgd` repeated multiple times (once per epsilon level). Labels overlap severely and are unreadable at standard figure sizes.

### Impact

The figure is visually cluttered and makes it difficult to extract the key message: which defence–attack combination performs best on each dataset.

### Root Cause

The script plots **all rows** from the aggregated registry, including every epsilon value for the baseline. Since CAPGD has 6 epsilon levels × 3 defence variants, each subplot gets ~18 configurations.

### Fix

Filter to **ε = 0.1 only** for the bar chart, which is the canonical epsilon used throughout the benchmark. Epsilon sweep details belong in the robustness curves figure (Issue 1).

```python
# In plot_robustness_bars():
agg_filtered = agg[agg["attack_epsilon"] == 0.1]
```

Additionally, use `defence_type / attack_type` as the x-label (dropping the redundant epsilon) and group bars by model_type within each subplot using colour coding or hatching patterns.

### Acceptance Criteria

- [ ] Each subplot contains ≤ 8 bars (one per defence–attack–model combination at ε = 0.1)
- [ ] X-axis labels are readable without rotation greater than 45°
- [ ] Model types are distinguishable by colour or pattern

---

## Issue 5: Input Validation Analysis — Potential Epsilon Mixing

**Priority:** P2 — Medium  
**File:** `input_validation_analysis.csv`, `input_validation_analysis.png`  
**Script:** `scripts/analyse_input_validation.py`

### Problem

In `input_validation_analysis.csv`, the CCFD neural baseline shows `clean_prauc_baseline = 0.731` and `robust_prauc_baseline = 0.596`. However, the registry data for CCFD neural none at ε = 0.1 gives `robust_pr_auc ≈ 0.584`. The ~2% discrepancy suggests the analysis script may be averaging across multiple epsilon values rather than isolating ε = 0.1.

### Impact

Small numerical inconsistency. Unlikely to change conclusions but could raise questions during thesis examination about data hygiene.

### Root Cause

The `compute_degradation()` function in `analyse_input_validation.py` may not be filtering strictly to ε = 0.1 before computing the baseline averages. If it includes ε = 0.01 or 0.05 runs (which have higher robust PR-AUC), the average is inflated.

### Fix

Verify the epsilon filter in `analyse_input_validation.py`:

```python
# Ensure strict epsilon filtering
baseline = agg[
    (agg["defence_type"] == "none") &
    (agg["attack_epsilon"] == 0.1)  # Must be exact, not approximate
]
```

Also confirm that `input_validation` rows used for comparison are at ε = 0.1 only. Regenerate the CSV and PNG after the fix.

### Acceptance Criteria

- [ ] All baseline values in the CSV match the registry's ε = 0.1 aggregates within ±0.001
- [ ] Script includes an assertion or filter log confirming epsilon = 0.1 is used exclusively

---

## Issue 6: CAPGD Has No Effect on Tree Models (Documentation Gap)

**Priority:** P2 — Medium  
**File:** Not a figure bug — requires documentation in the thesis/README  
**Data:** 28 rows in registry where `robust_pr_auc == clean_pr_auc`

### Problem

CAPGD (a gradient-based white-box attack) has zero effect on XGBoost tree models across all 4 datasets and all epsilon values. This produces `robust_pr_auc == clean_pr_auc` in every tree + CAPGD row.

### Impact

Without explanation, a reader may interpret this as a bug or as evidence that tree models are perfectly robust. Neither is correct. This is an **architectural limitation** of gradient-based attacks on non-differentiable models.

### Fix

Add a clear callout in the results section and in the summary table:

```markdown
**Note:** CAPGD is a gradient-based attack and cannot compute meaningful
perturbations for non-differentiable models such as XGBoost. Tree model
robustness is evaluated exclusively via black-box attacks (HopSkipJump,
Square Attack). The identical clean and robust PR-AUC values for tree +
CAPGD configurations reflect this architectural limitation, not model
robustness.
```

Also consider adding a footnote symbol (†) to tree + CAPGD cells in the summary table PNG, with the footnote text explaining the limitation.

### Acceptance Criteria

- [ ] Summary table annotates tree + CAPGD rows as "N/A — gradient-based attack inapplicable"
- [ ] Thesis draft includes a paragraph explaining this limitation in the methodology section
- [ ] The limitation is mentioned in the README's "Known Limitations" section

---

## Issue 7: Sparkov + Neural Baseline Near-Zero Performance

**Priority:** P3 — Low (documentation, not a bug)  
**Data:** Sparkov neural none: `robust_pr_auc ≈ 0.005` across all seeds and epsilon values

### Problem

The Neural (MLP) model achieves near-zero robust PR-AUC on the Sparkov dataset, even at very small perturbation budgets (ε = 0.05). The clean PR-AUC itself is moderate (~0.6), but the model collapses entirely under CAPGD attack.

### Impact

This is not a bug but requires explanation. Without context, a reader might question the validity of the Sparkov experiments or the neural model implementation.

### Likely Explanation

Sparkov has only 22 features (fewest of all 4 datasets) and 1.3M samples. The low feature dimensionality may mean the MLP's decision boundary is less complex and more easily perturbed. The extremely low fraud rate (0.58%) combined with PR-AUC evaluation amplifies even small prediction shifts.

### Fix

Document in the results analysis:

```markdown
The Neural model's near-zero robust PR-AUC on Sparkov reflects extreme
vulnerability to gradient-based perturbation in a low-dimensional feature
space (22 features). This contrasts with the tree model's robust PR-AUC
of 0.747 on the same dataset, suggesting that XGBoost's piecewise-constant
decision surface provides inherent robustness in low-dimensional domains.
This finding supports the hypothesis that model architecture significantly
impacts adversarial robustness in fraud detection.
```

### Acceptance Criteria

- [ ] Results section includes a paragraph explaining the Sparkov–neural phenomenon
- [ ] The explanation connects to the broader research narrative (model architecture impacts robustness)

---

## Issue 8: HopSkipJump Incomplete Coverage

**Priority:** P3 — Low (deferred per MVP Supplement plan)  
**Data:** 6/12 runs complete (CCFD 3/3, IEEE-CIS 2/3, LCLD 1/3, Sparkov 0/3)

### Problem

HopSkipJump experiments are only partially complete. Missing runs: IEEE-CIS seed 456, LCLD seeds 123 and 456, Sparkov all 3 seeds.

### Impact

The incomplete coverage means HopSkipJump results cannot be reported with the same multi-seed statistical rigour as other attacks. However, Square Attack provides complete black-box coverage (4 datasets × 3 seeds), so the attack dimension is still well-covered.

### Decision

**Defer** per the MVP Supplement plan. The remaining 6 runs require ~36–60 hours of CPU time. Mention partial HSJ coverage as a limitation, and note that complete black-box evaluation is available via Square Attack.

### Acceptance Criteria

- [ ] Thesis clearly states HSJ coverage is partial and specifies which dataset–seed combinations are missing
- [ ] Square Attack is positioned as the primary black-box evaluation method
- [ ] HSJ completion is listed as future work

---

## Fix Execution Order

| Order | Issue | Priority | Estimated Time | Dependency |
|-------|-------|----------|----------------|------------|
| 1 | Issue 1: Robustness Curves add defence lines | P0 | 45 min | None |
| 2 | Issue 4: Robustness Bars filter to ε = 0.1 | P2 | 30 min | None |
| 3 | Issue 2: Heatmap add ensemble column | P1 | 30 min | None |
| 4 | Issue 3: Statistical tests fix ensemble pairing | P1 | 60 min | None |
| 5 | Issue 5: Input Validation epsilon filter check | P2 | 20 min | None |
| 6 | Issue 6: Document CAPGD–tree limitation | P2 | 20 min | None |
| 7 | Issue 7: Document Sparkov–neural finding | P3 | 15 min | None |
| 8 | Issue 8: Document HSJ partial coverage | P3 | 10 min | None |
| | **Total** | | **~3.5 hours** | |

---

## Regeneration Commands

After applying all code fixes, regenerate all artefacts:

```bash
# Regenerate all figures
uv run python scripts/generate_figures.py --registry results/registry_clean.csv --output results/figures

# Regenerate statistical tests
uv run python scripts/statistical_tests.py --registry results/registry_clean.csv

# Regenerate input validation analysis
uv run python scripts/analyse_input_validation.py --registry results/registry_clean.csv

# Verify outputs
ls -la results/figures/
```

### Post-Fix Verification Checklist

- [ ] `robustness_curves.png` shows multiple lines/markers per subplot
- [ ] `defence_heatmap.png` has 3 columns (adversarial_training, input_validation, ensemble)
- [ ] `robustness_bars.png` has readable x-axis labels (ε = 0.1 only)
- [ ] `statistical_tests.csv` has valid p-values for ensemble comparisons
- [ ] `input_validation_analysis.csv` values match registry at ε = 0.1
- [ ] All figures use consistent colour schemes across the figure set
- [ ] No figure has overlapping text or clipped labels
