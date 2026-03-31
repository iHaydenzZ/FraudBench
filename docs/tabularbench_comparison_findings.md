# TabularBench Comparison Findings

**Notebooks**:
- `notebooks/tabularbench_comparison.ipynb` — Tiers 1–2 (mutability mask, feasibility rate)
- `notebooks/tabularbench_metric_analysis.ipynb` — Directions 3 & 7 (ADV/ADV+CTR gap, ranking sensitivity)

**Dataset**: LCLD (Lending Club Loan Default), 134,097 samples (10% subsample)
**Model**: Neural network (128-dim hidden, 20 epochs, class-weighted)
**Attack**: CAPGD, eps=0.1, 10 steps
**Seeds**: 42, 123, 456

---

## Tier 2: Mutability Mask

We classified 52 LCLD features into 41 immutable (LC-internal, credit-bureau) and 11 mutable (borrower-controlled). After OHE expansion: 65 immutable, ~123 mutable of 188 processed features.

### Robust Metrics (mean +/- std over 3 seeds)

| Metric | No Mask | With Mask | Delta |
|--------|---------|-----------|-------|
| Clean PR-AUC | 0.3015 +/- 0.0021 | 0.3015 +/- 0.0021 | 0 |
| Robust PR-AUC | 0.1051 +/- 0.0001 | 0.1051 +/- 0.0001 | ~0 |
| Clean Accuracy | 0.6419 +/- 0.0057 | 0.6419 +/- 0.0057 | 0 |
| **Robust Accuracy** | **0.0545 +/- 0.0187** | **0.1721 +/- 0.0291** | **+0.1177** |
| Clean Recall | 0.5573 +/- 0.0193 | 0.5573 +/- 0.0193 | 0 |
| Robust Recall | 0.0004 +/- 0.0002 | 0.0017 +/- 0.0003 | +0.0013 |

**Takeaway**: The +11.8pp robust accuracy delta means the original CAPGD attack relied substantially on perturbing features an attacker cannot control (credit bureau data, LC-internal pricing). The masked metrics are the more trustworthy measure of real-world vulnerability.

### Perturbation Statistics (constrained features)

| Feature | No Mask: Mean \|delta\| | No Mask: % Changed | Masked: Mean \|delta\| | Masked: % Changed |
|---------|------------------------|--------------------|----------------------|-------------------|
| loan_amnt | 763.22 | 99.2% | 789.01 | 99.4% |
| int_rate | 0.42 | 99.4% | **0.00** | **0.0%** |
| installment | 23.82 | 99.5% | **0.00** | **0.0%** |
| open_acc | 0.48 | 99.3% | **0.00** | **0.0%** |
| total_acc | 1.04 | 99.2% | **0.00** | **0.0%** |
| pub_rec | 0.04 | 70.9% | **0.00** | **0.0%** |
| pub_rec_bankruptcies | 0.02 | 57.5% | **0.00** | **0.0%** |
| annual_inc | 7515.32 | 99.8% | 7685.13 | 99.8% |

The mask correctly freezes all immutable features at zero perturbation. Mutable features (loan_amnt, annual_inc) are perturbed slightly more under the mask as CAPGD compensates.

---

## Tier 1: Feasibility Rate

### g1 Tolerance Calibration

TabularBench uses tol=$0.01. We ran a sweep on clean data:

| Tolerance | Clean Satisfaction |
|-----------|-------------------|
| $0.01 | 0.9944 |
| $0.05 | 0.9979 |
| **$0.10** | **0.9980** |
| $0.50 | 0.9980 |
| $1.00 | 0.9980 |

The jump is between 0.01 and 0.05 (0.9944 -> 0.9979), suggesting ~0.36% of clean samples have inverse-transform rounding error between $0.01-$0.05. We use **tol=$0.10** -- minimal further gain above $0.05 but safely accounts for StandardScaler precision loss. The previous notebook used $1.00 which was unnecessarily loose.

### Per-Constraint Feasibility Rates

| Constraint | Clean | Adv (no mask) | Adv (masked) |
|------------|-------|---------------|--------------|
| **g1 (installment formula, tol=0.1)** | 0.9980 | **0.0201** | **0.0087** |
| g2 (open_acc <= total_acc) | 1.0000 | 0.9844 | 1.0000 |
| g3 (bankruptcies <= pub_rec) | 0.9988 | 0.7792 | 1.0000 |
| **g4 (term OHE valid, proc space)** | -- | **0.1934** | **0.2218** |
| g4 (term in {36,60}, argmax) | 1.0000 | 1.0000 | 1.0000 |
| g5 (ratio_loan/inc) | 0.9998 | (trivial) | (trivial) |
| g6 (ratio_open/total) | 1.0000 | (trivial) | (trivial) |

#### Key findings per constraint:

**g1 (installment formula)** -- The most discriminating constraint. Only ~2% of unmasked and ~0.9% of masked adversarial examples preserve the financial relationship `installment = loan_amnt * r(1+r)^t / ((1+r)^t - 1)`. This is expected: CAPGD perturbs loan_amnt, int_rate, installment, and term independently, breaking their mathematical coupling.

**g3 (bankruptcies <= pub_rec)** -- The previous run's most violated constraint. 22% of unmasked adversarial examples have more bankruptcies than public records. The mask eliminates this entirely (both features are immutable).

**g4 (term OHE)** -- The argmax-based check always passes (it just picks the larger OHE column), masking the real problem: CAPGD pushes OHE columns to continuous values, breaking the discrete {36, 60} encoding. The processed-space OHE validity check reveals ~80% of adversarial examples have invalid term encoding. Notably, even the masked attack breaks term OHE (~78% changed), because **term is classified as mutable** (borrower chooses their loan term). CAPGD turns the discrete choice into a continuous value, which is unrealistic.

**g5/g6 (derived ratios)** -- Trivially satisfied for adversarial examples because the ratios are recomputed from the same perturbed base features. They provide no additional constraint power.

### Aggregate Feasibility (g1 + g2 + g3 + g4-OHE)

| Dataset | Feasibility Rate | Passing / Total |
|---------|-----------------|-----------------|
| Clean test set | **0.9968** | 26,735 / 26,820 |
| Adversarial (no mask) | **0.0014** | 37 / 26,820 |
| Adversarial (with mask) | **0.0022** | 59 / 26,820 |

**This is the headline result.** With all four constraints properly checked (including g1 and g4 OHE validity), essentially **zero** adversarial examples are domain-feasible. The previous notebook reported 81.9% feasibility for unmasked adversarial examples, but that was only checking g2 + g3 (with g1/g4 as N/A).

The masked attack does not improve aggregate feasibility because the binding constraints are g1 (installment formula couples mutable+immutable features) and g4 (CAPGD can't preserve discrete OHE encoding for mutable term).

### Term OHE Perturbation (processed space)

| Variant | Max OHE Delta | Rows Changed |
|---------|--------------|--------------|
| No mask | 0.1000 | 80.7% |
| Masked | 0.1000 | 77.9% |

Both variants perturb term OHE up to the full epsilon budget (0.1). Since term is mutable, the mask does not protect it. The max delta of exactly 0.1 means CAPGD saturates the epsilon ball for term columns.

---

## Validation

- **Inverse transform**: Max error = 0.000000, Mean error = 0.000000. Perfect reconstruction.
- **Per-row alignment**: 0/26,820 rows with max error > 0.01 between inverse-transformed and original raw features. Index alignment is correct.
- **Term reconstruction**: Both masked and unmasked produce identical term distributions {36: 20,272, 60: 6,548}, matching the original test split.

---

## Direction 3: ADV vs ADV+CTR Gap Analysis

**Notebook**: `notebooks/tabularbench_metric_analysis.ipynb`, Cells 1–5 (GPU required)

Maps FraudBench's LCLD results into TabularBench's ADV / ADV+CTR evaluation framework. ADV measures recall under unconstrained adversarial attack; ADV+CTR measures recall after filtering out constraint-violating adversarial examples.

### Core Metrics

| Metric | Value |
|--------|-------|
| Clean Recall (ID) | 0.5542 (55.42%) |
| ADV (unmasked) | 0.0027 (0.27%) |
| ADV (masked) | 0.0088 (0.88%) |
| ADV+CTR (unmasked) | 0.5538 (55.38%) |
| ADV+CTR (masked) | 0.5536 (55.36%) |
| **Gap unmasked (ADV+CTR − ADV)** | **+55.11 pp** |
| **Gap masked (ADV+CTR − ADV)** | **+54.48 pp** |

### Cross-Benchmark Comparison

| Framework | Model | Training | ID Recall | ADV | ADV+CTR | Gap |
|-----------|-------|----------|-----------|-----|---------|-----|
| TabularBench | STG | Adv+CTGAN | 83.28% | 82.00% | 81.16% | −0.84 pp |
| TabularBench | TabTransformer | Adv+CTGAN | 80.27% | 79.50% | 78.50% | −1.00 pp |
| TabularBench | TabTransformer | Adv+CutMix | 72.33% | 72.50% | 70.96% | −1.54 pp |
| TabularBench | RLN | Adv+None | 69.28% | 69.48% | 63.04% | −6.44 pp |
| TabularBench | STG | Std+None | 65.99% | 66.40% | 53.60% | −12.80 pp |
| TabularBench | RLN | Std+None | 68.51% | 68.30% | 0.02% | −68.28 pp |
| TabularBench | VIME | Std+None | 67.13% | 67.00% | 2.38% | −64.62 pp |
| **FraudBench** | **MLP** | **Std+None** | **55.42%** | **0.27%** | **55.38%** | **+55.11 pp** |
| **FraudBench** | **MLP** | **Std+Mask** | **55.42%** | **0.88%** | **55.36%** | **+54.48 pp** |

**FraudBench produces the largest gap on the LCLD leaderboard.** TabularBench's constraint-aware CAA shows gaps of 0.8–12.8 pp for adversarially-trained models. FraudBench's +55.11 pp gap indicates that CAPGD generates attacks that are nearly 100% effective at flipping predictions but >99.8% infeasible under domain constraints.

### Filtered Attack Success Rate (positive class, predictions flipped 1→0)

| Variant | Successful Attacks | Also Feasible | Filtered Success Rate |
|---------|-------------------|---------------|----------------------|
| Unmasked | 2,897 | 2 | 0.07% |
| Masked | 2,865 | 3 | 0.10% |

### Per-Constraint Failure on Flipped Positives (unmasked, n=2,897)

| Constraint | Failure Rate | Pass Rate |
|------------|-------------|-----------|
| g1 (installment) | 99.3% | 0.7% |
| g2 (open≤total) | 0.8% | 99.2% |
| g3 (bk≤pub_rec) | 57.2% | 42.8% |
| g4 (OHE valid) | 82.5% | 17.5% |

### Threshold Sensitivity

| Threshold | ADV Recall |
|-----------|-----------|
| 0.3 | 0.0038 |
| 0.4 | 0.0032 |
| 0.5 | 0.0027 |
| 0.6 | 0.0021 |

Near-zero ADV recall is stable across thresholds — the result is not an artifact of threshold choice.

---

## Direction 7: Ranking Sensitivity Analysis

**Notebook**: `notebooks/tabularbench_metric_analysis.ipynb`, Cells 6–10 (pure analysis)

Re-ranks TabularBench's 70-model LCLD leaderboard under alternative metrics to demonstrate that accuracy-based ranking is misleading for imbalanced fraud detection (20% positive class).

### Degenerate Model Identification

10 of 70 models are degenerate (MCC ≈ 0, trivial all-positive or all-negative prediction). All are TabNet variants. Most notable: TabNet adversarial+goggle achieves **100% ADV+CTR with only 20% accuracy** (predicts all-positive).

### Re-Ranking Formulas

1. **Original (TabularBench):** (Accuracy + ADV+CTR) / 2
2. **F1-based:** (F1 + ADV+CTR) / 2
3. **MCC-based:** (MCC_norm + ADV+CTR) / 2, where MCC_norm = (MCC + 1) / 2 × 100
4. **AUC-based:** (AUC + ADV+CTR) / 2
5. **Harmonic mean:** 2 × F1 × ADV+CTR / (F1 + ADV+CTR)

### Top-10 Ranking Comparison

| Rank | Arch | Training | Aug | Original | F1 | MCC | Harmonic |
|------|------|----------|-----|----------|----|-----|----------|
| 1 | STG | adversarial | ctgan | 1 | 2 | 2 | 1 |
| 2 | TabTransformer | adversarial | ctgan | 2 | 3 | 3 | 3 |
| 3 | TabTransformer | adversarial | cutmix | 3 | 5 | 5 | 4 |
| 4 | VIME | adversarial | ctgan | 4 | 4 | 4 | 2 |
| 5 | STG | adversarial | tvae | 5 | 7 | 7 | 6 |
| 6 | STG | adversarial | goggle | 6 | 8 | 8 | 7 |
| 7 | TabTransformer | adversarial | None | 7 | 6 | 6 | 5 |
| 8 | VIME | adversarial | tvae | 8 | 10 | 10 | 8 |
| 9 | TabTransformer | adversarial | tvae | 9 | 9 | 9 | 9 |
| 10 | RLN | adversarial | tvae | 10 | 11 | 11 | 10 |

### Rank Correlation Analysis

| Ranking Pair | Kendall's τ | p-value | Spearman's ρ | p-value |
|--------------|-------------|---------|--------------|---------|
| Original vs F1 | 0.663 | 7.6e-16 | 0.793 | 3.0e-16 |
| Original vs MCC | 0.683 | 1.1e-16 | 0.812 | 1.5e-17 |
| Original vs AUC | 0.681 | 9.7e-17 | 0.822 | 2.6e-18 |
| Original vs Harmonic | 0.735 | 7.0e-19 | 0.863 | 7.8e-22 |

All alternatives show Spearman's ρ < 0.9 and Kendall's τ < 0.8, indicating **substantial ranking differences**. Accuracy-based ranking rewards degenerate models and does not correlate strongly with metrics appropriate for imbalanced fraud detection (F1, MCC).

### Generated Figures

- `rank_sensitivity_lcld.png/pdf` — 3-panel scatter plot of rank shifts (degenerate models in red)
- `adv_vs_advctr_lcld.png` — ADV vs ADV+CTR scatter by training type

---

## Implications

1. **FraudBench's CAPGD generates ~99.8% infeasible adversarial examples** when checked against TabularBench's domain constraints. The bounds-only attack (feature-wise min/max clipping) is insufficient for producing realistic adversarial loans.

2. **g1 (installment formula) is the dominant constraint**, failing for 99.3% of successful adversarial attacks. This is a strong signal that constraint-aware attacks (like TabularBench's) are necessary for meaningful robustness evaluation on financial data.

3. **g4 (term discreteness) is the second binding constraint.** CAPGD cannot preserve one-hot encoding by design -- it operates in continuous space. This affects both masked and unmasked attacks equally since term is mutable.

4. **The mutability mask improves robustness metrics (+11.8pp accuracy)** but does not improve aggregate feasibility. The mask addresses a different problem (which features can be perturbed) than domain constraints (which perturbation values are valid). Both are necessary for realistic evaluation.

5. **FraudBench produces the largest ADV/ADV+CTR gap (+55.11 pp) on the LCLD leaderboard.** Post-hoc constraint filtering restores predictions to near-clean levels, whereas TabularBench's constraint-aware CAA shows gaps of only 0.8–12.8 pp for adversarially-trained models. This demonstrates that constraint-unaware attacks vastly overstate model vulnerability.

6. **Accuracy-based ranking is misleading for fraud detection.** Alternative metrics (F1, MCC) produce substantially different rankings (Kendall's τ < 0.74). TabularBench's leaderboard contains 10 degenerate models that accuracy-based scoring does not penalize. This motivates FraudBench's use of PR-AUC as the primary evaluation metric.

7. **Future work**: Integrate constraint-aware attacks from TabularBench into FraudBench. The two benchmarks are complementary -- FraudBench contributes tree models, black-box attacks, and PR-AUC; TabularBench contributes domain constraints and constraint-aware optimization.
