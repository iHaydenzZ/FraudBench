# TabularBench Comparison Findings

**Notebook**: `notebooks/tabularbench_comparison.ipynb`
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

## Implications

1. **FraudBench's CAPGD generates ~99.8% infeasible adversarial examples** when checked against TabularBench's domain constraints. The bounds-only attack (feature-wise min/max clipping) is insufficient for producing realistic adversarial loans.

2. **g1 (installment formula) is the dominant constraint**, failing for 98% of adversarial examples. This is a strong signal that constraint-aware attacks (like TabularBench's) are necessary for meaningful robustness evaluation on financial data.

3. **g4 (term discreteness) is the second binding constraint.** CAPGD cannot preserve one-hot encoding by design -- it operates in continuous space. This affects both masked and unmasked attacks equally since term is mutable.

4. **The mutability mask improves robustness metrics (+11.8pp accuracy)** but does not improve aggregate feasibility. The mask addresses a different problem (which features can be perturbed) than domain constraints (which perturbation values are valid). Both are necessary for realistic evaluation.

5. **Future work**: Integrate constraint-aware attacks from TabularBench into FraudBench. The two benchmarks are complementary -- FraudBench contributes tree models, black-box attacks, and PR-AUC; TabularBench contributes domain constraints and constraint-aware optimization.
