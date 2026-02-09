# FraudBench: Remaining Work & Improvement Plan

> Last updated: 2026-02-09
>
> This document tracks all remaining experiments, code fixes, and improvements needed to
> bring FraudBench from its current MVP state to the full benchmarking suite described in
> the Progress Report.

---

## Current Status Summary

The MVP framework is structurally complete: config-driven runner, 2 dataset loaders
(CCFD + IEEE-CIS), 2 model wrappers (Neural MLP + XGBoost), CAPGD attack, 2 defence
stubs (adversarial training + input validation), evaluation metrics, results registry,
and 27 passing unit tests.

**Experiments completed so far:**

| Dataset | Model | Defence | Attack | Clean PR-AUC | Robust PR-AUC | In Registry? |
|---------|-------|---------|--------|--------------|---------------|--------------|
| CCFD (full) | Neural MLP | None | CAPGD ε=0.1 | 0.758 | 0.638 (-15.8%) | ❌ No |
| IEEE-CIS (10%) | XGBoost | None | N/A (tree) | 0.582 | N/A | ❌ No |
| dummy_dataset | Tree | None | CAPGD ε=0.1 | 0.971 | 0.971 | ✅ Yes |

**MVP Success Criteria checklist:**

| Criterion | Status |
|-----------|--------|
| CAPGD decreases fraud PR-AUC/Recall vs clean baseline | ✅ Demonstrated |
| At least one defence improves robust fraud metrics | ❌ No defence experiments run |
| Constraint validity rate reported and acceptable | ⚠️ Partial — only dummy data |
| Results registry complete and reproducible from config | ⚠️ Partial — real results missing from CSV |

---

## Part 1: MVP — Remaining Experiments & Fixes

Everything below must be completed before the MVP can be considered "passed".

### 1.1 Critical Code Fixes (P0)

#### 1.1.1 Adversarial Training Does Not Use CAPGD

**Problem:** `defences/adversarial_training.py` implements a plain 3-step PGD with L∞
clipping only. It does not call the CAPGD attack from `attacks/`, does not load any
constraint schema, and does not respect domain constraints. The Progress Report explicitly
states: "adversarial training using CAPGD" with "constraint-aware" perturbations.

**Required changes:**

- Import and reuse the `capgd_attack()` function (or a lighter training-time variant)
  from `attacks/` inside the adversarial training loop.
- Pass the per-dataset `ConstraintSchema` into the training step so generated adversarial
  examples respect feature bounds, non-negativity, and one-hot groups.
- If full CAPGD per mini-batch is too slow, implement a "CAPGD-lite" with fewer steps
  (e.g., 3–5 steps) but still apply constraint projection after each step.
- Update the `adversarial_train_step()` signature to accept `schema` and `feature_types`.

**Definition of Done:** Adversarially trained model produces adversarial examples during
training that pass `ConstraintValidator.validate()` at ≥90% validity rate.

#### 1.1.2 Input Validation Is Incomplete

**Problem:** `defences/input_validation.py` only clips numeric features to min/max. It
skips categorical/binary features entirely (`pass`). Missing features explicitly required
by Init_plan:

- One-hot validity fix (ensure one-hot groups sum to 1 with argmax projection)
- Outlier rejection rule (e.g., Z-score threshold or IQR-based rejection)
- No rejection mechanism at all — the current implementation only "corrects", never
  "detects and rejects"

**Required changes:**

- Implement one-hot repair: for each one-hot group, apply argmax to force exactly one
  active category.
- Add a simple outlier detection flag: compute per-feature mean/std from training data
  in `fit()`, then in `transform()` flag or reject samples with any feature exceeding
  ±k standard deviations (k=3 or 4 as a starting point).
- Add a `reject` mode vs `sanitise` mode, controlled by config, so experiments can
  compare both strategies.
- Return rejection metadata (how many samples rejected, which features triggered).

**Definition of Done:** Input validation handles numeric clipping, one-hot repair, and
outlier detection. Defence produces measurable changes under CAPGD attack.

#### 1.1.3 Results Registry Is Missing Real Experiment Data

**Problem:** `results/registry.csv` contains only one row from a `dummy_dataset` test run.
The CCFD and IEEE-CIS results reported in `README.md` are not in the registry. This
breaks the "reproducible from config" requirement.

**Required changes:**

- Re-run `configs/ccfd.yaml` and `configs/ieee_cis.yaml` and verify results are written
  to `results/registry.csv`.
- If there is a bug in `evaluation/registry.py` preventing writes, fix it.
- Confirm the registry columns match AGENTS.md spec (check for missing columns like
  `train_time_sec`, `attack_time_sec`, `adv_validity_rate`).

**Definition of Done:** Every experiment reported in README.md has a corresponding row in
`results/registry.csv` with all metric and cost columns populated.

---

### 1.2 MVP Experiments to Run (P0)

The MVP validation matrix requires 6 experiments on CCFD. Currently only experiment #1
is complete.

| # | Dataset | Model | Defence | Attack | Status | Priority |
|---|---------|-------|---------|--------|--------|----------|
| 1 | CCFD | Neural MLP | None | Clean + CAPGD | ✅ Done | — |
| 2 | CCFD | Neural MLP | Input Validation | Clean + CAPGD | ❌ TODO | 🔴 P0 |
| 3 | CCFD | Neural MLP | Adversarial Training | Clean + CAPGD | ❌ TODO | 🔴 P0 |
| 4 | CCFD | XGBoost | None | Clean only* | ❌ TODO | 🟡 P1 |
| 5 | CCFD | XGBoost | Input Validation | Clean only* | ❌ TODO | 🟡 P1 |
| 6 | CCFD | XGBoost | Adversarial Training | Clean only* | ❌ TODO | 🟡 P1 |
| 7 | IEEE-CIS | Neural MLP | None | Clean + CAPGD | ❌ TODO | 🟢 P2 |
| 8 | IEEE-CIS | XGBoost | None | Clean only | ⚠️ Partial | 🟢 P2 |

\* XGBoost does not support gradient-based attacks (CAPGD skipped). Record this as a
finding. Clean metrics must still be logged.

**Config files needed:**

- `configs/ccfd_input_val.yaml` — CCFD + Neural + `defence.type: input_validation`
- `configs/ccfd_adv_train.yaml` — CCFD + Neural + `defence.type: adversarial_training`
- `configs/ccfd_tree.yaml` — CCFD + XGBoost + no defence (tree baseline)
- `configs/ccfd_tree_input_val.yaml` — CCFD + XGBoost + input validation
- `configs/ccfd_tree_adv_train.yaml` — CCFD + XGBoost + adversarial training

**Expected outcomes:**

- **Experiment 2 (Input Validation):** Clean PR-AUC ≈ 0.75 (minimal drop).
  Robust PR-AUC should improve modestly from 0.638 to ~0.65–0.70. Low computational
  overhead.
- **Experiment 3 (Adversarial Training):** Clean PR-AUC expected to drop 5–15% (to
  ~0.65–0.72) due to robustness-accuracy tradeoff. Robust PR-AUC should improve
  significantly to ~0.70–0.75. Training time increase ~3–10×.
- **Experiments 4–6 (XGBoost):** Establishes tree-model baselines on CCFD. CAPGD cannot
  attack XGBoost, which is itself a noteworthy finding for the thesis. Input validation
  may still show measurable effect on clean metrics.

---

### 1.3 Additional MVP Improvements (P1)

#### 1.3.1 Test Multiple Epsilon Values

Currently only ε=0.1 is tested. Run at minimum 3 values to construct a robustness curve:

- ε ∈ {0.05, 0.1, 0.2} for CCFD Neural MLP (no defence)
- Plot: Robust PR-AUC vs ε

This is critical for the thesis — a single epsilon point is not a convincing benchmark.

#### 1.3.2 Add F1-Score to Registry

Init_plan requires "fraud precision/recall/**F1**" but the registry columns only include
`clean_pr_auc`, `clean_recall`, `robust_pr_auc`, `robust_recall`, `clean_accuracy`,
`robust_accuracy`. Missing:

- `clean_f1`, `robust_f1`
- `clean_precision`, `robust_precision`

Update `evaluation/metrics.py` and `evaluation/registry.py` to include these.

#### 1.3.3 Runner: Verify Defence Integration Path

Confirm the runner correctly handles all three defence modes end-to-end:

- `defence.type: none` → train normally, evaluate clean + robust ✅ (working)
- `defence.type: input_validation` → train normally, apply IV before inference on both
  clean and adversarial test sets ❓ (needs verification)
- `defence.type: adversarial_training` → train with adversarial examples, evaluate
  clean + robust ❓ (needs verification)

Run each config and verify the pipeline completes without errors and results are logged.

---

## Part 2: Post-MVP — Full Benchmark Experiments & Improvements

Everything below extends the MVP into the full benchmarking suite described in the
Progress Report. Implement only after MVP passes all success criteria.

### 2.1 Phase A: Expand Datasets (Progress Report Phase 1)

#### 2.1.1 Add LCLD Dataset

- Implement `datasets/loaders/lcld.py`
- Create `datasets/cards/lcld.md` (label meaning, imbalance rate, feature types, known
  leakage risks)
- Define `constraints/lcld.json` constraint schema (loan amount bounds, interest rate
  ranges, immutable features like employment length)
- Run full experiment matrix (Neural + XGBoost × all defences × all attacks)

#### 2.1.2 Add Sparkov Dataset

- Implement `datasets/loaders/sparkov.py`
- Create `datasets/cards/sparkov.md`
- Define `constraints/sparkov.json`
- Sparkov is synthetic — useful as a controlled experimental condition

#### 2.1.3 Dataset-Specific Constraints

Currently, `runner/run.py` infers constraints from processed data using
`fake_types = {c: 'numeric' for c in columns}`. This is acceptable for CCFD (all numeric
after PCA) but inadequate for IEEE-CIS, LCLD, and Sparkov which have categorical,
binary, and ordinal features.

- Create hand-crafted constraint schemas per dataset in `constraints/` directory
- Include: feature type map, per-feature min/max, non-negativity flags, one-hot groups,
  immutable feature list
- Update runner to load dataset-specific schemas instead of auto-inferring

---

### 2.2 Phase B: Add Black-Box Attacks (Progress Report Phase 3)

#### 2.2.1 HopSkipJump Attack

- Implement or integrate from ART: `art.attacks.HopSkipJump`
- This is a decision-based black-box attack requiring only hard labels
- **Key value:** Can attack XGBoost and other non-differentiable models, filling the gap
  where CAPGD cannot be used
- Configure query budget (track number of model queries)
- Add `attack.type: hopskipjump` to config schema with parameters: `max_iter`,
  `max_eval`, `init_eval`

**Expected results:** Lower attack success rate than CAPGD (less information available),
but demonstrates vulnerability of tree models. Higher computational cost (more queries).

#### 2.2.2 Square Attack

- Implement or integrate from ART: `art.attacks.SquareAttack`
- Score-based black-box attack requiring class probabilities
- Progress Report notes: "potentially focusing on one dataset as a case study due to its
  computational demands"
- Add `attack.type: square` to config schema with parameters: `max_iter`, `eps`,
  `norm` (L2 or Linf)

**Expected results:** Query efficiency between CAPGD and HopSkipJump. Effective against
gradient-masking defences.

#### 2.2.3 Attack Comparison Table

After implementing all 3 attacks, produce a comparison:

| Attack | Type | Info Required | Can Attack Trees? | Constraint-Aware? |
|--------|------|---------------|-------------------|-------------------|
| CAPGD | White-box | Gradients | ❌ No | ✅ Yes |
| HopSkipJump | Black-box (decision) | Hard labels | ✅ Yes | Needs adaptation |
| Square Attack | Black-box (score) | Probabilities | ✅ Yes | Needs adaptation |

---

### 2.3 Phase C: Add Remaining Defences (Progress Report Phase 4)

#### 2.3.1 Ensemble Methods

Progress Report specifies: "an ensemble of three models, such as Logistic Regression,
XGBoost, and a Neural Network" with voting or stacking.

Implementation plan:

- Add `models/logistic.py` wrapper with standard interface
- Implement `defences/ensemble.py` with:
  - Majority voting mode
  - Stacking mode (meta-learner on base model outputs)
  - Optional: ensemble adversarial training (Tramèr et al.)
- Add `defence.type: ensemble` to config with parameters: `base_models`, `strategy`
  (vote/stack)
- Record per-model and ensemble metrics separately

**Expected results:** Clean accuracy slightly improved (ensemble typically better than
single model). Robust accuracy improved moderately — harder to fool all models
simultaneously. Computational cost: ~3× single model for training and inference.

#### 2.3.2 CTGAN Data Augmentation

- Integrate `ctgan` library or `sdv` (Synthetic Data Vault)
- Train CTGAN on fraud-class samples to generate synthetic fraud data
- Augment training set with synthetic samples before model training
- Add `defence.type: ctgan_augmentation` to config with parameters:
  `n_synthetic_samples`, `epochs`, `batch_size`
- Progress Report suggests targeting the most imbalanced dataset (CCFD at 0.17%)

**Expected results:** Clean accuracy may slightly improve (richer training data). Robust
accuracy improves moderately (more diverse fraud patterns seen). Effect depends heavily
on synthetic data quality.

---

### 2.4 Phase D: Transferability Analysis (Progress Report Phase 5)

#### 2.4.1 Transferability Matrix

Test adversarial examples crafted on one model against all other models:

```
Source model (attack generated on) → Target model (evaluated on)

              Target: MLP   Target: XGBoost   Target: LR   Target: Ensemble
Source: MLP      —            ?%                ?%           ?%
Source: XGBoost  ?%           —                 ?%           ?%
Source: LR       ?%           ?%                —            ?%
```

- Generate adversarial test sets from each model
- Evaluate all other models on those adversarial test sets
- Report transfer success rate (percentage of adversarial examples that fool the target)

**Expected results per literature:** ~94% transferability from gradient-based attacks across
architectures (Foe for Fraud, 2025). If confirmed, this has major implications — attackers
only need access to one model type to compromise others.

#### 2.4.2 Cross-Defence Transferability

- Test adversarial examples crafted on undefended models against defended models
- E.g., CAPGD examples from baseline MLP → adversarially trained MLP
- This measures whether defences protect against "unseen" attacks not tailored to them

---

### 2.5 Phase E: Statistical Rigour

#### 2.5.1 Multi-Seed Experiments

- Run all experiments with seeds ∈ {42, 123, 456} (minimum 3)
- Report mean ± standard deviation for all metrics
- Use McNemar's test for pairwise classifier comparison
- Apply Bonferroni correction for multiple comparisons

#### 2.5.2 Robustness Curves

For each (dataset, model, defence) combination:

- Test ε ∈ {0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3}
- Plot: Robust PR-AUC vs ε for each defence on the same chart
- This is the most visually compelling figure for the thesis

#### 2.5.3 Ablation Studies

- Adversarial training: vary the clean-to-adversarial ratio (α ∈ {0.3, 0.5, 0.7})
- Adversarial training: vary PGD steps in training (3, 5, 10)
- Input validation: vary outlier rejection threshold (Z-score = 2, 3, 4)
- Ensemble: vary number of base models (2, 3, 5)

---

### 2.6 Phase F: Reporting & Visualisation

#### 2.6.1 Auto-Generated Benchmark Report

- Script to read `results/registry.csv` and produce:
  - Summary tables (one per dataset)
  - Robustness curves (Robust PR-AUC vs ε)
  - Defence comparison bar charts
  - Transferability heatmap
  - Computational cost comparison table
- Output as HTML or PDF

#### 2.6.2 Thesis-Ready Figures

- Defence comparison: grouped bar chart (clean vs robust PR-AUC per defence)
- Robustness-accuracy tradeoff: scatter plot (clean PR-AUC vs robust PR-AUC per
  defence, one point per seed)
- Cost-benefit: training time vs robust PR-AUC gain
- Constraint validity: violin plot of validity rates across attacks and datasets

---

### 2.7 Phase G: Code Quality & Release

#### 2.7.1 Test Coverage

- Add integration tests for each defence mode end-to-end
- Add tests for new attacks (HopSkipJump, Square)
- Add tests for new datasets (LCLD, Sparkov)
- Target: ≥50 tests covering all critical paths

#### 2.7.2 Documentation

- API docstrings for all public functions
- Usage examples in README for each attack/defence combination
- Installation guide with exact dependency versions

#### 2.7.3 Open-Source Release

- Clean up repository for public release
- Add LICENSE file
- Write CONTRIBUTING.md
- Create reproducibility instructions (scripts to regenerate all results from scratch)
- Optional: PyPI package `fraudbench`

---

## Full Experiment Matrix (Final Target)

The complete set of experiments described in the Progress Report:

**Dimensions:**
- **Datasets (4):** CCFD, IEEE-CIS, LCLD, Sparkov
- **Models (2+):** Neural MLP, XGBoost (+ LR for ensemble)
- **Defences (5):** None, Adversarial Training, Input Validation, Ensemble, CTGAN
- **Attacks (3):** CAPGD, HopSkipJump, Square Attack
- **Seeds (3):** 42, 123, 456

**Total experiments:** 4 datasets × 2 models × 5 defences × 3 attacks × 3 seeds = **360 runs**
(minus incompatible combinations like CAPGD on XGBoost ≈ ~300 valid runs)

**Metrics per run:**
- Clean: PR-AUC, Precision, Recall, F1, Accuracy
- Robust: PR-AUC, Precision, Recall, F1, Accuracy
- Validity: constraint validity rate, adversarial validity rate
- Cost: training time (sec), attack generation time (sec), inference time (ms/sample)

**Additional analyses:**
- Transferability matrix per dataset
- Robustness curves (8 epsilon values per combination)
- Ablation studies for adversarial training and input validation hyperparameters
