# FraudBench: To-Do List

> **Last updated:** 2026-02-16
> **Estimated remaining work:** P1 ~6-10h, All ~12-20h

---

## P0 -- Must Fix (Affects Result Credibility)

### ~~1. Re-run IEEE-CIS Experiments~~ -- DONE

Re-run completed on Feb 12-16. New IEEE-CIS rows show `validity_rate ~0.997`. Old rows (Feb 9-10, 15 rows with `validity_rate = 0.0000`) remain in registry but are superseded. Exclude old rows when computing final results.

---

### 2. Document Input Validation Finding

Input validation consistently degrades robustness. This is a genuine finding (not a bug) -- CAPGD already respects constraints, so the z-score clipping destroys discriminative signal without blocking adversarial perturbations.

**Action:** Write up in thesis Discussion section with evidence (fraction of features clipped). Optionally test with different `z_threshold` values (5.0, 10.0) to show the effect.

---

## P1 -- Required for Benchmark Status

### 3. Run Black-Box Attack Experiments (24 experiments)

Tree models (XGBoost) are immune to gradient-based CAPGD. Without black-box attacks, we cannot evaluate defence effectiveness on tree models. This is the biggest gap.

Code and configs exist. Need to run 8 configs x 3 seeds = 24 experiments.

```bash
# HopSkipJump (12 experiments)
uv run python -m runner.run --config configs/ccfd_tree_hsj.yaml --seed 42
uv run python -m runner.run --config configs/ccfd_tree_hsj.yaml --seed 123
uv run python -m runner.run --config configs/ccfd_tree_hsj.yaml --seed 456
uv run python -m runner.run --config configs/ieee_cis_tree_hsj.yaml --seed 42
uv run python -m runner.run --config configs/ieee_cis_tree_hsj.yaml --seed 123
uv run python -m runner.run --config configs/ieee_cis_tree_hsj.yaml --seed 456
uv run python -m runner.run --config configs/lcld_tree_hsj.yaml --seed 42
uv run python -m runner.run --config configs/lcld_tree_hsj.yaml --seed 123
uv run python -m runner.run --config configs/lcld_tree_hsj.yaml --seed 456
uv run python -m runner.run --config configs/sparkov_tree_hsj.yaml --seed 42
uv run python -m runner.run --config configs/sparkov_tree_hsj.yaml --seed 123
uv run python -m runner.run --config configs/sparkov_tree_hsj.yaml --seed 456

# Square Attack (12 experiments)
uv run python -m runner.run --config configs/ccfd_tree_square.yaml --seed 42
uv run python -m runner.run --config configs/ccfd_tree_square.yaml --seed 123
uv run python -m runner.run --config configs/ccfd_tree_square.yaml --seed 456
uv run python -m runner.run --config configs/ieee_cis_tree_square.yaml --seed 42
uv run python -m runner.run --config configs/ieee_cis_tree_square.yaml --seed 123
uv run python -m runner.run --config configs/ieee_cis_tree_square.yaml --seed 456
uv run python -m runner.run --config configs/lcld_tree_square.yaml --seed 42
uv run python -m runner.run --config configs/lcld_tree_square.yaml --seed 123
uv run python -m runner.run --config configs/lcld_tree_square.yaml --seed 456
uv run python -m runner.run --config configs/sparkov_tree_square.yaml --seed 42
uv run python -m runner.run --config configs/sparkov_tree_square.yaml --seed 123
uv run python -m runner.run --config configs/sparkov_tree_square.yaml --seed 456
```

**Verify:** Tree model rows show `robust_pr_auc < clean_pr_auc`.

---

### ~~4. Epsilon Sweeps -- Multi-Seed~~ -- DONE

All 4 datasets have 3-seed epsilon sweeps (eps = {0.01, 0.05, 0.1, 0.15, 0.2, 0.3}) completed on Feb 16. Seed 42 has duplicate runs from Feb 12 and Feb 16 — deduplicate when computing final results.

---

### 5. Reproducibility Documentation

- [ ] Dataset download instructions (exact URLs + placement for all 4 datasets)
- [ ] Full reproduction command (`scripts/run_all_seeds.py` documented in README)
- [ ] Config field documentation (all YAML fields explained, including hopskipjump/square)
- [ ] README update (add LCLD, Sparkov, HopSkipJump, Square to README)

---

### 6. Document Tree + Adversarial Training Gap

4 configs are N/A (XGBoost + adversarial training across all datasets). Gradient-based adversarial training is architecturally inapplicable to tree models.

**Action:** Mark as "N/A" in benchmark results table with explanation. This is itself a meaningful finding about model-family-dependent defence applicability.

---

### 7. Auto-Generated Figures and Report

`scripts/generate_figures.py` exists and produces 5 figure types. After running all experiments:

```bash
uv run python scripts/generate_figures.py
```

Figures: robustness bars, attack comparison, summary table, defence heatmap, training time.

---

## P2 -- Enhancements (Strengthen the Benchmark)

### 8. Transferability Experiments

Test whether adversarial examples from Neural models fool XGBoost (and vice versa).

```bash
uv run python -m scripts.transferability --dataset ccfd --seed 42 --epsilon 0.1
uv run python -m scripts.transferability --dataset ieee_cis --seed 42 --epsilon 0.1
uv run python -m scripts.transferability --dataset lcld --seed 42 --epsilon 0.1
uv run python -m scripts.transferability --dataset sparkov --seed 42 --epsilon 0.1
```

**Estimated effort:** 3-5 hours.

---

### 9. Statistical Significance Testing

3-seed setup enables paired t-tests or Wilcoxon signed-rank tests for defence comparisons.

```bash
uv run python scripts/statistical_tests.py
```

**Estimated effort:** 1-2 hours.

---

### 10. Ensemble Defence

Progress Report promises 4 defence methods. Currently have 2. Ensemble (Logistic Regression + XGBoost + Neural with voting/stacking) would add a third.

**Estimated effort:** 4-8 hours.

---

### 11. CTGAN Data Augmentation

Train CTGAN on fraud-class samples to generate synthetic fraud data for training augmentation. Fourth defence method from Progress Report.

**Estimated effort:** 4-8 hours.

---

### 12. Pre-Trained Model Zoo

Save trained model weights for each (dataset, model_type, defence) to `results/models/` so researchers can evaluate new attacks without retraining.

**Estimated effort:** 2-4 hours.

---

## Post-Experiment Tasks

After all GPU experiments complete:

```bash
# Regenerate figures
uv run python scripts/generate_figures.py

# Statistical tests
uv run python scripts/statistical_tests.py

# Input validation analysis
uv run python scripts/analyse_input_validation.py
```

---

## Summary

| # | Task | Type | Experiments | Priority | Status |
|---|------|------|-------------|----------|--------|
| 1 | ~~Re-run IEEE-CIS (NaN fix)~~ | GPU | 15 | P0 | **Done** |
| 2 | Document input validation finding | Writing | -- | P0 | Pending |
| 3 | Black-box attacks (HSJ + Square) | GPU | 24 | P1 | Pending |
| 4 | ~~Epsilon sweeps multi-seed~~ | GPU | 8 (x6 eps) | P1 | **Done** |
| 5 | Reproducibility docs | Writing | -- | P1 | Pending |
| 6 | Document tree+adv_train gap | Writing | -- | P1 | Pending |
| 7 | Generate figures | Local | -- | P1 | Pending |
| 8 | Transferability experiments | GPU | 4 | P2 | Pending |
| 9 | Statistical tests | Local | -- | P2 | Pending |
| 10 | Ensemble defence | Code+GPU | TBD | P2 | Pending |
| 11 | CTGAN augmentation | Code+GPU | TBD | P2 | Pending |
| 12 | Model zoo | GPU | -- | P2 | Pending |

**Total GPU experiments remaining:** 24 (P1: black-box) or ~28 (all including transferability)

**Recommended:** Run `uv run python scripts/run_all_seeds.py` on Colab (T4 GPU) for batch execution.
