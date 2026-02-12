# FraudBench: To-Do List

> **Last updated:** 2026-02-12
> **Estimated remaining work:** P0+P1 ~10-14h, All ~16-24h

---

## P0 -- Must Fix (Affects Result Credibility)

### 1. Re-run IEEE-CIS Experiments (15 experiments)

The NaN validator bug is fixed in code, but the registry still contains old results with `validity_rate = 0.0000`. Must re-run all 15 IEEE-CIS experiments to get correct validity rates.

```bash
# All IEEE-CIS configs x 3 seeds (or use batch runner)
uv run python -m runner.run --config configs/ieee_cis.yaml --seed 42
uv run python -m runner.run --config configs/ieee_cis.yaml --seed 123
uv run python -m runner.run --config configs/ieee_cis.yaml --seed 456
uv run python -m runner.run --config configs/ieee_cis_adv_train.yaml --seed 42
uv run python -m runner.run --config configs/ieee_cis_adv_train.yaml --seed 123
uv run python -m runner.run --config configs/ieee_cis_adv_train.yaml --seed 456
uv run python -m runner.run --config configs/ieee_cis_input_val.yaml --seed 42
uv run python -m runner.run --config configs/ieee_cis_input_val.yaml --seed 123
uv run python -m runner.run --config configs/ieee_cis_input_val.yaml --seed 456
uv run python -m runner.run --config configs/ieee_cis_tree.yaml --seed 42
uv run python -m runner.run --config configs/ieee_cis_tree.yaml --seed 123
uv run python -m runner.run --config configs/ieee_cis_tree.yaml --seed 456
uv run python -m runner.run --config configs/ieee_cis_tree_input_val.yaml --seed 42
uv run python -m runner.run --config configs/ieee_cis_tree_input_val.yaml --seed 123
uv run python -m runner.run --config configs/ieee_cis_tree_input_val.yaml --seed 456
```

**Verify:** IEEE-CIS rows have `validity_rate >= 0.95`.

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

### 4. Epsilon Sweeps -- Multi-Seed (12 experiments, 6 epsilon each)

Single-seed sweeps exist (seed 42). Need seeds 123 and 456 for all 4 datasets.

```bash
uv run python -m runner.run --config configs/ccfd_eps_sweep.yaml --seed 123
uv run python -m runner.run --config configs/ccfd_eps_sweep.yaml --seed 456
uv run python -m runner.run --config configs/ieee_cis_eps_sweep.yaml --seed 123
uv run python -m runner.run --config configs/ieee_cis_eps_sweep.yaml --seed 456
uv run python -m runner.run --config configs/lcld_eps_sweep.yaml --seed 123
uv run python -m runner.run --config configs/lcld_eps_sweep.yaml --seed 456
uv run python -m runner.run --config configs/sparkov_eps_sweep.yaml --seed 123
uv run python -m runner.run --config configs/sparkov_eps_sweep.yaml --seed 456
```

**Verify:** Registry has multi-epsilon results with 3 seeds each.

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

| # | Task | Type | Experiments | Priority |
|---|------|------|-------------|----------|
| 1 | Re-run IEEE-CIS (NaN fix) | GPU | 15 | P0 |
| 2 | Document input validation finding | Writing | -- | P0 |
| 3 | Black-box attacks (HSJ + Square) | GPU | 24 | P1 |
| 4 | Epsilon sweeps multi-seed | GPU | 8 (x6 eps) | P1 |
| 5 | Reproducibility docs | Writing | -- | P1 |
| 6 | Document tree+adv_train gap | Writing | -- | P1 |
| 7 | Generate figures | Local | -- | P1 |
| 8 | Transferability experiments | GPU | 4 | P2 |
| 9 | Statistical tests | Local | -- | P2 |
| 10 | Ensemble defence | Code+GPU | TBD | P2 |
| 11 | CTGAN augmentation | Code+GPU | TBD | P2 |
| 12 | Model zoo | GPU | -- | P2 |

**Total GPU experiments remaining:** ~47 (P0+P1) or ~51 (all)

**Recommended:** Run `uv run python scripts/run_all_seeds.py` on Colab (T4 GPU) for batch execution.
