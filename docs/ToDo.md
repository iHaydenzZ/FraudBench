# FraudBench: To-Do List

> **Last updated:** 2026-02-20
> **Estimated remaining work:** P1 ~4-6h, All ~10-18h

---

## P0 -- Must Fix (Affects Result Credibility)

### ~~1. Re-run IEEE-CIS Experiments~~ -- DONE

Re-run completed on Feb 12-16. New IEEE-CIS rows show `validity_rate ~0.997`. Old rows from Feb 9-10 have been excluded from `results/registry_clean.csv` (the deduplicated canonical registry created Feb 20).

---

### 2. Document Input Validation Finding

Input validation consistently degrades robustness. This is a genuine finding (not a bug) -- CAPGD already respects constraints, so the z-score clipping destroys discriminative signal without blocking adversarial perturbations.

**Action:** Write up in thesis Discussion section with evidence (fraction of features clipped). Optionally test with different `z_threshold` values (5.0, 10.0) to show the effect.

---

## P1 -- Required for Benchmark Status

### 3. Run Black-Box Attack Experiments -- PARTIAL

Tree models (XGBoost) are immune to gradient-based CAPGD. Black-box attacks are needed to evaluate tree model robustness.

**Square Attack: DONE (12/12).** All 4 datasets x 3 seeds complete in `registry_clean.csv`.

**HopSkipJump: 6/12 done.** Remaining 6 experiments:

```bash
# Remaining HopSkipJump experiments (use scripts/run_remaining_hsj.py for parallelized CPU runs)
uv run python -m runner.run --config configs/ieee_cis_tree_hsj.yaml --seed 456
uv run python -m runner.run --config configs/lcld_tree_hsj.yaml --seed 123
uv run python -m runner.run --config configs/lcld_tree_hsj.yaml --seed 456
uv run python -m runner.run --config configs/sparkov_tree_hsj.yaml --seed 42
uv run python -m runner.run --config configs/sparkov_tree_hsj.yaml --seed 123
uv run python -m runner.run --config configs/sparkov_tree_hsj.yaml --seed 456
```

HSJ experiments are slow (~7-11h each). Use `scripts/run_remaining_hsj.py` for parallelized CPU execution.

**Verify:** Tree model rows show `robust_pr_auc < clean_pr_auc`.

---

### ~~4. Epsilon Sweeps -- Multi-Seed~~ -- DONE

All 4 datasets have 3-seed epsilon sweeps (eps = {0.01, 0.05, 0.1, 0.15, 0.2, 0.3}) completed on Feb 16. Duplicates resolved in `registry_clean.csv` (keeps latest timestamp per experiment+seed+epsilon).

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
| 3 | Black-box attacks (HSJ + Square) | CPU/GPU | 24 | P1 | **Partial** (Square done, HSJ 6/12) |
| 4 | ~~Epsilon sweeps multi-seed~~ | GPU | 8 (x6 eps) | P1 | **Done** |
| 5 | Reproducibility docs | Writing | -- | P1 | Pending |
| 6 | Document tree+adv_train gap | Writing | -- | P1 | Pending |
| 7 | Generate figures | Local | -- | P1 | Pending |
| 8 | Transferability experiments | GPU | 4 | P2 | Pending |
| 9 | Statistical tests | Local | -- | P2 | Pending |
| 10 | Ensemble defence | Code+GPU | TBD | P2 | Pending |
| 11 | CTGAN augmentation | Code+GPU | TBD | P2 | Pending |
| 12 | Model zoo | GPU | -- | P2 | Pending |

**Total GPU/CPU experiments remaining:** 6 (P1: remaining HSJ) or ~10 (all including transferability)

**Recommended:** Run `uv run python scripts/run_remaining_hsj.py` for parallelized CPU execution of remaining HSJ experiments.
