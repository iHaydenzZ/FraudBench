# FraudBench: Remaining Tasks

> **Date:** 2026-02-12
> **Registry snapshot:** 63 rows, all CAPGD, all epsilon=0.1
> **Purpose:** Checklist of outstanding work items. All code/config changes are done — these are **compute tasks** (run experiments) and one analysis task.

---

## P0 — Must Fix Before Submission

### 1. Re-run all IEEE-CIS experiments (15 experiments)

**Why:** The NaN validator bug has been fixed in code (`constraints/validator.py`), but the registry still contains the old results where `validity_rate = 0.0000` for all 15 IEEE-CIS rows. These must be re-run to get correct validity rates.

**Command:**
```bash
# Run all IEEE-CIS configs × 3 seeds
uv run python -m runner.run --config configs/ieee_cis_baseline.yaml --seed 42
uv run python -m runner.run --config configs/ieee_cis_baseline.yaml --seed 123
uv run python -m runner.run --config configs/ieee_cis_baseline.yaml --seed 456
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

Or use the batch runner (runs everything, including these):
```bash
uv run python scripts/run_all_seeds.py
```

**Verify:** After re-run, check that IEEE-CIS rows have `validity_rate >= 0.95`.

**On Colab:**
```python
!python -m runner.run --config configs/ieee_cis_baseline.yaml --seed 42
# ... etc.
```

---

## P1 — Required for Benchmark Status

### 2. Run 24 black-box attack experiments

**Why:** Tree models (XGBoost) are immune to gradient-based CAPGD. Without black-box attacks (HopSkipJump, Square), we cannot evaluate defence effectiveness on tree models at all. This is the biggest gap in the benchmark.

**Configs (8 configs × 3 seeds = 24 experiments):**
```bash
# HopSkipJump
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

# Square Attack
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

**Verify:** Tree model rows show `robust_pr_auc < clean_pr_auc` (actual robustness degradation, not immune).

---

### 3. Run epsilon sweep experiments

**Why:** A benchmark with only epsilon=0.1 is incomplete. Robustness curves (PR-AUC vs epsilon) are the standard output for adversarial robustness benchmarks.

**Configs (4 configs × 3 seeds = 12 experiments, each with 6 epsilon values):**
```bash
uv run python -m runner.run --config configs/ccfd_eps_sweep.yaml --seed 42
uv run python -m runner.run --config configs/ccfd_eps_sweep.yaml --seed 123
uv run python -m runner.run --config configs/ccfd_eps_sweep.yaml --seed 456
uv run python -m runner.run --config configs/ieee_cis_eps_sweep.yaml --seed 42
uv run python -m runner.run --config configs/ieee_cis_eps_sweep.yaml --seed 123
uv run python -m runner.run --config configs/ieee_cis_eps_sweep.yaml --seed 456
uv run python -m runner.run --config configs/lcld_eps_sweep.yaml --seed 42
uv run python -m runner.run --config configs/lcld_eps_sweep.yaml --seed 123
uv run python -m runner.run --config configs/lcld_eps_sweep.yaml --seed 456
uv run python -m runner.run --config configs/sparkov_eps_sweep.yaml --seed 42
uv run python -m runner.run --config configs/sparkov_eps_sweep.yaml --seed 123
uv run python -m runner.run --config configs/sparkov_eps_sweep.yaml --seed 456
```

**Verify:** Registry contains rows with epsilon values: 0.01, 0.05, 0.1, 0.15, 0.2, 0.3.

---

## P2 — Enhancements (Nice to Have)

### 4. Run transferability experiments

**Why:** Tests whether adversarial examples generated on Neural models can fool XGBoost (and vice versa). Answers: do defences need to be attack-type-specific?

**Command:**
```bash
uv run python -m scripts.transferability --dataset ccfd --seed 42 --epsilon 0.1
uv run python -m scripts.transferability --dataset ieee_cis --seed 42 --epsilon 0.1
uv run python -m scripts.transferability --dataset lcld --seed 42 --epsilon 0.1
uv run python -m scripts.transferability --dataset sparkov --seed 42 --epsilon 0.1
```

---

### 5. Populate pre-trained model zoo

**Why:** Allows other researchers to evaluate new attacks/defences without retraining. Save/load code exists in models, but no weights have been saved.

**Action:** After running all experiments, save trained model weights to `results/models/` for each (dataset, model_type, defence) combination.

---

## Post-Experiment Tasks

After all experiments complete:

```bash
# Regenerate all figures
uv run python scripts/generate_figures.py

# Re-run statistical tests with new data
uv run python scripts/statistical_tests.py

# Re-run input validation analysis
uv run python scripts/analyse_input_validation.py
```

---

## Summary

| # | Task | Type | Experiments | Priority |
|---|------|------|-------------|----------|
| 1 | Re-run IEEE-CIS (NaN fix validation) | GPU compute | 15 | P0 |
| 2 | Black-box attacks (HopSkipJump + Square) | GPU compute | 24 | P1 |
| 3 | Epsilon sweeps (robustness curves) | GPU compute | 12 (×6 eps each) | P1 |
| 4 | Transferability experiments | GPU compute | 4 | P2 |
| 5 | Model zoo (save weights) | GPU compute | — | P2 |
| 6 | Regenerate figures + stats | Local (fast) | — | After 1-3 |

**Total GPU experiments remaining:** ~51 (P0+P1) or ~55 (all)

**Recommended approach:** Run `uv run python scripts/run_all_seeds.py` on Colab (T4 GPU) to execute everything in one batch, then regenerate figures locally.
