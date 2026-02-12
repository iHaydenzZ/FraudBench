# FraudBench: Project Context

> **Last updated:** 2026-02-12

---

## 1. Project Overview

FraudBench (FRBS — Fraud-Robust Benchmark Suite) evaluates adversarial robustness of fraud detection models. It benchmarks XGBoost and Neural (MLP) models against constrained adversarial attacks on real fraud datasets, with optional defences (adversarial training, input validation).

**Goal:** Deliver a standardised, reproducible benchmark — not just a set of experiments — that other researchers can clone, reproduce, and extend.

**Scope tiers:**

| Tier | Content | Status |
|------|---------|--------|
| Tier 1 | MVP + epsilon curves + multi-seed | Done |
| Tier 2 (minimum) | Tier 1 + 4 datasets | Done (with caveats) |
| Tier 3 (target) | Tier 2 + black-box attacks | Pending |
| Tier 4 (stretch) | Tier 3 + Ensemble + CTGAN + Transferability | Not started |

**Current position:** ~Tier 2 with bugs. 4 datasets covered, but IEEE-CIS has a validity_rate bug and black-box attacks haven't been run.

---

## 2. Architecture

### Pipeline

Config-driven, 9-step pipeline orchestrated by `runner/run.py`:

```
Load Config -> Load Dataset -> Split (stratified, cached) -> Preprocess (StandardScaler/OneHotEncoder)
-> Train Model -> Evaluate Clean -> Infer Constraints -> Run Attack -> Log Results
```

### Key Abstractions

- `models/base.py`: `BaseModel` ABC with `fit()` and `predict_proba()`
- `constraints/schema.py`: `ConstraintSchema` inferred from training data; handles NaN/inf/mixed types; used by CAPGD for constraint-aware projection
- `evaluation/registry.py`: Logs results to `results/registry.csv` (22-column schema)

### Data Flow

`datasets/loader.py` loads raw data -> `datasets/splitter.py` creates stratified train/val/test splits (cached by dataset_name + seed + n_samples) -> `preprocessing/processor.py` fits StandardScaler + OneHotEncoder (cached via joblib)

### Attack Pipeline

- `attacks/capgd.py`: Constrained Auto-PGD (white-box, gradient-based). Perturbs fraud samples within feature bounds defined by `ConstraintSchema`.
- `attacks/hopskipjump.py`: Decision-based black-box attack via ART. Can attack tree models. **Code ready, experiments not run.**
- `attacks/square.py`: Score-based black-box attack via ART. Can attack tree models. **Code ready, experiments not run.**

### Defences

- `defences/adversarial_training.py`: PGD-3 augmentation during training with constraint projection
- `defences/input_validation.py`: Constraint-based bound clipping + z-score outlier clipping at inference

### Config Schema

```yaml
experiment_name: "name"
seed: 42
dataset:
  name: "ccfd"          # ccfd | ieee_cis | lcld | sparkov
  test_size: 0.2
  val_size: 0.2
  sample_frac: 0.1      # fraction of dataset to use
model:
  type: "neural"         # neural | tree
  params:
    epochs: 15           # neural only
    hidden_dim: 128      # neural only
    batch_size: 256      # neural only
attack:
  type: "capgd"          # capgd | hopskipjump | square
  epsilon: 0.1
  steps: 10
defence:
  type: "none"           # none | adversarial_training | input_validation
```

### Caching

Split indices and preprocessor artifacts are cached in `results/`. Cache keys include dataset name, seed, and sample count. Cache auto-invalidates if dataset size changes.

### Repo Structure

```
FraudBench/
├── attacks/          # CAPGD, HopSkipJump, Square Attack
├── configs/          # YAML experiment configs (32 files)
├── constraints/      # Feature constraint schema + validator
├── datasets/         # Loaders for 4 datasets + cards/
│   └── cards/        # Dataset documentation (4 cards)
├── defences/         # Adversarial training, input validation
├── evaluation/       # Metrics + registry
├── models/           # Tree (XGBoost) + Neural (MLP)
├── preprocessing/    # StandardScaler + OneHotEncoder pipeline
├── results/          # registry.csv + cached artifacts
├── runner/           # CLI entrypoint
├── scripts/          # Batch runner, figure generation, analysis
├── tests/            # Pytest suite (64 tests)
├── notebooks/        # Colab runner + debug notebooks
└── docs/             # This file + ToDo.md
```

---

## 3. Current Experiment State

### Registry Statistics

- **Total rows:** ~101 (includes re-runs and epsilon sweeps)
- **Datasets:** CCFD, IEEE-CIS, LCLD, Sparkov (4)
- **Models:** Neural MLP, XGBoost (2)
- **Attacks executed:** CAPGD only (HopSkipJump + Square code ready, not run)
- **Defences:** none, adversarial_training, input_validation (3)
- **Seeds:** 42, 123, 456
- **Unique configs executed:** 20/24 (83%) -- 4 tree+adv_train are N/A (gradients required)

### Coverage Matrix (CAPGD, 3 seeds each)

| Dataset | Model | No Defence | Adv Training | Input Validation |
|---------|-------|-----------|-------------|-----------------|
| CCFD | Neural | Done | Done | Done |
| CCFD | XGBoost | Done | N/A* | Done |
| IEEE-CIS | Neural | Done | Done | Done |
| IEEE-CIS | XGBoost | Done | N/A* | Done |
| LCLD | Neural | Done | Done | Done |
| LCLD | XGBoost | Done | N/A* | Done |
| Sparkov | Neural | Done | Done | Done |
| Sparkov | XGBoost | Done | N/A* | Done |

\* XGBoost + adversarial training is architecturally incompatible (requires gradients). This is itself a finding.

### Epsilon Sweeps (seed 42 only)

Sweeps at epsilon = {0.01, 0.05, 0.1, 0.15, 0.2, 0.3} completed for: CCFD, IEEE-CIS, LCLD, Sparkov (all Neural, no defence).

---

## 4. Key Research Findings

### Positive Results

1. **CAPGD effectively degrades Neural models** across all datasets:
   - CCFD: -18.5% robust PR-AUC
   - IEEE-CIS: -82.7%
   - LCLD: -65.4%
   - Sparkov: -99.1%

2. **Adversarial training significantly improves robustness** for Neural models:
   - CCFD: +7% (robust PR-AUC improvement over baseline)
   - IEEE-CIS: +254%
   - LCLD: +184%
   - Sparkov: +3742% (from near-zero)

3. **XGBoost is immune to gradient-based CAPGD** (expected -- tree models lack gradients). Clean = Robust for all tree experiments.

### Negative Results / Issues

1. **Input validation consistently degrades robustness** -- never improves it (worst case: -76%). Analysis suggests this is a genuine finding: CAPGD already respects constraints, so bound clipping has no effect. The z-score clipping destroys discriminative signal the model relies on.

2. **LCLD and Sparkov Neural clean PR-AUC is modest** (~0.30 and ~0.62), limiting interpretive power of robustness comparisons on these datasets.

---

## 5. Known Bugs

### P0: IEEE-CIS validity_rate = 0.0000

All IEEE-CIS experiment rows report `validity_rate = 0.0000`. Root cause: `constraints/validator.py` fails on NaN categorical values (NaN != NaN so `NaN not in allowed_values` is always True). The `has_missing` flag exists but is never checked during validation.

**Status:** Code fix applied. Old results with `validity_rate = 0` still in registry -- experiments need re-running.

### Addressed: Input Validation Performance

Investigated and determined to be a genuine finding (not a bug). Input validation as implemented is ineffective against constrained adversarial attacks. Should be documented in thesis Discussion section.

---

## 6. Colab Integration

### Workflow

1. Local development: `uv run` and `uv run pytest`
2. Push to GitHub
3. Run on Colab: `notebooks/colab_runner.ipynb`
4. Results auto-save to Google Drive

### Notes

- Colab uses `pip install -e .` (no `uv`)
- Runner command: `python -m runner.run --config configs/mvp.yaml`
- Datasets on Drive: `/content/drive/MyDrive/FraudBench/data/`
- Individual dataset dirs are symlinked into `datasets/`

### Compute Budget

- T4 GPU: ~1.96 units/hour
- A100 GPU: ~12.46 units/hour
- CPU: Free

**Always use T4 unless A100 is specifically needed.**

---

## 7. Progress Report Phase Alignment

| Phase | Weeks | Content | Status |
|-------|-------|---------|--------|
| Phase 1: Data Preparation | Wk 1-2 | 4 datasets cleaned, preprocessed | Done |
| Phase 2: Model Building | Wk 3-4 | XGBoost + Neural MLP baselines | Done |
| Phase 3: Attack Implementation | Wk 5-6 | CAPGD + black-box attacks | Partial (CAPGD only, black-box code ready) |
| Phase 4: Defence Integration | Wk 7-9 | 4 defence methods | Partial (2/4: adv training + input validation) |
| Phase 5: Evaluation & Benchmarking | Wk 10-12 | Full matrix + transferability + auto reports | Partial (results exist, no auto reports) |
| Phase 6: Analysis & Reporting | Wk 13 | Final report + visualizations | Not started |
