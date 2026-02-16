# FraudBench: Project Context

> **Last updated:** 2026-02-16

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

**Current position:** Tier 2 complete. 4 datasets covered with 3-seed CAPGD + epsilon sweeps. IEEE-CIS validity_rate bug fixed and re-run. Black-box attacks (HSJ + Square) not yet run.

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
├── tests/            # Pytest suite (77 tests)
├── notebooks/        # Colab runner + debug notebooks
├── docs/             # This file + ToDo.md
└── .github/          # CI/CD (GitHub Actions: ruff + pytest)
```

---

## 3. Current Experiment State

### Registry Statistics

- **Total rows:** 235 (includes re-runs and epsilon sweeps; some duplicates from multi-date runs)
- **Date range:** 2026-02-09 to 2026-02-16
- **Datasets:** CCFD (74), IEEE-CIS (53), LCLD (51), Sparkov (57)
- **Models:** Neural MLP (185), XGBoost (50)
- **Attacks executed:** CAPGD only (HopSkipJump + Square code ready, not run)
- **Defences:** none (155), adversarial_training (28), input_validation (52)
- **Seeds:** 42 (104), 123 (66), 456 (65)
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

### Epsilon Sweeps (3 seeds)

Sweeps at epsilon = {0.01, 0.05, 0.1, 0.15, 0.2, 0.3} completed for all 4 datasets (Neural, no defence) with seeds 42, 123, and 456. Seed 42 has duplicate runs from Feb 12 and Feb 16.

---

## 4. Key Research Findings

### Positive Results

1. **CAPGD effectively degrades Neural models** across all datasets (3-seed avg, eps=0.1, no defence):

   | Dataset | Clean PR-AUC | Robust PR-AUC | Drop |
   |---------|-------------|---------------|------|
   | CCFD | 0.717 | 0.582 | -18.8% |
   | IEEE-CIS | 0.446 | 0.070 | -84.3% |
   | LCLD | 0.306 | 0.105 | -65.7% |
   | Sparkov | 0.604 | 0.005 | -99.1% |

2. **Adversarial training significantly improves robustness** for Neural models (reduces PR-AUC drop under attack).

3. **XGBoost is immune to gradient-based CAPGD** (expected -- tree models lack gradients). Clean = Robust for all tree experiments (CCFD: 0.851, IEEE-CIS: 0.568, LCLD: 0.368, Sparkov: 0.747).

### Negative Results / Issues

1. **Input validation consistently degrades robustness** -- never improves it (worst case: -76%). Analysis suggests this is a genuine finding: CAPGD already respects constraints, so bound clipping has no effect. The z-score clipping destroys discriminative signal the model relies on.

2. **LCLD and Sparkov Neural clean PR-AUC is modest** (~0.30 and ~0.62), limiting interpretive power of robustness comparisons on these datasets.

---

## 5. Known Bugs

### Resolved: IEEE-CIS validity_rate = 0.0000

Root cause: `constraints/validator.py` failed on NaN categorical values (NaN != NaN so `NaN not in allowed_values` was always True). The `has_missing` flag existed but was never checked during validation.

**Status:** Fixed and re-run. Feb 12+ IEEE-CIS rows show `validity_rate ~0.997`. Old rows from Feb 9-10 (15 rows with `validity_rate = 0.0000`) remain in the registry but are superseded by the re-runs. These should be excluded when computing final results.

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

## 7. CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push to master and PRs targeting master:

- **lint** job: `ruff check .` + `ruff format --check .` (~30s)
- **test** job: CPU-only torch, `uv sync`, `pytest -m "not slow"` (~3-5min)

Dataset-dependent tests skip automatically via `@pytest.mark.skipif` decorators.

---

## 8. Progress Report Phase Alignment

| Phase | Weeks | Content | Status |
|-------|-------|---------|--------|
| Phase 1: Data Preparation | Wk 1-2 | 4 datasets cleaned, preprocessed | Done |
| Phase 2: Model Building | Wk 3-4 | XGBoost + Neural MLP baselines | Done |
| Phase 3: Attack Implementation | Wk 5-6 | CAPGD + black-box attacks | Partial (CAPGD only, black-box code ready) |
| Phase 4: Defence Integration | Wk 7-9 | 4 defence methods | Partial (2/4: adv training + input validation) |
| Phase 5: Evaluation & Benchmarking | Wk 10-12 | Full matrix + transferability + auto reports | Partial (CAPGD matrix complete, 3-seed eps sweeps done, no black-box or transferability) |
| Phase 6: Analysis & Reporting | Wk 13 | Final report + visualizations | Not started |
