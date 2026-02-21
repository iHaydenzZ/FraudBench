# FraudBench: Project Context

> **Last updated:** 2026-02-21

---

## 1. Project Overview

FraudBench (FRBS — Fraud-Robust Benchmark Suite) evaluates adversarial robustness of fraud detection models. It benchmarks XGBoost and Neural (MLP) models against constrained adversarial attacks on real fraud datasets, with optional defences (adversarial training, input validation).

**Goal:** Deliver a standardised, reproducible benchmark — not just a set of experiments — that other researchers can clone, reproduce, and extend.

**Scope tiers:**

| Tier | Content | Status |
|------|---------|--------|
| Tier 1 | MVP + epsilon curves + multi-seed | Done |
| Tier 2 (minimum) | Tier 1 + 4 datasets | Done |
| Tier 3 (target) | Tier 2 + black-box attacks | Partial (Square done, HSJ 6/12) |
| Tier 4 (stretch) | Tier 3 + Ensemble + CTGAN + Transferability | Partial (Ensemble done) |

**Current position:** Tier 3 complete (Square), Tier 4 partial (Ensemble done). HopSkipJump 6/12 (ccfd done; ieee_cis 2/3; lcld 1/3; sparkov 0/3). All CAPGD + epsilon sweep experiments complete. Ensemble experiments complete (24 runs). Deduplicated registry at `results/registry_clean.csv` (182 rows). All figures and analysis artefacts fixed and regenerated (Feb 21).

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
- `evaluation/registry.py`: Logs results to `results/registry.csv` (22-column schema). Deduplicated canonical data in `results/registry_clean.csv`.

### Data Flow

`datasets/loader.py` loads raw data -> `datasets/splitter.py` creates stratified train/val/test splits (cached by dataset_name + seed + n_samples) -> `preprocessing/processor.py` fits StandardScaler + OneHotEncoder (cached via joblib)

### Attack Pipeline

- `attacks/capgd.py`: Constrained Auto-PGD (white-box, gradient-based). Perturbs fraud samples within feature bounds defined by `ConstraintSchema`.
- `attacks/hopskipjump.py`: Decision-based black-box attack via ART. Can attack tree models. **6/12 experiments run** (ccfd 3/3, ieee_cis 2/3, lcld 1/3, sparkov 0/3).
- `attacks/square.py`: Score-based black-box attack via ART. Can attack tree models. **All 12 experiments complete.**

### Defences

- `defences/adversarial_training.py`: PGD-3 augmentation during training with constraint projection
- `defences/input_validation.py`: Constraint-based bound clipping + z-score outlier clipping at inference
- `defences/ensemble.py` + `models/ensemble.py`: Heterogeneous ensemble (LR + XGBoost + MLP) with soft voting

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
├── configs/          # YAML experiment configs (36 files)
├── constraints/      # Feature constraint schema + validator
├── datasets/         # Loaders for 4 datasets + cards/
│   └── cards/        # Dataset documentation (4 cards)
├── defences/         # Adversarial training, input validation
├── evaluation/       # Metrics + registry
├── models/           # Tree (XGBoost) + Neural (MLP)
├── preprocessing/    # StandardScaler + OneHotEncoder pipeline
├── results/          # registry.csv, registry_clean.csv + cached artifacts
├── runner/           # CLI entrypoint
├── scripts/          # Batch runner, figure generation, analysis, run_remaining_hsj.py
├── tests/            # Pytest suite (97 tests, 10 files)
├── notebooks/        # Colab runner + debug notebooks
├── docs/             # This file + ToDo.md
└── .github/          # CI/CD (GitHub Actions: ruff + pytest)
```

---

## 3. Current Experiment State

### Registry Statistics (registry_clean.csv — deduplicated)

- **Total rows:** 182 (deduplicated; keeps latest timestamp per experiment+seed+epsilon)
- **Raw registries:** `registry.csv`, `20260216_GPU_only_registry.csv` — merged and deduplicated into `registry_clean.csv`
- **Date range:** 2026-02-13 to 2026-02-21
- **Datasets:** CCFD, IEEE-CIS, LCLD, Sparkov (4 datasets)
- **Models:** Neural MLP, XGBoost, Ensemble (LR+XGBoost+MLP)
- **Attacks executed:** CAPGD (white-box), Square (black-box, complete), HopSkipJump (black-box, 6/12)
- **Defences:** none, adversarial_training, input_validation, ensemble
- **Seeds:** 42, 123, 456
- **Unique configs executed:** 20/24 baseline + 24 ensemble = 35 configurations

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

### Black-Box Attack Coverage (XGBoost, no defence, 3 seeds each)

| Dataset | Square Attack | HopSkipJump |
|---------|--------------|-------------|
| CCFD | Done (3/3) | Done (3/3) |
| IEEE-CIS | Done (3/3) | 2/3 (missing seed 456) |
| LCLD | Done (3/3) | 1/3 (missing seeds 123, 456) |
| Sparkov | Done (3/3) | 0/3 (not started) |

### Epsilon Sweeps (3 seeds)

Sweeps at epsilon = {0.01, 0.05, 0.1, 0.15, 0.2, 0.3} completed for all 4 datasets (Neural, no defence) with seeds 42, 123, and 456. Duplicates from multi-date runs resolved in `registry_clean.csv`.

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

4. **Black-box attacks successfully degrade XGBoost.** Square Attack and HopSkipJump both reduce robust metrics on tree models, confirming that XGBoost's CAPGD immunity does not extend to decision/score-based attacks. HopSkipJump is particularly devastating on CCFD (robust_accuracy drops to ~0.0).

### Negative Results / Issues

1. **Input validation consistently degrades robustness** -- never improves it (worst case: -76%). Analysis suggests this is a genuine finding: CAPGD already respects constraints, so bound clipping has no effect. The z-score clipping destroys discriminative signal the model relies on.

2. **LCLD and Sparkov Neural clean PR-AUC is modest** (~0.30 and ~0.62), limiting interpretive power of robustness comparisons on these datasets.

---

## 5. Known Bugs

### Resolved: IEEE-CIS validity_rate = 0.0000

Root cause: `constraints/validator.py` failed on NaN categorical values (NaN != NaN so `NaN not in allowed_values` was always True). The `has_missing` flag existed but was never checked during validation.

**Status:** Fixed and re-run. Feb 12+ IEEE-CIS rows show `validity_rate ~0.997`. Old rows from Feb 9-10 have been excluded from `registry_clean.csv` (the deduplicated canonical registry). No rows with `validity_rate = 0.0000` remain in the clean registry.

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
| Phase 3: Attack Implementation | Wk 5-6 | CAPGD + black-box attacks | Mostly done (CAPGD complete, Square complete, HSJ 6/12) |
| Phase 4: Defence Integration | Wk 7-9 | 4 defence methods | Partial (3/4: adv training + input validation + ensemble) |
| Phase 5: Evaluation & Benchmarking | Wk 10-12 | Full matrix + transferability + auto reports | Mostly done (CAPGD + Square + Ensemble complete, eps sweeps done, HSJ 6/12, no transferability) |
| Phase 6: Analysis & Reporting | Wk 13 | Final report + visualizations | In progress (figures fixed, stats done, docs partial) |
