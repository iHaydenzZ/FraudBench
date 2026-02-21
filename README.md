# FraudBench: Fraud-Robust Benchmark Suite

A reproducible benchmark suite for evaluating adversarial robustness of fraud detection models. FraudBench compares adversarial training and defence methods across four financial fraud datasets, benchmarking XGBoost and Neural (MLP) models against white-box and black-box adversarial attacks.

## Features

- **Datasets**: CCFD, IEEE-CIS, LCLD (Lending Club), and Sparkov (simulated transactions)
- **Models**: Tree-based (XGBoost) and Neural (MLP) with automatic class weighting
- **Attacks**: CAPGD (white-box), HopSkipJump (decision-based black-box), Square Attack (score-based black-box)
- **Defences**: Adversarial training, input validation, and ensemble (heterogeneous model voting)
- **Metrics**: PR-AUC, precision, recall, F1 (fraud-first evaluation)
- **Reproducibility**: Config-driven experiments with cached splits and preprocessing
- **Multi-seed evaluation**: 3 seeds per config for statistical rigour

## Installation

Requires Python 3.11+. Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Quick Start

Run an experiment with the default config:

```bash
uv run python -m runner.run --config configs/mvp.yaml
```

## Benchmark Results

### CCFD (Full Dataset - 284,807 samples)

| Metric | Clean | Robust (CAPGD) | Change |
|--------|-------|----------------|--------|
| PR-AUC | 0.758 | 0.638 | -15.8% |
| Recall | 88.8% | 85.7% | -3.5% |
| Precision | 18.8% | 3.4% | -81.9% |

- Model: Neural MLP (128 hidden, 20 epochs)
- Attack: CAPGD (epsilon=0.1, 10 steps)
- Training time: 10.5s | Attack time: 0.24s

### IEEE-CIS (10% Sample - 59,054 samples)

| Metric | Value |
|--------|-------|
| PR-AUC | 0.582 |
| Recall | 34.1% |
| Precision | 84.2% |

- Model: XGBoost (depth 6, 100 estimators)
- Note: Tree models don't support gradient-based attacks (CAPGD); use HopSkipJump or Square Attack

## Project Structure

```
FraudBench/
├── attacks/          # Attack implementations (CAPGD, HopSkipJump, Square)
├── configs/          # Experiment configurations (YAML)
├── constraints/      # Feature constraint schema and validation
├── datasets/         # Dataset loaders and cards
│   └── cards/        # Dataset documentation (ccfd, ieee_cis, lcld, sparkov)
├── defences/         # Defence implementations (input validation, adv training, ensemble)
├── evaluation/       # Metrics and results registry
├── models/           # Model wrappers (Tree, Neural)
├── preprocessing/    # Data preprocessing pipeline
├── results/          # Experiment results and cached artifacts
├── runner/           # Experiment runner
├── scripts/          # Batch run and figure generation scripts
└── tests/            # Unit tests (87 tests)
```

## Datasets

| Dataset | Samples | Features | Fraud Rate | Source |
|---------|---------|----------|------------|--------|
| CCFD | 284,807 | 30 | 0.17% | Kaggle (real) |
| IEEE-CIS | 590,540 | 392 | 3.5% | IEEE-CIS / Vesta (real) |
| LCLD | ~2.26M (filtered) | 63 | 19.6% | Lending Club (real) |
| Sparkov | ~1.85M | 11 | 0.52% | Sparkov (simulated) |

See `datasets/cards/` for detailed dataset documentation (preprocessing, known issues, citation).

**Download links:**
- **CCFD**: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (~144 MB)
- **IEEE-CIS**: [Kaggle — IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) (~1.1 GB)
- **LCLD**: [Kaggle — Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (~1.6 GB)
- **Sparkov**: [Kaggle — Sparkov Fraud Transactions](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (~490 MB)

Datasets should be placed at the path specified by `DEFAULT_DATA_ROOT` in `datasets/loader.py`:
```
<DEFAULT_DATA_ROOT>/
├── CCFD/
│   └── creditcard.csv
├── ieee-fraud-detection/
│   ├── train_transaction.csv
│   └── train_identity.csv (optional)
├── LCLD/
│   └── loan.csv
└── Sparkov/
    ├── fraudTrain.csv
    └── fraudTest.csv
```

## Configuration

Experiments are configured via YAML files in `configs/`. Example:

```yaml
experiment_name: "ccfd_baseline"
seed: 42

dataset:
  name: "ccfd"           # or "ieee_cis", "lcld", "sparkov"
  test_size: 0.2
  val_size: 0.2
  sample_frac: 0.1       # Optional: sample for faster iteration

model:
  type: "neural"         # or "tree"
  params:
    epochs: 20
    hidden_dim: 128
    batch_size: 256

attack:
  type: "capgd"          # or "hopskipjump", "square"
  epsilon: 0.1
  steps: 10              # capgd only
  epsilon_values: [0.05, 0.1, 0.2]  # optional: sweep multiple epsilons

defence:
  type: "none"           # or "adversarial_training", "input_validation", "ensemble"
  params:                # Defence-specific parameters (optional)
    epsilon: 0.1         # adversarial_training: PGD perturbation budget
    mode: "sanitise"     # input_validation: "sanitise" (clip) or "reject" (drop)
    z_threshold: 3.0     # input_validation: z-score outlier threshold
```

**Field reference:**

| Section | Field | Description |
|---------|-------|-------------|
| (root) | `experiment_name` | Unique experiment identifier for the registry |
| (root) | `seed` | Random seed (overridable via `--seed` CLI flag) |
| `dataset` | `name` | One of: `ccfd`, `ieee_cis`, `lcld`, `sparkov` |
| `dataset` | `split_strategy` | `"stratified"` (preserves class ratios) |
| `dataset` | `test_size` | Test set fraction (e.g. `0.2`) |
| `dataset` | `val_size` | Validation set fraction (e.g. `0.2`) |
| `dataset` | `sample_frac` | Optional subsample fraction for faster runs |
| `model` | `type` | `"neural"`, `"tree"`, or `"ensemble"` |
| `model.params` | `epochs`, `hidden_dim`, `batch_size`, `lr` | Neural/Ensemble MLP hyperparameters |
| `model.params` | `max_depth`, `n_estimators`, `learning_rate` | Tree (XGBoost) hyperparameters |
| `attack` | `type` | `"capgd"`, `"hopskipjump"`, `"square"` |
| `attack` | `epsilon` | L-inf perturbation budget |
| `attack` | `steps` | PGD iterations (CAPGD only) |
| `attack` | `max_iter` | Max iterations (Square/HSJ) |
| `attack` | `norm` | Norm type: `"inf"`, `2` (Square/HSJ) |
| `defence` | `type` | `"none"`, `"adversarial_training"`, `"input_validation"`, `"ensemble"` |
| `defence.params` | `epsilon` | Adversarial training PGD budget |
| `defence.params` | `mode` | Input validation: `"sanitise"` or `"reject"` |
| `defence.params` | `z_threshold` | Input validation: z-score threshold (default 3.0) |

## Available Configs

| Config | Dataset | Model | Attack | Defence |
|--------|---------|-------|--------|---------|
| `ccfd.yaml` | CCFD | Neural | CAPGD | None |
| `ccfd_tree.yaml` | CCFD | Tree | CAPGD | None |
| `ccfd_ensemble.yaml` | CCFD | Ensemble | CAPGD | Ensemble |
| `ccfd_ensemble_square.yaml` | CCFD | Ensemble | Square | Ensemble |
| `ccfd_input_val.yaml` | CCFD | Neural | CAPGD | Input Val (z=3) |
| `ccfd_input_val_z5.yaml` | CCFD | Neural | CAPGD | Input Val (z=5) |
| `ccfd_input_val_z10.yaml` | CCFD | Neural | CAPGD | Input Val (z=10) |

Similar configs exist for `ieee_cis`, `lcld`, and `sparkov`. See `configs/` for the full set.

## Running Experiments

```bash
# Default MVP config (CCFD 10% sample)
uv run python -m runner.run --config configs/mvp.yaml

# Full CCFD experiment
uv run python -m runner.run --config configs/ccfd.yaml

# IEEE-CIS experiment (tree model)
uv run python -m runner.run --config configs/ieee_cis.yaml
```

## Results & Caching

Results are logged to `results/registry.csv` with columns:
- Experiment metadata (dataset, model, defence, attack)
- Clean metrics (PR-AUC, recall, accuracy)
- Robust metrics (under attack)
- Validity rates, training time, attack time

### Cache Invalidation
Artifacts include dataset size to prevent stale cache issues:
```
results/split_indices_ccfd_n284807_seed42.json
results/preprocessor_ccfd_n284807_seed42.joblib
```

Changing `sample_frac` or dataset automatically creates new cache files.

## Models

### Tree Model (XGBoost)
Best for datasets with categorical features or high missing rates.
```yaml
model:
  type: "tree"
  params:
    max_depth: 6
    n_estimators: 100
    learning_rate: 0.1
```
Note: Does not support gradient-based attacks (CAPGD skipped). Use HopSkipJump or Square Attack instead.

### Neural Model (MLP)
Supports adversarial attacks. Automatic class weighting for imbalanced data.
```yaml
model:
  type: "neural"
  params:
    epochs: 20
    hidden_dim: 128
    batch_size: 256
    class_weight: true   # Default: true (handles imbalance)
```

### Ensemble Model (LR + XGBoost + MLP)
Heterogeneous ensemble with soft voting. Combines LogisticRegression, XGBoost, and a 3-layer MLP. The MLP component is exposed for CAPGD gradient-based attacks.
```yaml
model:
  type: "ensemble"
  params:
    epochs: 15
    hidden_dim: 128
    batch_size: 256
    lr: 0.001
```

## Attacks

### CAPGD (Constrained Auto-PGD)
White-box attack with constraint-aware projection. Requires gradients — works with **neural** and **ensemble** models (targets MLP component).
```yaml
attack:
  type: "capgd"
  epsilon: 0.1      # L-inf perturbation budget
  steps: 10         # PGD iterations
  step_size: 0.025  # Optional (default: epsilon/4)
```

Features:
- Respects feature bounds (min/max from training data)
- Projects to feasible region after each step
- Reports constraint validity rate

### HopSkipJump
Decision-based black-box attack via ART. Works on any model type (tree or neural).
```yaml
attack:
  type: "hopskipjump"
  epsilon: 0.1
```

### Square Attack
Score-based black-box attack via ART. Works on any model type (tree or neural).
```yaml
attack:
  type: "square"
  epsilon: 0.1
```

## Defences

### Adversarial Training
PGD-based adversarial training during model fitting. Neural models only (requires gradients).
```yaml
defence:
  type: "adversarial_training"
  params:
    epsilon: 0.1       # PGD perturbation budget
```

### Input Validation
Z-score outlier detection applied at inference time. Works with any model type.
```yaml
defence:
  type: "input_validation"
  params:
    mode: "sanitise"     # "sanitise" (clip outliers) or "reject" (drop outlier rows)
    z_threshold: 3.0     # z-score threshold for outlier detection
```

### Ensemble
Heterogeneous model ensemble (LR + XGBoost + MLP) with soft voting. Diversity across model families is the defensive mechanism.
```yaml
model:
  type: "ensemble"
defence:
  type: "ensemble"
```

## Testing

Run the test suite (87 tests across 10 test files):

```bash
uv run pytest tests/ -v
```

Test coverage includes:
- Dataset loading and splitting
- Constraint schema inference (NaN, mixed types, inf values)
- Cache invalidation logic
- Attack projection (CAPGD, HopSkipJump, Square)
- Defence integration (adversarial training, input validation, ensemble)
- Results registry
- Runner pipeline (model dispatch, defence validation)
- Figure generation and statistical tests (paired t-test, Wilcoxon)

## Constraint Handling

The constraint schema handles edge cases:
- **NaN values**: Excluded from min/max computation
- **Mixed types**: Safe sorting for categorical features
- **Infinite values**: Clamped to +/-1e10
- **Constant columns**: Buffer added to avoid min==max
- **Missing tracking**: `has_missing` flag on each feature

## Reproducing Results

```bash
# Run a single experiment
uv run python -m runner.run --config configs/ccfd.yaml --seed 42

# Run all baseline experiments (28 configs x 3 seeds)
uv run python scripts/run_all_seeds.py

# Run ensemble experiments (4 datasets x 3 seeds x 2 attacks = 24 runs)
uv run python scripts/run_ensemble_experiments.py              # all
uv run python scripts/run_ensemble_experiments.py --gpu-only   # CAPGD only
uv run python scripts/run_ensemble_experiments.py --cpu-only   # Square only

# Run z-threshold sweep (8 configs x 1 seed = 8 runs)
uv run python scripts/run_z_threshold_sweep.py

# Run remaining HopSkipJump experiments (parallelized CPU)
uv run python scripts/run_remaining_hsj.py

# Generate figures from results
uv run python scripts/generate_figures.py --registry results/registry_clean.csv

# Run statistical tests (paired t-test + Wilcoxon signed-rank)
uv run python scripts/statistical_tests.py --registry results/registry_clean.csv

# Analysis scripts
uv run python scripts/analyse_adv_training.py --registry results/registry_clean.csv
uv run python scripts/analyse_z_threshold.py --registry results/registry.csv
uv run python scripts/analyse_input_validation.py --registry results/registry_clean.csv
```

Results are logged to `results/registry.csv`. Use `results/registry_clean.csv` (deduplicated) for analysis. Figures are saved to `results/figures/`.

## Known Limitations

- **Adversarial training + tree/ensemble models**: Adversarial training requires gradients (backpropagation) and is incompatible with tree and ensemble models. The runner raises `ValueError` for these combinations.
- **CAPGD + tree models**: CAPGD is a gradient-based (white-box) attack and only works on neural models. For ensemble models, CAPGD targets the MLP component. Use HopSkipJump or Square Attack for tree-only models. In the registry, tree + CAPGD rows show `robust_pr_auc == clean_pr_auc` — this reflects the architectural limitation, not model robustness.
- **Ensemble GPU requirement**: All ensemble experiments require GPU for the MLP training component, even when using black-box attacks (Square, HSJ).
- **HopSkipJump partial coverage**: HSJ experiments are partially complete (6/12 runs). Missing: IEEE-CIS seed 456, LCLD seeds 123 and 456, Sparkov all 3 seeds. Square Attack provides complete black-box coverage. HSJ completion is listed as future work.
- **Sparkov neural vulnerability**: The neural model achieves near-zero robust PR-AUC on Sparkov under CAPGD attack (even at small ε). This reflects extreme vulnerability in a low-dimensional feature space (22 features), not a bug. Tree models achieve robust PR-AUC ≈ 0.747 on the same dataset.

## License

MIT
