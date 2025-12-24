# FRBS: Fraud-Robust Benchmark Suite

A reproducible benchmark suite for evaluating adversarial robustness of fraud detection models. This MVP implements constrained adversarial attacks (CAPGD) and defences on tabular fraud detection datasets.

## Features

- **Datasets**: CCFD (Credit Card Fraud Detection) and IEEE-CIS Fraud Detection
- **Models**: Tree-based (XGBoost) and Neural (MLP) with automatic class weighting
- **Attacks**: CAPGD (Constrained Auto-PGD) with domain-aware projections
- **Defences**: Adversarial training and input validation
- **Metrics**: PR-AUC, precision, recall, F1 (fraud-first evaluation)
- **Reproducibility**: Config-driven experiments with cached splits and preprocessing
- **Robustness**: Handles NaN, mixed types, and edge cases in constraint inference

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
- Note: Tree models don't support gradient-based attacks

## Project Structure

```
MVP/
├── attacks/          # Attack implementations (CAPGD)
├── configs/          # Experiment configurations (YAML)
├── constraints/      # Feature constraint schema and validation
├── datasets/         # Dataset loaders and cards
│   └── cards/        # Dataset documentation (ccfd.md, ieee_cis.md)
├── defences/         # Defence implementations
├── evaluation/       # Metrics and results registry
├── models/           # Model wrappers (Tree, Neural)
├── preprocessing/    # Data preprocessing pipeline
├── results/          # Experiment results and cached artifacts
├── runner/           # Experiment runner
└── tests/            # Unit tests (27 tests)
```

## Datasets

| Dataset | Samples | Features | Fraud Rate | Recommended Model |
|---------|---------|----------|------------|-------------------|
| CCFD | 284,807 | 30 | 0.17% | Neural (all numeric) |
| IEEE-CIS | 590,540 | 392 | 3.5% | Tree (handles categoricals) |

Datasets should be placed at the path specified by `DEFAULT_DATA_ROOT` in `datasets/loader.py`:
```
/path/to/datasets/
├── CCFD/
│   └── creditcard.csv
└── IEEE-CIS/
    └── ieee-fraud-detection/
        ├── train_transaction.csv
        └── train_identity.csv (optional)
```

## Configuration

Experiments are configured via YAML files in `configs/`. Example:

```yaml
experiment_name: "ccfd_baseline"
seed: 42

dataset:
  name: "ccfd"           # or "ieee_cis"
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
  type: "capgd"
  epsilon: 0.1
  steps: 10

defence:
  type: "none"           # or "adversarial_training", "input_validation"
```

## Available Configs

| Config | Dataset | Model | Description |
|--------|---------|-------|-------------|
| `mvp.yaml` | CCFD (10%) | Neural | Quick baseline test |
| `ccfd.yaml` | CCFD (full) | Neural | Full CCFD experiment |
| `ccfd_quick.yaml` | CCFD (5%) | Neural | Fast iteration |
| `ieee_cis.yaml` | IEEE-CIS (10%) | Tree | IEEE-CIS with XGBoost |

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
Note: Does not support gradient-based attacks (CAPGD skipped).

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

## Attacks

### CAPGD (Constrained Auto-PGD)
White-box attack with constraint-aware projection:
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

## Defences

### Adversarial Training
```yaml
defence:
  type: "adversarial_training"
  params:
    epsilon: 0.1
```

### Input Validation
```yaml
defence:
  type: "input_validation"
```

## Testing

Run the test suite (27 tests):

```bash
uv run pytest tests/ -v
```

Test coverage includes:
- Dataset loading and splitting
- Constraint schema inference (NaN, mixed types, inf values)
- Cache invalidation logic
- Attack projection

## Constraint Handling

The constraint schema handles edge cases:
- **NaN values**: Excluded from min/max computation
- **Mixed types**: Safe sorting for categorical features
- **Infinite values**: Clamped to +/-1e10
- **Constant columns**: Buffer added to avoid min==max
- **Missing tracking**: `has_missing` flag on each feature

## License

MIT
