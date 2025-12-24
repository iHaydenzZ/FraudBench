# FRBS: Fraud-Robust Benchmark Suite

A reproducible benchmark suite for evaluating adversarial robustness of fraud detection models. This MVP implements constrained adversarial attacks (CAPGD) and defences on tabular fraud detection datasets.

## Features

- **Datasets**: CCFD (Credit Card Fraud Detection) and IEEE-CIS Fraud Detection
- **Models**: Tree-based (XGBoost) and Neural (MLP) classifiers
- **Attacks**: CAPGD (Constrained Auto-PGD) with domain-aware projections
- **Defences**: Adversarial training and input validation
- **Metrics**: PR-AUC, precision, recall, F1 (fraud-first evaluation)
- **Reproducibility**: Config-driven experiments with saved splits and preprocessing

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

## Project Structure

```
MVP/
├── attacks/          # Attack implementations (CAPGD)
├── configs/          # Experiment configurations (YAML)
├── constraints/      # Feature constraint schema and validation
├── datasets/         # Dataset loaders and cards
│   └── cards/        # Dataset documentation
├── defences/         # Defence implementations
├── evaluation/       # Metrics and results registry
├── models/           # Model wrappers (Tree, Neural)
├── preprocessing/    # Data preprocessing pipeline
├── reports/          # Generated reports (output)
├── results/          # Experiment results and artifacts
├── runner/           # Experiment runner
└── tests/            # Unit tests
```

## Datasets

| Dataset | Samples | Features | Fraud Rate | Type |
|---------|---------|----------|------------|------|
| CCFD | 284,807 | 30 | 0.17% | All numeric (PCA) |
| IEEE-CIS | 590,540 | 392 | 3.5% | Mixed numeric/categorical |

Datasets should be placed in the path specified by `DEFAULT_DATA_ROOT` in `datasets/loader.py`.

## Configuration

Experiments are configured via YAML files in `configs/`. Example:

```yaml
experiment_name: "ccfd_baseline"
seed: 42

dataset:
  name: "ccfd"
  test_size: 0.2
  val_size: 0.2
  sample_frac: 0.1  # Optional: sample for faster iteration

model:
  type: "neural"  # or "tree"
  params:
    epochs: 15
    hidden_dim: 128

attack:
  type: "capgd"
  epsilon: 0.1
  steps: 10

defence:
  type: "none"  # or "adversarial_training", "input_validation"
```

## Available Configs

| Config | Description |
|--------|-------------|
| `mvp.yaml` | CCFD baseline with neural model (10% sample) |
| `ccfd.yaml` | Full CCFD experiment |
| `ccfd_quick.yaml` | Quick CCFD test (5% sample, 5 epochs) |
| `ieee_cis.yaml` | IEEE-CIS experiment (10% sample) |

## Running Experiments

```bash
# Default MVP config
uv run python -m runner.run --config configs/mvp.yaml

# Full CCFD experiment
uv run python -m runner.run --config configs/ccfd.yaml

# IEEE-CIS experiment
uv run python -m runner.run --config configs/ieee_cis.yaml
```

## Results

Results are logged to `results/registry.csv` with columns:
- Experiment metadata (dataset, model, defence, attack)
- Clean metrics (PR-AUC, recall, accuracy)
- Robust metrics (under attack)
- Validity rates and timing

Reproducibility artifacts are saved:
- `results/split_indices_{dataset}_seed{N}.json` - Train/val/test splits
- `results/preprocessor_{dataset}_seed{N}.joblib` - Fitted preprocessor

## Models

### Tree Model (XGBoost)
```yaml
model:
  type: "tree"
  params:
    max_depth: 6
    n_estimators: 100
```

### Neural Model (MLP)
```yaml
model:
  type: "neural"
  params:
    epochs: 20
    hidden_dim: 128
    batch_size: 256
    class_weight: true  # Handle imbalance (default: true)
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

Run the test suite:

```bash
uv run pytest tests/ -v
```

## Evaluation Metrics

- **PR-AUC**: Primary metric (handles class imbalance)
- **Precision/Recall/F1**: Fraud class performance
- **Accuracy**: Secondary metric
- **Validity Rate**: Constraint satisfaction of adversarial examples

## License

MIT
