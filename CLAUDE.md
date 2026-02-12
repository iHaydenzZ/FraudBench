# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Philosophy

### Iterate Fast
- Start with the simplest working version (MVP), then improve incrementally.
- "Make it work → Make it right → Make it fast." Never try to do all three at once.
- Hard-code first, abstract later. Premature abstraction is worse than duplication.
- Small, frequent commits — one logical change per commit.

### Design Principles (apply during "make it right" phase, not before)
- **YAGNI**: Only build what's needed now. Do not design for imaginary future requirements.
- **KISS**: Prefer the simplest solution that works. Complexity must be justified.
- **DRY**: Eliminate duplication only when a pattern appears 3+ times (Rule of Three).
- **Least Astonishment**: Code should behave as a reader would intuitively expect.
- **Law of Demeter**: Minimize knowledge between objects; avoid deep chaining (a.b.c.d).

### SOLID (apply when refactoring, not on first draft)
- **SRP**: One reason to change per class/module.
- **OCP**: Extend via new code, not by modifying existing stable code.
- **LSP**: Subtypes must be substitutable for their base types.
- **ISP**: Small, focused interfaces over large, general ones.
- **DIP**: Depend on abstractions, not concrete implementations.

### Code Quality Checklist
- High cohesion, low coupling. Use polymorphism over type-based conditionals.
- Encapsulate what varies behind stable interfaces (Protected Variations).
- **Structure as pipelines**: Ingest → Process → Output. Separate I/O from core logic.
- **Externalize config**: No magic numbers. Extract paths, thresholds, and hyperparameters into a config object or file.
- **Use structured logging** (`logging`, not `print`). Log input stats, timing, and state changes.
- **Fail gracefully**: Wrap risky I/O/network calls in try/catch. One item's failure must not crash the batch.
- **Checkpoint long tasks**: Save progress periodically so work can resume from the last good state.
- Write tests that target likely fault points, not just happy paths.

## Project Overview

FRBS (Fraud-Robust Benchmark Suite) evaluates adversarial robustness of fraud detection models. It benchmarks XGBoost and Neural (MLP) models against constrained adversarial attacks (CAPGD) on real fraud datasets (CCFD, IEEE-CIS), with optional defences (adversarial training, input validation).

## Commands

```bash
# Install dependencies
uv sync

# Run an experiment
uv run python -m runner.run --config configs/mvp.yaml

# Run all tests (27 tests)
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_constraints.py -v

# Run a single test by name
uv run pytest tests/test_dataset.py -v -k "test_split_sizes"
```

## Architecture

**Config-driven pipeline**: YAML configs in `configs/` drive the entire experiment flow. The runner (`runner/run.py`) orchestrates a 9-step pipeline:

Load Config → Load Dataset → Split (stratified, cached) → Preprocess (StandardScaler/OneHotEncoder) → Train Model → Evaluate Clean → Infer Constraints → Run Attack → Log Results

**Key abstractions**:
- `models/base.py`: `BaseModel` ABC with `fit()` and `predict_proba()` — all models implement this interface
- `constraints/schema.py`: `ConstraintSchema` inferred from training data, handles NaN/inf/mixed types; used by CAPGD for constraint-aware projection
- `evaluation/registry.py`: Logs results to `results/registry.csv`

**Data flow**: `datasets/loader.py` loads raw data → `datasets/splitter.py` creates stratified train/val/test splits (cached by dataset_name + seed + n_samples) → `preprocessing/processor.py` fits StandardScaler + OneHotEncoder (cached via joblib)

**Attack pipeline**: `attacks/capgd.py` implements Constrained Auto-PGD. It perturbs fraud samples within feature bounds defined by `ConstraintSchema`, then the model is re-evaluated to measure robustness degradation.

**Defences**: `defences/adversarial_training.py` (PGD-3 augmentation during training), `defences/input_validation.py` (constraint-based clipping at inference)

## Config Schema

```yaml
experiment_name: "name"
seed: 42
dataset:
  name: "ccfd"          # or "ieee_cis"
  test_size: 0.2
  val_size: 0.2
  sample_frac: 0.1      # fraction of dataset to use
model:
  type: "neural"         # or "tree"
  params:
    epochs: 15           # neural only
    hidden_dim: 128      # neural only
    batch_size: 256      # neural only
attack:
  type: "capgd"
  epsilon: 0.1
  steps: 10
defence:
  type: "none"           # or "adversarial_training", "input_validation"
```

## Testing

Tests use pytest with `tmp_path` fixtures for isolation. Some dataset tests use `@pytest.mark.skipif` to skip when raw data files are not present. Test files:
- `test_dataset.py` — loader, splitter, caching, stratification
- `test_constraints.py` — schema inference, NaN/inf handling, validation
- `test_attacks.py` — constraint projection, CAPGD integration

## Caching

Split indices and preprocessor artifacts are cached in `results/`. Cache keys include dataset name, seed, and sample count — cache auto-invalidates if the dataset size changes.

## Colab Integration

### Development Workflow

1. **Local development**: Write and debug code locally using `uv run` and `uv run pytest`.
2. **Push to GitHub**: `git add . && git commit -m "..." && git push`
3. **Run on Colab**: Open `notebooks/colab_runner.ipynb`, pull latest code, run experiments.
4. **Results on Drive**: All experiment outputs are saved to Google Drive automatically.

### Important Notes

- Colab does NOT have `uv`. Use `pip install -e .` instead.
- The runner command is the same on both local and Colab:
  - Local:  `uv run python -m runner.run --config configs/mvp.yaml`
  - Colab:  `python -m runner.run --config configs/mvp.yaml`
- Datasets live on Google Drive at `/content/drive/MyDrive/FraudBench/data/`
- Individual dataset dirs are symlinked into `datasets/` (not the whole folder, since it contains Python source).

### Compute Unit Budget

Approximate Colab burn rates:
- T4 GPU: ~1.96 units/hour
- A100 GPU: ~12.46 units/hour
- CPU: Free (no unit consumption)

**Always use T4 unless A100 is specifically needed.** Debug on CPU runtime.
