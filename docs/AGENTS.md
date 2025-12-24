# Repository Guidelines

## Project Structure & Module Organization
The MVP is fully implemented with the following structure:
```
MVP/
├── attacks/          # CAPGD attack implementation
├── configs/          # Experiment configs (mvp.yaml, ccfd.yaml, ieee_cis.yaml)
├── constraints/      # Feature constraint schema and validation
├── datasets/         # Dataset loaders (CCFD, IEEE-CIS)
│   └── cards/        # Dataset documentation (ccfd.md, ieee_cis.md)
├── defences/         # Adversarial training, input validation
├── evaluation/       # Metrics (PR-AUC, recall, F1) and results registry
├── models/           # Tree (XGBoost) and Neural (MLP) wrappers
├── preprocessing/    # Data preprocessing pipeline with caching
├── results/          # Experiment results, cached splits/preprocessors
├── runner/           # CLI entrypoint
└── tests/            # Pytest test suite (27 tests)
```

## Build, Test, and Development Commands
Requires Python 3.11+. Using [uv](https://github.com/astral-sh/uv):
```bash
uv sync                                              # Install dependencies
uv run python -m runner.run --config configs/mvp.yaml  # Run experiment
uv run pytest tests/ -v                              # Run test suite (27 tests)
```

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and keep files ASCII unless the dataset requires otherwise.
- Modules and functions use `snake_case`; classes use `CapWords`.
- Config files live in `configs/` and are named for the experiment (for example, `mvp.yaml`).
- Keep the results registry machine-readable (CSV/JSON) with stable column names.

## Testing Guidelines
The test suite uses pytest with 27 tests covering:
- **Dataset loading**: CCFD and IEEE-CIS loaders, split persistence
- **Constraint validation**: NaN handling, mixed types, inf values, constant columns
- **Cache invalidation**: Split indices include dataset size to prevent stale cache
- **Attack projection**: CAPGD constraint-aware projections

Run tests with isolation (uses `tmp_path` fixture to avoid cache conflicts):
```bash
uv run pytest tests/ -v
```

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects with an optional scope (<=72 chars).
- Example: `feat(attacks): add CAPGD projection`
- For pull requests, include a concise description, link related issues, and list the config(s) used.
- Attach key metrics or result artifacts when outputs change.

## Configuration & Results Hygiene
- Keep all experiment settings in `configs/`; runs should be reproducible without code edits.
- Write outputs to `results/`; the registry CSV includes:
  - Metadata: `timestamp`, `experiment_name`, `dataset`, `model_type`, `defence_type`, `attack_type`
  - Attack params: `attack_epsilon`, `validity_rate`, `adv_validity_rate`
  - Metrics: `clean_pr_auc`, `clean_recall`, `robust_pr_auc`, `robust_recall`, `clean_accuracy`, `robust_accuracy`
  - Cost: `train_time_sec`, `attack_time_sec`

### Cache File Naming
Artifacts include dataset size to ensure cache invalidation when `sample_frac` changes:
```
results/split_indices_{dataset}_n{samples}_seed{seed}.json
results/preprocessor_{dataset}_n{samples}_seed{seed}.joblib
```

## Benchmark Results (Latest)

| Dataset | Model | Clean PR-AUC | Robust PR-AUC | Change |
|---------|-------|--------------|---------------|--------|
| CCFD (full) | Neural | 0.758 | 0.638 | -15.8% |
| IEEE-CIS (10%) | Tree | 0.582 | N/A | (no gradient attack) |

Notes:
- CCFD: Neural MLP (128 hidden, 20 epochs), CAPGD (ε=0.1, 10 steps)
- IEEE-CIS: XGBoost (depth 6, 100 estimators) — tree models don't support gradient-based attacks
