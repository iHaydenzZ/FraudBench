# Repository Guidelines

## Project Structure & Module Organization
This repository currently contains `Init_plan.md`, which defines the MVP scope. The intended MVP layout (per that plan) is:
- `datasets/` for loaders and `datasets/cards/` for dataset cards
- `models/`, `attacks/`, `defences/`, and `constraints/` for core components
- `runner/` for the CLI/entrypoint and `configs/` for experiment configs
- `results/` and `reports/` for generated outputs

## Build, Test, and Development Commands
No runnable pipeline is committed yet. When the MVP runner is implemented, the expected entrypoint is:
- `python -m runner.run --config configs/mvp.yaml` — run a config-driven experiment
If tests are added, use a single top-level command so they can run in CI (e.g., `pytest`).

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and keep files ASCII unless the dataset requires otherwise.
- Modules and functions use `snake_case`; classes use `CapWords`.
- Config files live in `configs/` and are named for the experiment (for example, `mvp.yaml`).
- Keep the results registry machine-readable (CSV/JSON) with stable column names.

## Testing Guidelines
No testing framework or coverage target is defined yet. When adding tests:
- Place them under `tests/` and name files `test_*.py` (pytest-compatible).
- Prioritize dataset loading, constraint validation, and attack projection logic.

## Commit & Pull Request Guidelines
There is no Git history in this repo to infer a convention. Until one is established:
- Use short, imperative commit subjects with an optional scope (<=72 chars).
- Example: `feat(attacks): add CAPGD projection`.
For pull requests, include a concise description, link related issues, and list the config(s) used. Attach key metrics or result artifacts when outputs change.

## Configuration & Results Hygiene
- Keep all experiment settings in `configs/`; runs should be reproducible without code edits.
- Write outputs to `results/` and `reports/`, and include dataset/model/defence/attack identifiers plus cost metrics in the registry.
