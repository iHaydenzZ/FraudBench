# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Philosophy

### Iterate Fast
- Start with the simplest working version (MVP), then improve incrementally.
- "Make it work -> Make it right -> Make it fast." Never try to do all three at once.
- Hard-code first, abstract later. Premature abstraction is worse than duplication.
- Small, frequent commits -- one logical change per commit.

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
- **Structure as pipelines**: Ingest -> Process -> Output. Separate I/O from core logic.
- **Externalize config**: No magic numbers. Extract paths, thresholds, and hyperparameters into a config object or file.
- **Use structured logging** (`logging`, not `print`). Log input stats, timing, and state changes.
- **Fail gracefully**: Wrap risky I/O/network calls in try/catch. One item's failure must not crash the batch.
- **Checkpoint long tasks**: Save progress periodically so work can resume from the last good state.
- Write tests that target likely fault points, not just happy paths.

## Commands

```bash
# Install dependencies
uv sync

# Run an experiment
uv run python -m runner.run --config configs/mvp.yaml

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_constraints.py -v

# Run a single test by name
uv run pytest tests/test_dataset.py -v -k "test_split_sizes"

# Generate figures from registry
uv run python scripts/generate_figures.py

# Batch run all experiments
uv run python scripts/run_all_seeds.py

# Run remaining HopSkipJump experiments (parallelized CPU)
uv run python scripts/run_remaining_hsj.py
```

## Project Context

See `docs/Context.md` for architecture, experiment state, and research findings.
See `docs/ToDo.md` for remaining tasks and priorities.

**Canonical results:** Two registries, different purposes:
- **ICDM 2026 paper**: `results/icdm_2026/icdm_master_registry.csv` (390 rows) + companion tables — see `results/icdm_2026/README.md`. Regenerate companion tables with `uv run python scripts/generate_leaderboards.py`.
- **Thesis / legacy analysis**: `results/registry_clean.csv` (182 deduplicated rows). Raw registries (`registry.csv`, `20260216_GPU_only_registry.csv`) contain duplicates and superseded data.

**CI locally:** The lockfile pins CUDA-only torch, so `uv run` fails on macOS ARM. Use the venv directly: `ruff check .`, `ruff format --check .`, `.venv/bin/pytest tests/ -v -m "not slow"`.
