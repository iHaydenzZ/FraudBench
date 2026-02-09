# FRBS Full Roadmap: MVP Completion Through Tier 3

> **Created:** 2026-02-09
>
> **Goal:** Expand FRBS from current MVP state to a 4-dataset, 3-attack benchmark suite
> with statistical rigour and thesis-ready figures.
>
> **Tier 2 (hard minimum):** MVP + all 4 datasets + multi-seed + epsilon curves
> **Tier 3 (target):** Tier 2 + black-box attacks (HopSkipJump, Square via ART)

---

## Current State (verified 2026-02-09)

- 33/33 tests passing
- 9 experiment rows in `results/registry.csv`
- MVP success criteria met (CAPGD degrades metrics, adv training improves robust PR-AUC, registry populated)
- 21-column registry schema with F1/precision
- Epsilon sweep done (0.05, 0.1, 0.2)
- Constraint-aware adversarial training implemented
- Input validation with outlier detection + sanitise/reject modes

**Gaps carried forward:**
- `ccfd_tree_adv_train` config never created (experiment #6)
- IEEE-CIS experiments not in registry
- LCLD and Sparkov datasets on disk but no loaders/cards/configs
- `datasets/LCLD/` and `datasets/Sparkov/` not gitignored

---

## Phase 1: MVP Gap Fill

**Goal:** Complete the MVP experiment matrix. Clean baseline before expanding.

### 1.1 Create missing config

Create `configs/ccfd_tree_adv_train.yaml` — CCFD + XGBoost + adversarial training.

### 1.2 Run missing experiments

- `configs/ccfd_tree_adv_train.yaml` (experiment #6)
- `configs/ieee_cis.yaml` (experiment #8 — XGBoost baseline)
- Consider `configs/ieee_cis_neural.yaml` (experiment #7 — Neural MLP on IEEE-CIS)

Verify each adds a row to `results/registry.csv`.

### 1.3 Gitignore cleanup

Add to `.gitignore`:
```
datasets/LCLD/
datasets/Sparkov/
```

### 1.4 Done when

- MVP experiment matrix (8 experiments) fully populated in registry
- No untracked dataset directories in `git status`
- All tests still pass

---

## Phase 2: LCLD + Sparkov Dataset Integration

**Goal:** Expand from 2 to 4 datasets. Biggest code phase.

### 2.1 Explore datasets

Before writing loaders, inspect both datasets to understand:
- Column names, types, target variable
- Missing value rates
- Class balance / fraud rate
- Feature categories (numeric, categorical, binary)

**LCLD (Lending Club Loan Default):**
- Source: `datasets/LCLD/loan.csv`
- Data dictionary: `datasets/LCLD/LCDataDictionary.xlsx`
- Expected: loan amount, interest rate, grade, purpose, etc.
- Target: likely `loan_status` (default/charged-off = fraud-equivalent)

**Sparkov (Synthetic Fraud):**
- Source: `datasets/Sparkov/fraudTrain.csv` + `datasets/Sparkov/fraudTest.csv`
- Decision: **concat and re-split** using our standard stratified splitter for consistency
- Expected: transaction amount, merchant, category, etc.

### 2.2 Implement loaders

Add `load_lcld()` and `load_sparkov()` to `datasets/loader.py`.
Register both in `load_dataset()` dispatcher.

Each loader must:
- Return `DatasetObj` with `X`, `y`, `feature_types`, `feature_names`, `meta`
- Support `sample_frac` parameter
- Handle missing values appropriately for the dataset
- Correctly classify features as `numeric` or `categorical`

For Sparkov: concat train+test files, then let the splitter handle the split.

### 2.3 Write dataset cards

Create `datasets/cards/lcld.md` and `datasets/cards/sparkov.md` with:
- Overview, source, domain
- Dataset statistics (samples, features, fraud rate)
- Label meaning
- Feature description (groups/types)
- Known issues / leakage risks
- Preprocessing notes
- Split strategy
- Citation

### 2.4 Write tests

Add test classes to `tests/test_dataset.py`:
- `TestLCLDLoader` — load, feature types, fraud rate, sampling
- `TestSparkovLoader` — load, feature types, fraud rate, sampling
- Use `@pytest.mark.skipif` for when raw data files aren't present

### 2.5 Create configs

Per dataset, create baseline configs:
- `configs/lcld.yaml` — LCLD + Neural + no defence
- `configs/lcld_tree.yaml` — LCLD + XGBoost + no defence
- `configs/sparkov.yaml` — Sparkov + Neural + no defence
- `configs/sparkov_tree.yaml` — Sparkov + XGBoost + no defence

Defence variants (if time):
- `configs/lcld_input_val.yaml`, `configs/lcld_adv_train.yaml`
- `configs/sparkov_input_val.yaml`, `configs/sparkov_adv_train.yaml`

### 2.6 Run experiments

Run all new configs. Verify registry rows.

### 2.7 Done when

- `load_dataset("lcld")` and `load_dataset("sparkov")` work
- Dataset cards exist under `datasets/cards/`
- Tests pass for both new datasets
- Baseline experiments (neural + tree, no defence) in registry for both datasets
- All existing tests still pass

---

## Phase 3: Multi-Seed Statistical Rigour

**Goal:** Run all experiments with seeds {42, 123, 456} for mean +/- std reporting.

### 3.1 Add `--seed` CLI override

Modify `runner/run.py` to accept `--seed` argument that overrides `config['seed']` at runtime:

```bash
uv run python -m runner.run --config configs/ccfd.yaml --seed 123
```

This avoids config file explosion. Minimal change to the argparse block.

### 3.2 Add seed to registry

Ensure `seed` is recorded as a column in the registry (add if missing) so results
can be grouped by seed for aggregation.

### 3.3 Run experiment matrix

For each (dataset, model, defence) combination, run with all 3 seeds.

**Estimated run count:**
- 4 datasets x 2 models x 3 defences x 3 seeds = ~72 runs
- Minus incompatible combos (tree + adv_training on non-neural) = ~60 valid runs
- Plus epsilon sweeps on CCFD Neural (3 epsilons x 3 seeds = 9 runs)
- Total: ~70 runs

### 3.4 Done when

- `--seed` override works
- Registry has 3 rows per experiment (one per seed)
- All existing tests still pass

---

## Phase 4: Black-Box Attacks via ART

**Goal:** Integrate HopSkipJump and Square Attack from IBM's Adversarial Robustness Toolbox.
This fills the critical gap: CAPGD cannot attack XGBoost.

### 4.1 Add ART dependency

```bash
uv add adversarial-robustness-toolbox
```

### 4.2 Create ART model wrapper

ART requires its own classifier interface. Create `attacks/art_wrapper.py` that wraps
our `BaseModel` (both Tree and Neural) into ART's `ClassifierMixin` interface.

Key methods to implement:
- `predict()` — calls `model.predict_proba()`, returns 2-class probabilities
- `nb_classes` — returns 2
- `input_shape` — returns feature count
- For HopSkipJump (decision-based): only needs hard labels from `predict()`
- For Square (score-based): needs probability outputs from `predict()`

### 4.3 Implement HopSkipJump wrapper

Create `attacks/hopskipjump.py`:
- Thin wrapper around `art.attacks.evasion.HopSkipJump`
- Same interface as CAPGD: `attack(model, X, y, schema, params) -> X_adv`
- Apply constraint projection after attack (project-after strategy)
- Config params: `max_iter`, `max_eval`, `init_eval`

### 4.4 Implement Square Attack wrapper

Create `attacks/square.py`:
- Thin wrapper around `art.attacks.evasion.SquareAttack`
- Same interface as above
- Config params: `max_iter`, `eps`, `norm`

### 4.5 Constraint handling

**Strategy: project-after.**
- Run ART's attack as-is
- Project the resulting `X_adv` through `ConstraintSchema` (bound clipping, non-negativity)
- Report validity rate before and after projection
- If validity is too low (< 80%), upgrade to per-step projection via ART's custom clip callback

### 4.6 Update runner

Modify `runner/run.py` to dispatch to the new attacks based on `attack.type`:
- `"capgd"` — existing path
- `"hopskipjump"` — new path
- `"square"` — new path

For tree models: skip CAPGD (existing behavior), but **run** HopSkipJump/Square.

### 4.7 Create configs

For each dataset, create black-box attack configs:
- `configs/ccfd_hsj.yaml` — CCFD + Neural + HopSkipJump
- `configs/ccfd_tree_hsj.yaml` — CCFD + XGBoost + HopSkipJump (key experiment!)
- `configs/ccfd_square.yaml` — CCFD + Neural + Square
- (Repeat for other datasets as needed)

### 4.8 Write tests

Add `tests/test_black_box_attacks.py`:
- Test ART wrapper produces valid probabilities
- Test HopSkipJump returns perturbed DataFrame
- Test Square returns perturbed DataFrame
- Test constraint projection is applied
- Small synthetic data, fast execution

### 4.9 Run experiments

Priority order:
1. XGBoost + HopSkipJump on all datasets (fills the "tree invulnerability" gap)
2. Neural + HopSkipJump on CCFD (comparison with CAPGD)
3. Square Attack on key combos

### 4.10 Done when

- `attack.type: hopskipjump` and `attack.type: square` work in configs
- XGBoost can be attacked (robust metrics populated)
- Tests pass for both new attacks
- Registry has black-box attack results

---

## Phase 5: Figures & Reporting

**Goal:** Thesis-ready figures and summary tables, generated from registry.

### 5.1 Create figure generation script

`scripts/generate_figures.py` — reads `results/registry.csv`, outputs to `results/figures/`.

Run via:
```bash
uv run python scripts/generate_figures.py
```

### 5.2 Core figures

1. **Robustness curves** — Robust PR-AUC vs epsilon, one line per defence, per dataset.
   Hero figure for the thesis.

2. **Defence comparison bar charts** — Grouped bars: clean vs robust PR-AUC per defence.
   One chart per dataset.

3. **Attack comparison table** — CAPGD vs HopSkipJump vs Square:
   - Attack success rate (% of samples with reduced confidence)
   - Constraint validity rate
   - Compute cost (time per sample)
   One table per dataset.

4. **Cross-dataset summary table** — All 4 datasets side by side.
   Shows which findings generalize vs are dataset-specific.

5. **Error bars** — Mean +/- std from 3 seeds on all figures.

### 5.3 When to run

This script can be run incrementally:
- After Phase 1: CCFD robustness curve + defence comparison
- After Phase 2: Expand to all 4 datasets
- After Phase 3: Add error bars
- After Phase 4: Add black-box attack comparisons

### 5.4 Done when

- All 5 figure types generated as PNG/PDF in `results/figures/`
- Figures use consistent styling (font size, color scheme)
- Script is reproducible from CLI

---

## Dependency Graph

```
Phase 1 (MVP gaps)
    |
    v
Phase 2 (LCLD + Sparkov)
    |
    v
Phase 3 (Multi-seed) ---------> Phase 5 (Figures) [incremental]
    |
    v
Phase 4 (Black-box attacks) --> Phase 5 (Figures) [final]
```

Phases 1-3 = Tier 2 (hard minimum).
Phase 4 = Tier 3 (target).
Phase 5 runs incrementally alongside all phases.

---

## Config Schema Additions

New config fields needed across phases:

```yaml
# Phase 2: new dataset names
dataset:
  name: "lcld"       # or "sparkov"

# Phase 4: new attack types
attack:
  type: "hopskipjump"
  max_iter: 50
  max_eval: 1000

attack:
  type: "square"
  max_iter: 1000
  eps: 0.1
  norm: "linf"       # or "l2"
```

CLI additions:
```bash
# Phase 3: seed override
uv run python -m runner.run --config configs/ccfd.yaml --seed 123
```

---

## Full Experiment Matrix (Final Target — Tier 3)

**Dimensions:**
- Datasets (4): CCFD, IEEE-CIS, LCLD, Sparkov
- Models (2): Neural MLP, XGBoost
- Defences (3): None, Adversarial Training, Input Validation
- Attacks (3): CAPGD (neural only), HopSkipJump, Square
- Seeds (3): 42, 123, 456

**Valid combinations:** ~180 runs (excluding CAPGD on XGBoost)
**Plus epsilon sweeps:** ~30 additional runs

**Registry columns (21):** timestamp, experiment_name, dataset, model_type, defence_type,
attack_type, attack_epsilon, validity_rate, adv_validity_rate, clean_pr_auc,
clean_precision, clean_recall, clean_f1, robust_pr_auc, robust_precision,
robust_recall, robust_f1, clean_accuracy, robust_accuracy, train_time_sec, attack_time_sec

**(+1 in Phase 3):** seed
