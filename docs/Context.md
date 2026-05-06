# FraudBench: Project Context

> **Last updated:** 2026-05-06
> **Note:** §1–§8 describe the original MVP benchmark (stable since Feb 21). §9 documents the constraint-aware evaluation arc that has been the active research line since April 2026 (target: **ICAIF 2026, deadline 2026-08-02**). §10 documents the FA-AT (Fraud-Aware AT) work that became the primary research line on 2026-05-06 and feeds both the **thesis (draft 2026-05-15, final 2026-05-29)** and the ICAIF paper. **Master plan:** `docs/FraudBench_Thesis_ICAIF_Plan.md`.

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

**Current position (MVP benchmark):** Tier 3 complete (Square), Tier 4 partial (Ensemble done). HopSkipJump 6/12 (ccfd done; ieee_cis 2/3; lcld 1/3; sparkov 0/3). All CAPGD + epsilon sweep experiments complete. Ensemble experiments complete (24 runs). Deduplicated registry at `results/registry_clean.csv` (182 rows). All figures and analysis artefacts fixed and regenerated (Feb 21).

**Active research line (April 2026):** Constraint-aware evaluation arc targeting ICAIF 2026 (~July deadline). Phase 1 (cross-dataset feasibility audit) and Phase 3 (LCLD g1+M1 projection) complete; Phase 2 (cross-dataset extension) at 50% — IEEE-CIS OHE-projection MVP done 2026-04-22, M+OHE follow-up next. See §9 for full status and `docs/constraint_evaluation_guidance.md` for the strategic plan.

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

---

## 9. Constraint-Aware Evaluation Arc (April 2026 main line)

**Goal:** ICAIF 2026 paper (deadline **2026-08-02**, 8-page ACM 2-column, no appendix). Hypothesis: adversarial robustness on financial tabular data must account for domain constraints, attacker capabilities, and feature semantics — and stock CAPGD with post-hoc constraint filtering systematically *underestimates* realistic attack success.

**Strategy doc:** `docs/constraint_evaluation_guidance.md`
**Master plan (thesis + ICAIF, dual-track):** `docs/FraudBench_Thesis_ICAIF_Plan.md`

### Findings docs (chronological)

| Date | Finding | Notebook(s) | Doc |
|------|---------|-------------|-----|
| 2026-04-15 | Mask ablation M0–M6 on LCLD: monotone robust accuracy gradient (0.042 → 0.340), PR-AUC locked at 0.1051 across all 8 variants | `mask_ablation.ipynb` | `mask_ablation_findings.md` |
| ~2026-04-15 | TabularBench metric audit: accuracy vs F1/MCC ranking divergence (Kendall τ < 0.74); 10/70 degenerate TabNet models | `tabularbench_comparison.ipynb`, `tabularbench_metric_analysis.ipynb` | `tabularbench_comparison_findings.md` |
| 2026-04-22 | Cross-dataset Phase 1 feasibility audit: refuted a-priori richness gradient; binary "constrained vs unconstrained" dichotomy; OHE-validity is universal binding constraint | `cross_dataset_feasibility.ipynb` | `cross_dataset_feasibility_findings.md` |
| 2026-04-22 | LCLD g1-projection + M1+g1 (initial): reported 0.05% → 50.2% → 95.3% — superseded by 2026-04-28 fix below | `g1_projection_attack.ipynb` | `g1_projection_findings.md` |
| 2026-04-28 | LCLD g1+M1 corrected (EVAL_TOL fix, commit `326483d`): filtered success **0.11% → 76.5% → 100.0%** with flip-count delta ≤1 (same-model). Soft blocker resolved — root cause was float64 round-trip drift on integer-valued g3 columns, not a sparse-categorical artifact. | `g1_projection_attack.ipynb` | `g1_projection_findings.md` |
| 2026-04-22 | IEEE-CIS OHE-projection MVP (Phase 2 cross-dataset replication): filtered success 0.00% → 59.7% with flip-count delta ≤5 (same-model); residual gap is `i_d_nonneg` | `ieee_cis_ohe_projection_attack.ipynb` | `ieee_ohe_projection_findings.md` |

### Headline numbers (paper §5)

| Dataset | Stock CAPGD FSR | Constraint-aware FSR | Stock adv feasibility | Binding constraint after projection |
|---|---:|---:|---:|---|
| **LCLD** | 0.11% | **100.0%** (M1+g1) | 0.12% | g3 (closed by M1, after EVAL_TOL fix) |
| **IEEE-CIS** | 0.00% | **59.7%** (OHE-only; M+OHE pending → ~95% expected) | 0.014% | i_d_nonneg (closeable by M-mask) |
| **Sparkov** | — (not yet attacked) | — | 0.38% | s_state / s_category / s_gender OHE |
| **CCFD** | — (no constraint to filter) | — | 100% | None (PCA-anonymised) |

### Roadmap status (per `constraint_evaluation_guidance.md` §5)

| Phase | Status |
|-------|--------|
| Phase 1: Cross-dataset feasibility audit | ✅ Done (2026-04-22) |
| Phase 2: Cross-dataset M+OHE replication | 🟡 50% — IEEE-CIS OHE + M+OHE done (2026-04-29), Sparkov OHE deprioritised (paper-table polish, not logical necessity) |
| Phase 3: LCLD g1+M1 projection | ✅ Done (2026-04-28, post EVAL_TOL fix) |
| Phase 4: Novel defence (fraud-aware AT / FA-AT) | 🟢 **Active primary line as of 2026-05-06** — see §10 |

### Outstanding soft blockers

1. **CCFD robust PR-AUC variance** (0.58 ± 0.23 across 3 seeds). Fix: more seeds or longer training; otherwise note in paper.
2. ~~**EVAL_TOL fix — scope narrowed by audit (2026-04-29).**~~ — **resolved 2026-04-29** (commit `d52208f`). Audit narrowed scope from 4 notebooks to 1: `mask_ablation`, `tabularbench_comparison`, `tabularbench_metric_analysis` already use `TOLERANCE = 0.01` on g2/g3 (TabularBench `EqualConstraint` convention) which absorbs ULP drift ~10¹³× over; only `cross_dataset_feasibility.ipynb` used strict `<=`. Fix applied + verified on Colab same day: per-seed clean g3 / g2 = **1.0000** across all 3 seeds (was 0.9617 / 0.9966 avg), aggregate LCLD clean_feasibility = **0.998 ± 0.0002** (was 0.956 ± 0.059, variance collapsed ~300×). Adv g3 also rose 0.692 → 0.777 (+8.6pp) — false-failure removal on adv rows the attacker didn't touch, not a capability change. Two-tolerance convention documented in `constraint_evaluation_guidance.md` §9.

> **Resolved 2026-04-28:** the previously-listed "LCLD seed-42 sparse-categorical issue" was misdiagnosed; root cause was float64 round-trip drift on integer-valued constraint columns, fixed by `EVAL_TOL = 1e-6` in `notebooks/g1_projection_attack.ipynb` (commit `326483d`).

### Cross-references

- Strategic plan + dataset constraint inventory: `docs/constraint_evaluation_guidance.md`
- Per-finding details (numbers, caveats, methodology): the 5 findings docs above
- Implementation plans (where applicable): `docs/plans/`

---

## 10. FA-AT Primary Research Line + Dual-Track Writing (May 2026)

**As of 2026-05-06**, FA-AT (Fraud-Aware Adversarial Training) was promoted from "deferred Phase 4" to the primary research contribution, with a dual-track writing schedule:

| Milestone | Date | Type |
|---|---|---|
| Thesis draft (supervisor review) | **2026-05-15** | Internal — structure / method / ablation plan; experimental results not required |
| Thesis final | **2026-05-29** | ≥ 40 pages, accepts negative results + long limitations |
| ICAIF 2026 submission | **2026-08-02** | 8 pages ACM 2-column, no appendix, CMT system |

**Master plan:** `docs/FraudBench_Thesis_ICAIF_Plan.md` — contains tier specs (Tier 0/1/2), 88-day phased plan (Phase 1A 5/06–5/15, Phase 1B 5/16–5/29, Phase 2 5/30–8/02), Plan B trigger at 5/22 mid-point, task IDs E1–E11 / W1–W13 / I1–I6.

### Tiered FA-AT specification

| Tier | Target | Required outputs |
|---|---|---|
| Tier 0 (by 5/15 draft) | Method *plausibility* | Per-feature ε formula, cost-sensitive loss form, mutability classification table, ablation **plan**, LCLD single-seed sanity check (loss decreases) |
| Tier 1 (by 5/29 thesis) | Thesis-grade evidence | 3 datasets (LCLD + Sparkov + IEEE-CIS), 3 seeds, Wilcoxon + Cohen's d, 2 ablations (cost on/off × per-feature ε on/off), cross-attack transfer, §D degenerate audit |
| Tier 2 (by 8/02 ICAIF) | Single-claim narrative | 4 datasets, 5 seeds, compact 2×2 ablation table, cross-attack robustness figure, vs. Foe for Fraud differentiation |

### 5/22 Plan B trigger

Mid-point of Phase 1B. FA-AT vs. standard AT effect size determines ICAIF framing:

| Cohen's d observation | ICAIF framing |
|---|---|
| ≥ 0.5 on ≥ 2 datasets | **A** — FA-AT primary contribution |
| 0.2 – 0.5 | A but weakened; expand dataset count in Phase 2 |
| < 0.2 across all datasets | **B** — OHE structural floor primary, FA-AT secondary (negative-finding framing) |

Thesis (5/29) accepts negative results regardless — Plan B only affects the 8-page ICAIF rewrite.

### Writing strategy

Thesis and ICAIF paper are **two separate documents**, not compress/expand siblings. Phase 2 (5/30 onward) **rewrites** the 8-page paper from a blank document; thesis content is the experimental repository it draws from, not a draft to compress.
