# FraudBench: To-Do List

> **Last updated:** 2026-04-28
> **Current active work:** Constraint-aware evaluation arc for ICAIF 2026 (~July deadline). See ICAIF section below.
> **Legacy MVP estimated remaining work:** P1 ~4-6h (HSJ + docs), All ~8-14h — see P0/P1/P2 sections below.

---

## ICAIF 2026 -- Constraint-Aware Evaluation (current main line)

> Strategic doc: `docs/constraint_evaluation_guidance.md`
> Status snapshot: `docs/Context.md` §9
> 5 findings docs: `mask_ablation_findings.md`, `tabularbench_comparison_findings.md`, `cross_dataset_feasibility_findings.md`, `g1_projection_findings.md`, `ieee_ohe_projection_findings.md`
> Conference deadline: ~2026-07 (8-page ACM format)

### A. Highest priority -- IEEE-CIS M+OHE follow-up

Build on `notebooks/ieee_cis_ohe_projection_attack.ipynb`. Add an M-mask layer to the existing OHE-projected attack:

- **Mask:** mutable = `TransactionAmt`, `ProductCD` OHE, `addr1/2`, `dist1/2` (per `constraint_evaluation_guidance.md` §3.3); freeze D1–D15, C1–C14, V1–V339, card1–6
- **Expected outcome:** agg feasibility 0.535 → ~1.0; FSR 59.7% → ~95%; flipped-count preserved within model-init noise
- **Why it matters:** Produces the paper's cleanest cross-dataset comparison row (LCLD M1+g1 = **100%** vs IEEE-CIS M+OHE ≈ 95-100%); closes the residual D-non-negativity gap noted in `ieee_ohe_projection_findings.md`
- **Effort:** ~3 cells (mask definition, M+OHE attack loop, results table)

### B. Soft blockers (`constraint_evaluation_guidance.md` §5)

- [x] ~~LCLD seed-42 sparse-categorical fix~~ — **resolved 2026-04-28** (commit `326483d`). Root cause was float64 round-trip drift on integer-valued g3 columns, not a sparse-categorical artifact. Fix: `EVAL_TOL = 1e-6` in `check_g2` / `check_g3`. New M1+g1 filtered success: **100%** (was 95.3%). See `g1_projection_findings.md` §"Methodology fix".
- [ ] **Propagate EVAL_TOL fix — scope narrowed by audit (2026-04-29).** Audit of the four LCLD-touching notebooks found that `mask_ablation`, `tabularbench_comparison`, and `tabularbench_metric_analysis` already use `TOLERANCE = 0.01` on their g2/g3 inequality checks (inherited from TabularBench's `EqualConstraint` convention) — that margin absorbs the ~1e-16 ULP drift ~10¹³× over, so they were never affected. Only `cross_dataset_feasibility.ipynb` used strict `<=`. Fix applied there 2026-04-29; awaits Colab re-run to verify per-seed clean g3 pass reaches ~1.0 across all 3 seeds (pre-fix avg 0.9617). The two-tolerance convention is now documented in `constraint_evaluation_guidance.md` §9.
- [ ] CCFD robust PR-AUC variance: more seeds or longer training; otherwise note in paper.

### C. Lower priority -- Sparkov OHE-projection

Analog of IEEE-CIS OHE-projection. Three OHE blocks: state (0.0002), category (0.017), gender (0.265) stock adv pass. Two datasets already establish the cross-dataset pattern; a third is paper-table polish, not a logical necessity.

### D. Phase 1 residual (per `constraint_evaluation_guidance.md` §5)

- [ ] Cross-attack robustness transfer (CAPGD adv examples vs Square / HSJ)
- [ ] Degenerate model audit on FraudBench (TabularBench analog)

### E. Deferred -- Phase 4 novel defence

Fraud-aware AT (per-feature ε allocation, cost-sensitive AT). Per `constraint_evaluation_guidance.md` §5: only if time remains after Phase 2 + paper writing.

---

## P0 -- Must Fix (Affects Result Credibility) [Legacy MVP]

### ~~1. Re-run IEEE-CIS Experiments~~ -- DONE

Re-run completed on Feb 12-16. New IEEE-CIS rows show `validity_rate ~0.997`. Old rows from Feb 9-10 have been excluded from `results/registry_clean.csv` (the deduplicated canonical registry created Feb 20).

---

### 2. Document Input Validation Finding

Input validation consistently degrades robustness. This is a genuine finding (not a bug) -- CAPGD already respects constraints, so the z-score clipping destroys discriminative signal without blocking adversarial perturbations.

**Action:** Write up in thesis Discussion section with evidence (fraction of features clipped). Optionally test with different `z_threshold` values (5.0, 10.0) to show the effect.

---

## P1 -- Required for Benchmark Status

### 3. Run Black-Box Attack Experiments -- PARTIAL

Tree models (XGBoost) are immune to gradient-based CAPGD. Black-box attacks are needed to evaluate tree model robustness.

**Square Attack: DONE (12/12).** All 4 datasets x 3 seeds complete in `registry_clean.csv`.

**HopSkipJump: 6/12 done.** Remaining 6 experiments:

```bash
# Remaining HopSkipJump experiments (use scripts/run_remaining_hsj.py for parallelized CPU runs)
uv run python -m runner.run --config configs/ieee_cis_tree_hsj.yaml --seed 456
uv run python -m runner.run --config configs/lcld_tree_hsj.yaml --seed 123
uv run python -m runner.run --config configs/lcld_tree_hsj.yaml --seed 456
uv run python -m runner.run --config configs/sparkov_tree_hsj.yaml --seed 42
uv run python -m runner.run --config configs/sparkov_tree_hsj.yaml --seed 123
uv run python -m runner.run --config configs/sparkov_tree_hsj.yaml --seed 456
```

HSJ experiments are slow (~7-11h each). Use `scripts/run_remaining_hsj.py` for parallelized CPU execution.

**Verify:** Tree model rows show `robust_pr_auc < clean_pr_auc`.

---

### ~~4. Epsilon Sweeps -- Multi-Seed~~ -- DONE

All 4 datasets have 3-seed epsilon sweeps (eps = {0.01, 0.05, 0.1, 0.15, 0.2, 0.3}) completed on Feb 16. Duplicates resolved in `registry_clean.csv` (keeps latest timestamp per experiment+seed+epsilon).

---

### 5. Reproducibility Documentation

- [ ] Dataset download instructions (exact URLs + placement for all 4 datasets)
- [ ] Full reproduction command (`scripts/run_all_seeds.py` documented in README)
- [ ] Config field documentation (all YAML fields explained, including hopskipjump/square)
- [ ] README update (add LCLD, Sparkov, HopSkipJump, Square to README)

---

### 6. Document Tree + Adversarial Training Gap

4 configs are N/A (XGBoost + adversarial training across all datasets). Gradient-based adversarial training is architecturally inapplicable to tree models.

**Action:** Mark as "N/A" in benchmark results table with explanation. This is itself a meaningful finding about model-family-dependent defence applicability.

---

### ~~7. Auto-Generated Figures and Report~~ -- DONE

`scripts/generate_figures.py` produces 6 figure types + summary CSV. All figures fixed and regenerated (Feb 21):
- Robustness curves now show all defences (single-epsilon as scatter markers)
- Robustness bars filtered to ε=0.1 only (readable labels)
- Defence heatmap includes ensemble column (vs neural baseline)
- Summary table annotates tree+CAPGD rows with † footnote
- Statistical tests support cross-model ensemble comparisons
- Input validation analysis uses strict ε=0.1 filter

See `docs/FIX_DOCUMENT.md` for issue details and `docs/plans/2026-02-21-fix-figures-and-analysis.md` for the implementation plan.

---

## P2 -- Enhancements (Strengthen the Benchmark)

### 8. Transferability Experiments

Test whether adversarial examples from Neural models fool XGBoost (and vice versa).

```bash
uv run python -m scripts.transferability --dataset ccfd --seed 42 --epsilon 0.1
uv run python -m scripts.transferability --dataset ieee_cis --seed 42 --epsilon 0.1
uv run python -m scripts.transferability --dataset lcld --seed 42 --epsilon 0.1
uv run python -m scripts.transferability --dataset sparkov --seed 42 --epsilon 0.1
```

**Estimated effort:** 3-5 hours.

---

### ~~9. Statistical Significance Testing~~ -- DONE

`scripts/statistical_tests.py` runs pairwise paired t-tests + Wilcoxon signed-rank + Cohen's d. Cross-model ensemble comparisons added Feb 21. Results in `results/figures/statistical_tests.csv`.

---

### ~~10. Ensemble Defence~~ -- DONE

Ensemble model (LR + XGBoost + MLP with soft voting) implemented in `defences/ensemble.py` and `models/ensemble.py`. 24 experiment runs complete in registry. Configs: `*_ensemble.yaml` and `*_ensemble_square.yaml`.

---

### 11. CTGAN Data Augmentation

Train CTGAN on fraud-class samples to generate synthetic fraud data for training augmentation. Fourth defence method from Progress Report.

**Estimated effort:** 4-8 hours.

---

### 12. Pre-Trained Model Zoo

Save trained model weights for each (dataset, model_type, defence) to `results/models/` so researchers can evaluate new attacks without retraining.

**Estimated effort:** 2-4 hours.

---

## Post-Experiment Tasks

After all GPU experiments complete:

```bash
# Regenerate figures
uv run python scripts/generate_figures.py

# Statistical tests
uv run python scripts/statistical_tests.py

# Input validation analysis
uv run python scripts/analyse_input_validation.py
```

---

## Summary

| # | Task | Type | Experiments | Priority | Status |
|---|------|------|-------------|----------|--------|
| 1 | ~~Re-run IEEE-CIS (NaN fix)~~ | GPU | 15 | P0 | **Done** |
| 2 | Document input validation finding | Writing | -- | P0 | Pending |
| 3 | Black-box attacks (HSJ + Square) | CPU/GPU | 24 | P1 | **Partial** (Square done, HSJ 6/12) |
| 4 | ~~Epsilon sweeps multi-seed~~ | GPU | 8 (x6 eps) | P1 | **Done** |
| 5 | Reproducibility docs | Writing | -- | P1 | Pending |
| 6 | Document tree+adv_train gap | Writing | -- | P1 | Pending |
| 7 | ~~Generate figures~~ | Local | -- | P1 | **Done** |
| 8 | Transferability experiments | GPU | 4 | P2 | Pending |
| 9 | ~~Statistical tests~~ | Local | -- | P2 | **Done** |
| 10 | ~~Ensemble defence~~ | Code+GPU | 24 | P2 | **Done** |
| 11 | CTGAN augmentation | Code+GPU | TBD | P2 | Pending |
| 12 | Model zoo | GPU | -- | P2 | Pending |

**Total GPU/CPU experiments remaining:** 6 (P1: remaining HSJ) or ~10 (all including transferability)

**Recommended:** Run `uv run python scripts/run_remaining_hsj.py` for parallelized CPU execution of remaining HSJ experiments.
