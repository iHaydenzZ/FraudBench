# FraudBench: To-Do List

> **Last updated:** 2026-05-06
> **Current active work:** FA-AT (Fraud-Aware AT) as primary research line, feeding both thesis (draft 2026-05-15, final 2026-05-29) and ICAIF 2026 (deadline **2026-08-02**). See FA-AT + Tier 0/1/2 section below.
> **Master plan:** `docs/FraudBench_Thesis_ICAIF_Plan.md`
> **Legacy MVP estimated remaining work:** P1 ~4-6h (HSJ + docs), All ~8-14h — see P0/P1/P2 sections below.

---

## Dual-track Schedule (Thesis + ICAIF)

| Milestone | Date | What ships |
|---|---|---|
| Thesis draft (Phase 1A end) | **2026-05-15** | Method spec + ablation plan + Tier 0 sanity check; experimental results NOT required |
| Plan B trigger (Phase 1B mid) | **2026-05-22** | Decide FA-AT framing based on Cohen's d on ≥ 2 datasets |
| Thesis final (Phase 1B end) | **2026-05-29** | ≥ 40 pages incl. Tier 1 results (3 datasets × 3 seeds, 2 ablations, cross-attack transfer, §D degenerate audit) |
| ICAIF submission (Phase 2 end) | **2026-08-02** | 8 pages ACM 2-col, no appendix, **rewritten from blank doc** (not compressed thesis) |

---

## ICAIF 2026 + Thesis — Constraint-Aware Evaluation + FA-AT (current main line)

> Master plan: `docs/FraudBench_Thesis_ICAIF_Plan.md`
> Strategic doc: `docs/constraint_evaluation_guidance.md`
> Status snapshot: `docs/Context.md` §9 (constraint-aware) + §10 (FA-AT + dual-track)
> 5 findings docs: `mask_ablation_findings.md`, `tabularbench_comparison_findings.md`, `cross_dataset_feasibility_findings.md`, `g1_projection_findings.md`, `ieee_ohe_projection_findings.md`

### A. ~~Highest priority -- IEEE-CIS M+OHE follow-up~~ -- **DONE (2026-04-29)**

`notebooks/ieee_cis_ohe_projection_attack.ipynb` Cells 13–17 (commits `385420f` + `567159d`). Result documented in `ieee_ohe_projection_findings.md` "Central finding 2".

- **Outcome:** FSR saturates at **100%** (matching LCLD M1+g1); agg feasibility 0.483 → 1.000; D-non-negativity gap closed (i_d_nonneg 0.480 ± 0.41 → 1.000 ± 0.0001, 4000× variance reduction)
- **But:** feasible-flipped count crashed from ~120 to ~7.7 per seed (16× drop). Robust accuracy bounced back to 0.897 (essentially clean). Asymmetric vs LCLD M1+g1, where attack count *increased* (~2200 → ~2888)
- **Reframed as paper finding:** capability and feasibility are separate axes that compose differently across datasets. LCLD's mutable subset overlaps strongly with predictive features; IEEE-CIS's predictive signal lives in 339 opaque V-features that any realistic mutability profile must freeze
- **Open follow-up promoted from "highest priority" to "next decision":** mutable-set sensitivity sweep on IEEE-CIS to map the dose-response curve along the capability axis (see §A' below)

### A'. Next decision — IEEE-CIS mutable-set sensitivity sweep

Optional but defensible. Bracket the M+OHE attack-count number by running:
- **Tighter M:** just `TransactionAmt` + `ProductCD` (~6 mutable processed dims) — expected feas-flip ≈ 3
- **Wider M:** add `P_emaildomain`, `R_emaildomain`, `M1`–`M9` (~30 mutable processed dims) — expected feas-flip somewhere between 7.7 and 120

Produces a 3–4-point dose-response curve for the §5 trade-off claim. Estimated effort: 2–3 cells, ~20 min compute. Skip if paper-table polish vs Sparkov OHE-projection (§C) is the higher-priority next item.

### B. Soft blockers (`constraint_evaluation_guidance.md` §5)

- [x] ~~LCLD seed-42 sparse-categorical fix~~ — **resolved 2026-04-28** (commit `326483d`). Root cause was float64 round-trip drift on integer-valued g3 columns, not a sparse-categorical artifact. Fix: `EVAL_TOL = 1e-6` in `check_g2` / `check_g3`. New M1+g1 filtered success: **100%** (was 95.3%). See `g1_projection_findings.md` §"Methodology fix".
- [x] ~~**Propagate EVAL_TOL fix — scope narrowed by audit (2026-04-29).**~~ **resolved 2026-04-29** (commit `d52208f`). Audit narrowed scope from 4 notebooks to 1: `mask_ablation`, `tabularbench_comparison`, `tabularbench_metric_analysis` already use `TOLERANCE = 0.01` on g2/g3 (TabularBench `EqualConstraint` convention) which absorbs ULP drift ~10¹³× over; only `cross_dataset_feasibility.ipynb` used strict `<=`. Verified on Colab same day: per-seed clean g3 / g2 = **1.0000** across all 3 seeds (was 0.9617 / 0.9966 avg); LCLD clean_feasibility = **0.998 ± 0.0002** (was 0.956 ± 0.059). Two-tolerance convention documented in `constraint_evaluation_guidance.md` §9.
- [ ] CCFD robust PR-AUC variance: more seeds or longer training; otherwise note in paper.

### C. Lower priority -- Sparkov OHE-projection

Analog of IEEE-CIS OHE-projection. Three OHE blocks: state (0.0002), category (0.017), gender (0.265) stock adv pass. Two datasets already establish the cross-dataset pattern; a third is paper-table polish, not a logical necessity.

### D. Phase 1 residual (per `constraint_evaluation_guidance.md` §5)

- [ ] Cross-attack robustness transfer (CAPGD adv examples vs Square / HSJ)
- [ ] Degenerate model audit on FraudBench (TabularBench analog)

### E. **Active primary line — FA-AT (Fraud-Aware Adversarial Training)**

Promoted from "deferred Phase 4" to primary research contribution on 2026-05-06. Tiered specification — see `docs/FraudBench_Thesis_ICAIF_Plan.md` §2 for full task list (E1–E11, W1–W13, I1–I6).

#### Tier 0 — by 2026-05-15 thesis draft (method *plausibility*, not results)

- [ ] FA-AT method section: per-feature ε mapping formula + cost-sensitive loss form + algorithm pseudocode (W8)
- [ ] Mutability classification table per dataset (fully / semi / immutable)
- [ ] Ablation **plan** (motivation + design — not results) (W9 stub)
- [ ] LCLD single-seed sanity check: code runs, loss decreases (E4)
- [ ] OHE projection repair implementation (FA-AT baseline) (E3)
- [ ] §A' IEEE-CIS sensitivity sweep (~20 min compute) (E1)
- [ ] HopSkipJump remaining runs (E2)

#### Tier 1 — by 2026-05-29 thesis final (thesis-grade evidence)

- [ ] FA-AT multi-seed (3 seeds) on LCLD + Sparkov + IEEE-CIS (E7) — **largest effort**
- [ ] 2 ablations: (i) cost weighting on/off keeping per-feature ε; (ii) per-feature ε on/off keeping cost weighting (E8)
- [ ] Cross-attack transfer: CAPGD-trained model vs. Square / HSJ attacks (E9)
- [ ] §D Degenerate model audit (FraudBench analog of TabularBench's TabNet finding) (E10)
- [ ] Constraint feasibility report: g1 / g4 satisfaction rate of FA-AT generated adv examples
- [ ] Cross-dataset OHE failure rate table (E6)
- [ ] vs. standard AT head-to-head on robust PR-AUC (Wilcoxon signed-rank + Cohen's d)
- [ ] 4 dataset cards in formal style (W11)

#### 2026-05-22 mid-point checkpoint — Plan B trigger

Decision point based on FA-AT vs. standard AT effect size:

| Cohen's d observation | Action | ICAIF framing |
|---|---|---|
| ≥ 0.5 on ≥ 2 datasets | Continue Tier 1 + Tier 2 | A — FA-AT primary |
| 0.2 – 0.5 | Tier 1 ships in thesis; Phase 2 expands dataset count | A weakened |
| < 0.2 across all | Thesis honestly reports negative finding | **B** — OHE structural floor primary, FA-AT secondary |

Thesis (5/29) accepts negative results either way. Plan B only changes the 8-page ICAIF framing.

#### Tier 2 — by 2026-08-02 ICAIF submission (single-claim narrative)

Phase 2 (5/30 → 8/02) **rewrites** the paper from a blank document. Do not compress thesis.

- [ ] FA-AT extension to 4 datasets, 5 seeds (E11) — **largest effort**
- [ ] Compact 2×2 ablation table (per-feature ε on/off × cost weighting on/off)
- [ ] Cross-attack robustness figure (1)
- [ ] vs. Foe for Fraud (Aug 2025) differentiation paragraph + experiments
- [ ] Robust PR-AUC + constraint-aware feasibility dual-metric reporting
- [ ] **Cut from ICAIF (live in thesis instead):** long ablation tables, supplementary, detailed dataset cards, §D degenerate audit
- [ ] Reproducibility package: uv lock + configs (I1)
- [ ] Quickstart notebook (I3)
- [ ] Pip-installable package (I4)
- [ ] Static leaderboard scaffold (I5)
- [ ] Extensibility guide (I6)
- [ ] ICAIF 8-page rewrite (W13)

#### Risks tracked in master plan §6

1. FA-AT not significant → 5/22 trigger → Plan B
2. 8 pages + no appendix means Phase 2 cannot compress thesis — must rewrite
3. Foe for Fraud (Aug 2025) differentiation: per-feature ε is *defense-time*; their transferability is *attack-time*. Different axes — but reviewers won't see this automatically; must state explicitly
4. Cross-attack transfer negative results (FA-AT robust to CAPGD but not HSJ) → handle in Limitations
5. OHE projection repair on LCLD only fixes g4; g1 nonlinear remains dominant killer (~98% failure rate)

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
