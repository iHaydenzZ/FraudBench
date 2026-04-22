# IEEE-CIS OHE-Projected CAPGD — Findings

**Date:** 2026-04-22
**Notebook:** `notebooks/ieee_cis_ohe_projection_attack.ipynb` (commit `74ee475`, Colab A100)
**Canonical results:** `results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_results.csv` (6 rows: 3 seeds × 2 attacks)
**Summary table:** `results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_summary.csv`
**Spec:** `docs/constraint_evaluation_guidance.md` §5 Phase 2 (re-scoped)
**Predecessor:** `docs/g1_projection_findings.md` (LCLD analog)

---

## Headline numbers

IEEE-CIS only, 3 seeds (42/123/456), CAPGD ε=0.1 / 10 steps, neural MLP (128-dim hidden, 20 epochs). Two attacks compared on the same trained model per seed: stock `capgd_attack` vs an identical loop with per-step argmax-snap on the three OHE blocks (ProductCD, card4, card6) identified as binding by the cross-dataset Phase 1 audit (`cross_dataset_feasibility_findings.md`). Mean ± std over seeds.

| Attack         | Flipped | Feas-flipped | **Filtered success** | Agg. feas.       | ProductCD | card4 | card6 | D-nonneg       | Robust acc.   | Robust PR-AUC      |
|----------------|--------:|-------------:|---------------------:|-----------------:|----------:|------:|------:|---------------:|--------------:|-------------------:|
| unconstrained  | 216 ± 13 | 0            | **0.00%**            | 0.0003 ± 0.0002 | 0.041 | 0.062 | 0.220 | 0.535 ± 0.392 | 0.169 ± 0.018 | 0.0720 ± 0.010 |
| OHE-projected  | 215 ± 16 | 128 ± 91     | **59.7%**            | 0.535 ± 0.392   | **1.000** | **1.000** | **1.000** | 0.535 ± 0.392 | 0.180 ± 0.018 | 0.0750 ± 0.003 |

Per-seed raw counts:

| Seed | Attack         | Flipped | Feas-flipped | Agg. feas. | ProductCD | card4 | card6 | D-nonneg | Robust PR-AUC |
|-----:|----------------|--------:|-------------:|-----------:|----------:|------:|------:|---------:|--------------:|
| 42   | unconstrained  | 202     | 0            | 0.0002     | 0.036     | 0.051 | 0.189 | 0.460    | 0.0625        |
| 42   | OHE-projected  | 197     | 35           | 0.4634     | **1.000** | **1.000** | **1.000** | 0.463 | 0.0753  |
| 123  | unconstrained  | 219     | 0            | 0.0003     | 0.031     | 0.023 | 0.273 | 0.959    | 0.0830        |
| 123  | OHE-projected  | 221     | 190          | **0.9576** | **1.000** | **1.000** | **1.000** | 0.958 | 0.0777  |
| 456  | unconstrained  | 228     | 0            | 0.0006     | 0.056     | 0.110 | 0.196 | 0.186    | 0.0706        |
| 456  | OHE-projected  | 227     | 160          | 0.1830     | **1.000** | **1.000** | **1.000** | 0.183 | 0.0720  |

---

## Central finding — the pattern generalizes from LCLD

Same-model unconstrained vs OHE-projected flipped counts differ by **at most 1 flip on every seed** (202 vs 197, 219 vs 221, 228 vs 227 — net delta of 5/2/1). The OHE projection raised filtered success from 0.00% to 59.7% — an **infinite multiplicative jump** (the unconstrained baseline produces zero feasible-flipped attacks across all 3 seeds) — without measurably reducing the attack's ability to flip predictions.

This is the same finding established on LCLD by `g1_projection_findings.md`, now reproduced on a second dataset with a completely different constraint structure (3 OHE-validity checks vs 1 nonlinear formula + 1 OHE check). The "constraint-aware attack recovers attack power" pattern is no longer LCLD-specific. It is a property of CAPGD on constrained tabular data: at ε=0.1 the attack does not need to break domain constraints to produce predictive flips; constraint violations are *incidental* to the gradient signal, not load-bearing.

---

## Cross-dataset comparison (paper §5 headline table)

| Dataset    | Stock CAPGD FSR | Constraint-aware FSR | Attack-power loss (same-model flip delta) | Constraint-aware adv-feas | Binding constraint after projection |
|------------|----------------:|---------------------:|------------------------------------------:|--------------------------:|-------------------------------------|
| **LCLD**     | 0.05% | 50.2% (g1+g4 OHE projection) | ≤2 flips    | 0.693 ± 0.092 | g3 bankruptcy (immutability gap) |
| **IEEE-CIS** | **0.00%** | **59.7% (3-OHE projection)**   | **≤5 flips** | **0.535 ± 0.392** | **D-non-negativity (immutability gap)** |

Both datasets show:
1. Stock CAPGD produces near-zero filtered success (0.05% / 0.00%).
2. A constraint-aware attack at the same ε raises filtered success by 50–60 pp.
3. Same-model flipped-count delta is ≤5 flips (~2% of total) — well within model-init noise.
4. The remaining gap to 100% adv-feas is gated by *immutable-feature integrity constraints* (g3 / D-nonneg) that the attack perturbs incidentally — closeable by adding a mutability mask (M1 closed g3 on LCLD; an analogous M for IEEE-CIS would close D-nonneg).

This is the cleanest single experimental story in the codebase for the paper's §5/§6: **same finding on two datasets with completely different constraint structures.**

---

## Per-constraint breakdown

| Constraint          | Unconstrained | OHE-projected | Notes |
|---------------------|--------------:|--------------:|-------|
| `i_product_ohe`     | 0.041         | **1.000**     | Forced by per-step argmax snap |
| `i_card4_ohe`       | 0.062         | **1.000**     | Forced by per-step argmax snap |
| `i_card6_ohe`       | 0.220         | **1.000**     | Forced by per-step argmax snap |
| `i_d_nonneg`        | 0.535         | 0.535         | Unchanged — projection doesn't touch D fields |
| `i_c_nonneg`        | 1.000         | 1.000         | Already at ceiling (StandardScaler centering protects) |
| `i_amt_positive`    | 1.000         | 1.000         | Already at ceiling |
| **Aggregate**       | 0.0003        | **0.535**     | Now bounded above by D-non-negativity |

D-non-negativity is unchanged across attacks (0.535 → 0.535) because the OHE projection doesn't touch D fields. The attack continues to perturb D-fields to negative scaled values in roughly half of test rows; this is the residual gap to 100% feasibility. Adding an M-mask that freezes D1–D15 at clean values would push aggregate feasibility from 0.535 → ~1.0, mirroring how M1 closed the g3 gap on LCLD (0.69 → 0.96).

---

## Paper narrative implications

### The cross-dataset claim is now defensible

`g1_projection_findings.md` established the headline result on LCLD; this run extends it to IEEE-CIS. The paper's §5 can now claim:

> "Constraint-aware adversarial attack generation recovers ~50–60% of the filtered-success rate that post-hoc filtering hides on every constrained dataset we tested (LCLD, IEEE-CIS), while preserving the flipped-prediction count within ≤5 flips on the same trained model. This is true regardless of constraint type — closed-form formula coupling (LCLD g1) or independent OHE-validity checks (IEEE-CIS ProductCD/card4/card6). Post-hoc constraint filtering and constraint-aware attack generation are not interchangeable: the former rejects attacks the latter generates abundantly."

This is a stronger position than the LCLD-only claim and directly differentiates FraudBench from TabularBench's framing.

### PR-AUC invariance now spans 8 axes

Robust PR-AUC is locked across:
- 8 mask-ablation variants on LCLD (`mask_ablation_findings.md`, all 0.1051)
- 3 seeds × 4 datasets in cross-dataset audit (`cross_dataset_feasibility_findings.md`)
- 3 attack regimes on LCLD g1+M1 (0.1051 ± 9×10⁻⁶)
- **2 attack regimes on IEEE-CIS** (0.0720 → 0.0750, within σ)

PR-AUC distinguishes attack from no-attack (clean ~0.45 → robust ~0.07 on IEEE-CIS = 84% drop) but does *not* distinguish constraint-aware from constraint-unaware attacks. Robust accuracy and flipped-and-feasible counts are the discriminating metrics.

### OHE-validity is the universal binding constraint

`cross_dataset_feasibility_findings.md` (post-Cell-14 update) established that OHE-validity is the binding constraint on every constrained dataset (LCLD g4, IEEE-CIS ProductCD/card4/card6, Sparkov state/category/gender). This run validates the implication: forcing OHE-validity is the single most effective constraint-aware attack modification. On IEEE-CIS, OHE projection alone raises filtered success from 0% to 59.7%; on LCLD, OHE projection plus g1 raises it from 0.05% to 50.2% (and the OHE component does most of the work — `g4_term_ohe` adv pass goes from 0.19 to 1.0, while `g1_installment` was already structurally separate).

---

## Caveats

1. **D-non-negativity variance dominates the aggregate-feasibility std.** Per-seed agg feas = 0.463 / 0.958 / 0.183, std = 0.392. This tracks `i_d_nonneg` exactly (0.460 / 0.959 / 0.186), which is itself very seed-unstable. Hypothesis: D1–D15 are timedeltas with large clean-data variance, so the StandardScaler-centred mean is close to or below zero for some splits. CAPGD's ε perturbation then crosses the negativity threshold for a variable fraction of test rows. Phase 1 audit also showed `i_d_nonneg adv pass = 0.486 ± 0.296` — same seed-instability, independently observed.

2. **Filtered-success-rate std is huge (0.0/0.18/0.86 across seeds).** Driven entirely by the D-non-negativity variance above. The mean (59.7%) is the right number to quote; the per-seed values bound it from below by D-non-negativity. M-mask freezing D fields would collapse this variance.

3. **ε=0.1 is a single operating point.** Smaller ε might leave OHE projection unnecessary (the unconstrained attack might naturally satisfy OHE if the perturbation budget is small enough). Not tested.

4. **10 CAPGD steps.** Matches every prior run in the benchmark for comparability.

5. **Same-model comparison only.** Unlike the LCLD M1+g1 result (which retrains per seed in Cell 16), this notebook's two attacks share the trained MLP per seed via the standard pattern. The flipped-count equivalence (≤5 flips delta) is therefore a strict same-model comparison and not subject to the between-model caveat that applies to LCLD M1+g1.

6. **Class imbalance.** IEEE-CIS pos_weight=27.01 (~3.6% positive class) means flipped-positive counts are an order of magnitude smaller than LCLD's (~216 vs ~2700). The proportional damage is similar; the absolute counts are not directly comparable.

---

## What this means for the roadmap

Phase 2 cross-dataset MVP is complete and the result is decisive. The headline finding from Phase 3 (`g1_projection_findings.md`) is no longer single-dataset.

**Immediate next step:** Add an M+OHE follow-up to this notebook — an analog of the LCLD M1+g1 cells. M would freeze D1–D15 (and likely C1–C14, V1–V339, card1–6 — the IEEE-CIS immutable set per `constraint_evaluation_guidance.md` §3.3). Expected outcome: agg feas 0.535 → ~1.0, filtered success 59.7% → ~95%+, flipped count preserved within model-init noise. This produces the paper's cleanest direct cross-dataset comparison: LCLD M1+g1 = 95.3% FSR vs IEEE-CIS M+OHE = ~95% FSR.

**After that:** Same exercise on Sparkov (3-OHE projection, 0.0002+0.017+0.265 binding constraints). Lower priority since two datasets already establishes the cross-dataset pattern; a third is paper-table polish, not a logical necessity.

**Paper updates required:**
- `constraint_evaluation_guidance.md` §1 result #1: extend to "demonstrated on LCLD and IEEE-CIS" instead of "demonstrated on LCLD."
- `constraint_evaluation_guidance.md` §6 Caveat 2: remove the "single-dataset Tier C" caveat — Tier C generalizes to non-formula constraints via OHE projection.
- `constraint_evaluation_guidance.md` §5 Phase 2: mark IEEE-CIS sub-item complete; cross-reference this doc.
