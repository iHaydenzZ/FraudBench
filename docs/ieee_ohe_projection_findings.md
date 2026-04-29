# IEEE-CIS OHE-Projected CAPGD — Findings

**Date:** 2026-04-22 (M+OHE follow-up added 2026-04-29)
**Notebook:** `notebooks/ieee_cis_ohe_projection_attack.ipynb` (commits `74ee475` / `567159d`, Colab A100)
**Canonical results:** `results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_results.csv` (9 rows: 3 seeds × 3 attacks)
**Summary table:** `results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_summary.csv`
**Spec:** `docs/constraint_evaluation_guidance.md` §5 Phase 2 (re-scoped)
**Predecessor:** `docs/g1_projection_findings.md` (LCLD analog)

---

## Headline numbers

IEEE-CIS only, 3 seeds (42/123/456), CAPGD ε=0.1 / 10 steps, neural MLP (128-dim hidden, 20 epochs). Three attacks compared per seed:

1. **unconstrained** — stock `capgd_attack`.
2. **OHE-projected** — stock loop with per-step argmax-snap on the three OHE blocks (ProductCD, card4, card6) identified as binding by the cross-dataset Phase 1 audit (`cross_dataset_feasibility_findings.md`).
3. **M + OHE** — adds an M-mask layer freezing all features except `TransactionAmt`, `ProductCD` OHE, `addr1/2`, `dist1/2` (per `constraint_evaluation_guidance.md` §3.3). 10 of 537 processed dimensions remain mutable.

(1) and (2) share a trained MLP per seed (same-model comparison); (3) retrains per seed (between-model — see Caveat 5). Mean ± std over seeds.

| Attack          | Flipped     | Feas-flipped | **Filtered success** | Agg. feas.        | ProductCD | card4 | card6 | D-nonneg          | Robust acc.   | Robust PR-AUC   |
|-----------------|------------:|-------------:|---------------------:|------------------:|----------:|------:|------:|------------------:|--------------:|----------------:|
| unconstrained   | 220.7 ± 2.1 | 0            | **0.0%**             | 0.0001 ± 0.0001   | 0.044     | 0.062 | 0.248 | 0.480 ± 0.413     | 0.145 ± 0.016 | 0.077 ± 0.003   |
| OHE-projected   | 220.0 ± 2.0 | 120 ± 80     | **54.5%**            | 0.483 ± 0.411     | **1.000** | **1.000** | **1.000** | 0.483 ± 0.411 | 0.155 ± 0.014 | 0.082 ± 0.003   |
| **M + OHE**     | **7.7 ± 1.2** | **7.7 ± 1.2** | **100.0%**       | **1.000 ± 0.0001**| **1.000** | **1.000** | **1.000** | **1.000 ± 0.0001** | **0.897 ± 0.020** | **0.402 ± 0.046** |

Per-seed raw counts:

| Seed | Attack         | Flipped | Feas-flipped | Agg. feas. | ProductCD | card4 | card6 | D-nonneg  | Robust PR-AUC |
|-----:|----------------|--------:|-------------:|-----------:|----------:|------:|------:|----------:|--------------:|
| 42   | unconstrained  | 219     | 0            | 0.0000     | 0.060     | 0.023 | 0.217 | 0.386     | 0.0772        |
| 42   | OHE-projected  | 220     | 30           | 0.3974     | **1.000** | **1.000** | **1.000** | 0.397 | 0.0817      |
| 42   | **M + OHE**    | **9**   | **9**        | **1.0000** | **1.000** | **1.000** | **1.000** | **1.000** | **0.3901** |
| 123  | unconstrained  | 220     | 0            | 0.0003     | 0.021     | 0.035 | 0.297 | 0.931     | 0.0803        |
| 123  | OHE-projected  | 218     | 147          | **0.9293** | **1.000** | **1.000** | **1.000** | 0.929 | 0.0860      |
| 123  | **M + OHE**    | **7**   | **7**        | **0.9998** | **1.000** | **1.000** | **0.9999** | **1.000** | **0.4520** |
| 456  | unconstrained  | 223     | 0            | 0.0002     | 0.051     | 0.129 | 0.230 | 0.121     | 0.0750        |
| 456  | OHE-projected  | 222     | 183          | 0.1217     | **1.000** | **1.000** | **1.000** | 0.122 | 0.0793      |
| 456  | **M + OHE**    | **7**   | **7**        | **1.0000** | **1.000** | **1.000** | **1.000** | **1.000** | **0.3632** |

> **Note on numerics.** The 2026-04-22 MVP commit (`74ee475`) reported headline numbers of 215–216 flipped / 128 feas-flipped / 59.7% FSR for OHE-projected. The 2026-04-29 re-run that added M+OHE re-executed Cells 10–12 from scratch and produced the slightly different numbers above (220 flipped / 120 feas-flipped / 54.5% FSR) due to non-deterministic CUDA training. The qualitative findings are unchanged; the table reflects the canonical CSV on disk.

---

## Central finding 1 — OHE projection generalizes the LCLD pattern

Same-model unconstrained vs OHE-projected flipped counts differ by **at most 5 flips on every seed** (219 vs 220, 220 vs 218, 223 vs 222 — net delta ≤ 5/seed). The OHE projection raised filtered success from 0.0% to 54.5% — an **infinite multiplicative jump** (the unconstrained baseline produces zero feasible-flipped attacks across all 3 seeds) — without measurably reducing the attack's ability to flip predictions.

This is the same finding established on LCLD by `g1_projection_findings.md`, now reproduced on a second dataset with a completely different constraint structure (3 OHE-validity checks vs 1 nonlinear formula + 1 OHE check). The "constraint-aware attack recovers attack power" pattern at the OHE-projection step is **not** LCLD-specific. It is a property of CAPGD on constrained tabular data: at ε=0.1 the attack does not need to break domain constraints to produce predictive flips; constraint violations are *incidental* to the gradient signal, not load-bearing.

---

## Central finding 2 — but M+OHE reveals a capability-vs-feasibility asymmetry

Where LCLD's M1+g1 saturates filtered success at 100% **and** preserves attack count (~2888/seed feasible-flipped per `g1_projection_findings.md`), IEEE-CIS M+OHE saturates filtered success at 100% **but crashes attack count from ~120 to ~7.7 feasible-flipped per seed (16× drop).** Robust accuracy bounces from 0.155 back to 0.897 — essentially clean accuracy 0.894 — and robust PR-AUC from 0.082 back to 0.402 (clean is 0.446). The model is barely degraded under M+OHE.

The two datasets therefore behave **asymmetrically** under the M+structure progression:

|                          | Stock CAPGD                | OHE/g1 projection             | M + structure                              |
|--------------------------|---------------------------:|------------------------------:|-------------------------------------------:|
| **LCLD** absolute feasible-flipped | 3                  | ~2200                         | ~2888 (peak)                               |
| **IEEE-CIS** absolute feasible-flipped | 0              | 120                           | 7.7 (16× **lower** than OHE-only)          |
| LCLD trajectory          | weak                       | strong                        | strongest                                  |
| IEEE-CIS trajectory      | weak                       | strongest                     | feasibility-saturated, attack-collapsed    |

Per-positive-row normalization does not rescue the symmetry: LCLD M1+g1 ≈ 2888/3500 ≈ 82% of positive-class rows attacked feasibly; IEEE-CIS M+OHE = 7.7/220 ≈ 3.5%. Both datasets have feasibility-saturated attacks at peak, but the *rate* at which a realistic capability profile is attackable differs by an order of magnitude.

The structural reason is dimensional. LCLD's M1 leaves 22 of 63 raw features mutable (~35%), and those 22 carry strong predictive signal (loan_amnt, dti, purpose, addr_state, etc.). IEEE-CIS's M leaves 6 of 53 interpretable raw features mutable (~11%), and the 339 opaque V-features that the model leans on are 100% frozen. ε=0.1 perturbations on 10 of 537 processed dimensions don't carry enough gradient signal to budge the model.

This asymmetry is itself a result. **Capability constraints and feasibility constraints are not freely composable.** A mutability profile tight enough to guarantee feasibility may be tight enough to neutralize the attack — and which regime applies depends on whether the dataset's predictive signal lives in mutable or immutable features. On LCLD the predictive features and the mutable features overlap heavily (a borrower's loan amount, DTI, and address are both attacker-controlled and strongly predictive). On IEEE-CIS they don't (Vesta's V-features dominate predictions but are immutable internal aggregates).

---

## Cross-dataset comparison (paper §5 headline table)

| Dataset    | Stock CAPGD FSR | Constraint-aware FSR (peak) | Peak feas-flipped per seed | Same-model flip delta | Binding constraint after projection | Capability vs feasibility |
|------------|----------------:|----------------------------:|---------------------------:|-----------------------:|-------------------------------------|--------------------------|
| **LCLD**     | 0.05% | 50.2% (g1+OHE) → **100.0% (M1+g1)** | 2888 (M1+g1) | ≤2 flips (g1+OHE)     | g3 bankruptcy (immutability gap)        | **Composable** — attack count rises monotonically across regimes |
| **IEEE-CIS** | **0.0%** | **54.5% (3-OHE)** → **100.0% (M+OHE)** | **120 (3-OHE) / 7.7 (M+OHE)** | **≤5 flips (3-OHE)** | **D-non-negativity (immutability gap)** | **Trade-off** — M+OHE saturates feasibility but cuts attack count 16× |

Both datasets show:
1. Stock CAPGD produces near-zero filtered success.
2. A constraint-aware attack at the same ε raises filtered success by 50–60 pp without measurably reducing attack count (≤5 flips on the same model).
3. Adding M-mask saturates filtered success at 100% by closing the immutability gap (g3 / D-non-negativity).

But they differ in step 3's *attack-count* effect. LCLD's M1 adds attack power (~2200 → ~2888). IEEE-CIS's M removes it (~120 → ~7.7). This is the central new finding from the M+OHE follow-up.

---

## Per-constraint breakdown

| Constraint          | Unconstrained   | OHE-projected | M + OHE           | Notes |
|---------------------|----------------:|--------------:|------------------:|-------|
| `i_product_ohe`     | 0.044           | **1.000**     | **1.000**         | Forced by per-step argmax snap |
| `i_card4_ohe`       | 0.062           | **1.000**     | **1.000**         | Forced by argmax (card4 also frozen under M) |
| `i_card6_ohe`       | 0.248           | **1.000**     | **1.000**         | Forced by argmax (card6 also frozen under M) |
| `i_d_nonneg`        | 0.480 ± 0.413   | 0.483 ± 0.411 | **1.000 ± 0.0001**| Frozen under M → seed-instability collapses to zero |
| `i_c_nonneg`        | 1.000           | 1.000         | 1.000             | Already at ceiling; M freezes it for safety |
| `i_amt_positive`    | 1.000           | 1.000         | 1.000             | Already at ceiling |
| **Aggregate**       | 0.0001 ± 0.0001 | 0.483 ± 0.411 | **1.000 ± 0.0001**| Now bounded above by floating-point only |

The OHE-projected → M+OHE progression closes the D-non-negativity gap completely (0.483 → 1.000) and **collapses the seed-instability std from 0.413 to 0.0001 — a 4000× variance reduction.** This is the cleanest empirical evidence in the codebase for the prediction in `cross_dataset_feasibility_findings.md` that D-non-negativity variance was traceable to D-field perturbation and would vanish under feature freezing. It also predicts the M-mask behavior on similar timedelta-style fields in other tabular datasets.

---

## Paper narrative implications

### The cross-dataset claim is now a richer story (not a simpler one)

`g1_projection_findings.md` established the headline OHE/g1-projection result on LCLD; this run extends it to IEEE-CIS at the OHE-projection step (54.5% FSR with ≤5-flip same-model delta). At the M+structure step, the two datasets diverge: LCLD M1+g1 saturates at 100% FSR while *increasing* attack power, IEEE-CIS M+OHE saturates at 100% FSR while *crashing* attack power. The paper §5 framing should be:

> "At the OHE-projection / g1-projection step, constraint-aware adversarial attacks recover ~50–55pp of filtered success that post-hoc filtering hides on every constrained dataset we tested (LCLD, IEEE-CIS), with same-model flipped-count delta ≤5 flips. Adding a mutability mask saturates filtered success at 100% by closing immutability gaps (g3 / D-non-negativity) on both datasets — but the attack-count effect is dataset-dependent. On LCLD, M1+g1 increases absolute feasible attack count from ~2200 to ~2888; on IEEE-CIS, M+OHE decreases it from ~120 to ~7.7. The asymmetry tracks the relative information density of mutable vs immutable features: LCLD's mutable subset overlaps strongly with predictive features, while IEEE-CIS's predictive signal lives in 339 opaque V-features that any realistic mutability profile must freeze."

This is **not** the clean-symmetric paper claim the original ToDo predicted, but it is a stronger result. It distinguishes capability and feasibility as separate axes that compose differently across datasets, and identifies the structural property (mutable-vs-predictive overlap) that determines which regime applies.

### PR-AUC invariance breaks under M+OHE

Robust PR-AUC was previously locked at 0.07 ± 0.01 across stock vs OHE-projected attacks on IEEE-CIS (matching the "8-axis invariance" pattern documented elsewhere). Under M+OHE it jumps to 0.402 ± 0.046 — recovering most of the way from robust 0.082 toward clean 0.446. **Robust PR-AUC discriminates capability-saturated attacks from capability-suppressed attacks**, where it does not discriminate constraint-aware from constraint-unaware attacks at fixed capability. This is a useful refinement of the PR-AUC sensitivity story for the paper §5 metric-analysis subsection.

### The "OHE-validity is universal" claim is unaffected

`cross_dataset_feasibility_findings.md` (post-Cell-14 update) established that OHE-validity is the binding constraint on every constrained dataset. M+OHE strengthens this rather than complicating it: even when M closes other gaps, OHE-validity remains the load-bearing structural constraint that the attack must satisfy explicitly. The single most effective constraint-aware attack modification on any constrained dataset is forcing OHE-validity.

---

## Caveats

1. ~~**D-non-negativity variance dominates the aggregate-feasibility std.**~~ — **closed by M+OHE 2026-04-29.** The seed-instability of 0.413 in OHE-only collapsed to 0.0001 under M+OHE. The mean (1.000) is the right number to quote; per-seed values are 1.000 / 0.9998 / 1.000 (the 0.9998 on seed 123 is one test row whose clean D-field already failed the check, which the float64 immutable restore preserves correctly).

2. ~~**Filtered-success-rate std is huge (0.0/0.18/0.86 across seeds for OHE-only).**~~ — under M+OHE, FSR is 1.000 / 1.000 / 1.000 (per-seed: 9/9, 7/7, 7/7), std ≈ 0. The OHE-only volatility was downstream of the D-non-negativity variance.

3. **ε=0.1 is a single operating point.** Smaller ε might leave OHE projection unnecessary (the unconstrained attack might naturally satisfy OHE if the perturbation budget is small enough), and might also leave more attack power under M+OHE (smaller perturbations are more effective with fewer dimensions). Not tested.

4. **10 CAPGD steps.** Matches every prior run in the benchmark for comparability. M+OHE attack power might recover with more steps (more iterations to optimize within the small mutable subspace).

5. **M+OHE retrains per seed (different init from Cell 10).** Unlike the unconstrained-vs-OHE comparison (which shares the trained MLP per seed via the standard pattern in Cell 10), M+OHE runs in Cell 16 with its own model fit. The flipped-count comparison vs OHE-only is therefore between-model, not strict same-model. Filtered-success rate is a within-row ratio so the 100% saturation is unaffected; the attack-count delta (~120 → 7.7) is between-model. Consistent with the LCLD M1+g1 cell pattern.

6. **Class imbalance.** IEEE-CIS pos_weight=27.01 (~3.6% positive class) means flipped-positive counts are an order of magnitude smaller than LCLD's (~220 vs ~2700). The proportional damage at the OHE step is similar; absolute counts are not directly comparable. M+OHE drives counts to single-digit territory where Poisson-style noise becomes meaningful.

7. **Mutable feature set is "candidate" per `constraint_evaluation_guidance.md` §3.3.** TransactionAmt, ProductCD are well-supported (attacker-controlled by definition). addr1/2 and dist1/2 are inferred. A reviewer could push back, in which case the most defensible mutable set collapses to TransactionAmt + ProductCD only (~6 mutable processed dims), which would presumably crash attack count further. A sensitivity sweep across mutable-set choices is open follow-up work.

---

## What this means for the roadmap

Phase 2 cross-dataset MVP is complete. The OHE-projection-step finding from Phase 3 (`g1_projection_findings.md`) generalizes to a second dataset; the M+structure-step finding does not generalize symmetrically — it generalizes asymmetrically, which is itself a paper-worthy result.

**Immediate next steps (decision points):**

1. **Mutable-set sensitivity sweep on IEEE-CIS** — bracket the 7.7 feas-flip number by running a tighter M (just TransactionAmt + ProductCD; expected ~3 feas-flip) and a wider M (add P/R_emaildomain, M1–M9, more addr/dist; expected somewhere between 7.7 and 120). This produces a dose-response curve along the capability axis and lets the paper claim "we mapped the trade-off" instead of "we hit one point". Estimated effort: 2–3 cells, ~20 min compute.

2. **Sparkov OHE-projection (lower priority).** Same exercise as IEEE-CIS step (1) but on the third constrained dataset — 3-OHE projection on state/category/gender. Per `cross_dataset_feasibility_findings.md`, Sparkov's binding constraints are also OHE-validity checks, so the OHE-projection step should reproduce. Sparkov M is harder to define (geo-consistency makes lat/long mutable but coupled). Two datasets already establish the OHE-projection pattern; a third is paper-table polish.

3. **CCFD robust PR-AUC variance** (carryover blocker from `constraint_evaluation_guidance.md` §5). More seeds or longer training; otherwise note in paper.

**Paper updates required:**
- `constraint_evaluation_guidance.md` §1 result #1: extend to "demonstrated on LCLD and IEEE-CIS at the OHE/g1-projection step; M+structure-step behavior differs (LCLD M1+g1 saturates while preserving attack count; IEEE-CIS M+OHE saturates while crashing attack count)."
- `constraint_evaluation_guidance.md` §6 Caveat 2: extend the "single-dataset Tier C" framing — the OHE-projection finding now generalizes; the M+structure finding generalizes asymmetrically.
- `constraint_evaluation_guidance.md` §5 Phase 2: mark M+OHE follow-up complete; add a new line about the capability-vs-feasibility trade-off finding and the open mutable-set sensitivity question.
- Add a new entry to §8 Decision Log (2026-04-29): "M+OHE on IEEE-CIS reveals capability-vs-feasibility trade-off; LCLD-IEEE asymmetry framed as research finding rather than experimental gap."
