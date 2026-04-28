# g1-Projected CAPGD on LCLD — Findings

**Date:** 2026-04-28 (post float-round-trip fix; supersedes 2026-04-22 figures)
**Notebook:** `notebooks/g1_projection_attack.ipynb` (commit `326483d`, Colab A100)
**Canonical results:** `results/adv_examples/g1_projection/g1_projection_results.csv` (9 rows: 3 seeds × 3 attacks)
**Spec:** `docs/constraint_evaluation_guidance.md` §5 Phase 3

---

## Headline numbers

LCLD only, 3 seeds (42/123/456), CAPGD ε=0.1 / 10 steps, neural MLP (128-dim hidden, 20 epochs). Three attacks compared on the same model per seed for the unconstrained vs g1-projected pair; M1+g1 retrains the model per seed (different random init — see Caveats §4). Mean ± std over seeds.

| Attack         | Flipped pos. | Feas.-flipped | Filtered success | Agg. feas.       | g1    | g3            | Robust acc.    | Robust PR-AUC      |
|----------------|-------------:|--------------:|-----------------:|-----------------:|------:|--------------:|---------------:|-------------------:|
| unconstrained  | 2769 ± 130   | 3 ± 1         | **0.11%**        | 0.0012 ± 0.0005  | 0.012 | 0.802 ± 0.009 | 0.036 ± 0.013  | 0.10506 ± 4×10⁻⁶  |
| g1-projected   | 2769 ± 129   | 2111 ± 121    | **76.5%**        | 0.791 ± 0.010    | 1.000 | 0.804 ± 0.011 | 0.039 ± 0.013  | 0.10506 ± 4×10⁻⁶  |
| **M1 + g1**    | 2888 ± 140   | 2888 ± 140    | **100.0%**       | 1.000 ± 0.000    | 1.000 | 1.000 ± 0.000 | 0.153 ± 0.014  | 0.10507 ± 7×10⁻⁶  |

Per-seed raw counts (current canonical CSV):

| Seed | Attack         | Flipped | Feas.-flipped | Agg. feas. | g1    | g2    | g3    | g4    | Robust acc. | Robust PR-AUC |
|-----:|----------------|--------:|--------------:|-----------:|------:|------:|------:|------:|------------:|--------------:|
| 42   | unconstrained  | 2824    | 3             | 0.0018     | 0.018 | 0.984 | 0.791 | 0.213 | 0.0517      | 0.10505 |
| 42   | g1-projected   | 2824    | 1970          | 0.7785     | 1.000 | 0.984 | 0.791 | 1.000 | 0.0550      | 0.10505 |
| 42   | **M1 + g1**    | 2770    | 2770          | **1.0000** | 1.000 | 1.000 | 1.000 | 1.000 | 0.1681      | 0.10507 |
| 123  | unconstrained  | 2870    | 2             | 0.0009     | 0.009 | 0.980 | 0.811 | 0.159 | 0.0210      | 0.10505 |
| 123  | g1-projected   | 2869    | 2155          | 0.7962     | 1.000 | 0.980 | 0.814 | 1.000 | 0.0231      | 0.10505 |
| 123  | **M1 + g1**    | 2809    | 2809          | **1.0000** | 1.000 | 1.000 | 1.000 | 1.000 | 0.1581      | 0.10507 |
| 456  | unconstrained  | 2614    | 4             | 0.0009     | 0.010 | 0.990 | 0.805 | 0.203 | 0.0350      | 0.10506 |
| 456  | g1-projected   | 2614    | 2208          | 0.7976     | 1.000 | 0.990 | 0.807 | 1.000 | 0.0385      | 0.10506 |
| 456  | **M1 + g1**    | 3084    | 3084          | **1.0000** | 1.000 | 1.000 | 1.000 | 1.000 | 0.1338      | 0.10508 |

---

## Central finding — CAPGD retains attack power under g1 projection

Same-model unconstrained vs g1-projected flipped counts differ by **at most 1 flip on any seed** (2824/2824, 2870/2869, 2614/2614). The g1 projection raised filtered success from 0.11% to 76.5% — a **~700× increase in "useful" attack success rate** — without measurably reducing the attack's ability to flip predictions.

This falsifies the intuition that g1's feature coupling is a meaningful defense. Before this run the expected outcome was a trade-off: constraint-aware attacks should produce more feasible examples at the cost of flipping fewer predictions. The data says there is no trade-off on LCLD at ε=0.1: CAPGD was never meaningfully using installment perturbations to cause misclassifications. It perturbed installment because the ε-budget was available, not because the gradient signal on installment had high value. Force installment to be derived, CAPGD redirects budget to `loan_amnt` / `term` and produces the same end-to-end damage.

---

## Adding M1 closes the credit-bureau gap completely

g1-projection alone leaves g3 (`pub_rec_bankruptcies ≤ pub_rec`) at the same ~80% pass rate as the unconstrained attack — CAPGD was perturbing immutable credit-bureau fields independently of installment. Combining g1 projection with the M1 mutability mask (freezes 65 immutable raw fields including `pub_rec`, `pub_rec_bankruptcies`, `open_acc`, `total_acc`, `int_rate`, `installment`) drives the constrained attack to its theoretical ceiling:

- **Aggregate feasibility:** 0.791 → **1.000** (closes the gap exactly)
- **g3 pass rate:** 0.804 → **1.000** (frozen by M1)
- **g2 pass rate:** 0.985 → **1.000** (frozen by M1)
- **Filtered success rate:** 76.5% → **100.0%** (every flipped prediction is feasible)
- **Robust accuracy:** 0.039 → **0.153** (×3.9 — most damaging defender-side metric)

**Every M1+g1 sample passes every constraint, on every seed.** The attack saturates the constrained-feasibility envelope.

The 0.11% → 76.5% → 100% filtered-success progression is the single cleanest experimental story in the codebase for the paper's §5/§6.

### Re-derivation note

M1 lists `installment` as immutable (attackers cannot directly set it), but g1 demands installment = f(loan_amnt, int_rate, term). Treating installment as immutable in isolation makes g1 fail whenever loan_amnt or term get perturbed. Resolution: the M1+g1 attack restores `installment` to the float64 value derived from the *post-attack* loan_amnt + restored int_rate + snapped term, in DataFrame space, after the attack loop completes. This matches what `check_g1_installment` reconstructs via `inverse_transform_numeric`. Without this fix g1 collapses from 1.0 to ~0.01 (see commit `adcd78d`).

---

## Per-constraint breakdown

| Constraint | Unconstr. | g1-projected | **M1 + g1** | Notes |
|---|---:|---:|---:|---|
| g1 (installment formula)    | 0.012  | **1.000** | **1.000** | Forced by per-step projection (and DataFrame-space re-derivation under M1) |
| g4 (term OHE validity)      | 0.192  | **1.000** | **1.000** | Forced by per-step one-hot snapping |
| g2 (open_acc ≤ total_acc)   | 0.985  | 0.985     | **1.000** | M1 freezes both fields → invariant under attack |
| g3 (bankruptcies ≤ pub_rec) | 0.802  | 0.804     | **1.000** | M1 freezes both fields → invariant under attack |
| Aggregate                   | 0.0012 | 0.791     | **1.000** | Saturates: every M1+g1 sample is fully feasible |

g3 was the binding constraint after g1-projection alone; M1 closes it. g1 + M1 reaches the ceiling exactly because the four constraints split cleanly into "frozen by M1" (g2, g3) and "forced by projection" (g1, g4) — there is no constraint left for the attack to break.

---

## Methodology fix: float64 round-trip drift

The headline numbers above supersede the prior Phase 3 figures (0.05% → 50.2% → 95.3%, dated 2026-04-22, commit `ba6e371`). The earlier numbers were depressed by a ~10pp bias on seed 42 (and ~0.5pp on seed 456) caused by a previously-undiagnosed evaluation pipeline bug.

**Mechanism (confirmed by Phase A.1–A.5 diagnostics, see `scratch/diagnose_*.py`):**
- LCLD's raw `pub_rec` and `pub_rec_bankruptcies` are integer counts; ~83% of test rows have both = 0
- The standard `(x - mean) / scale * scale + mean` round-trip via `StandardScaler` introduces ~1 ULP (≤ 1e-15) drift on these columns
- The two columns have different mean/scale, so they drift in *different* directions — for the (0, 0) cohort, bank ULP-positive while pub_rec ULP-negative makes `bank > pub_rec` strict
- Seed 42's specific train-set produces the maximum drift configuration (2958 of 26820 test rows flip), seed 123 happens to round-trip bit-exactly (zero drift), seed 456 sits between (124 flips)
- The bug was in the evaluation, not the attack: M1 freezes the columns correctly, but `check_g3` applied to round-tripped values reports false violations

**Fix (commit `326483d`):** Add `EVAL_TOL = 1e-6` absolute tolerance to `check_g2` and `check_g3`. Raw values are integer counts so the smallest real violation is 1.0 — 1e-6 cannot mask any true constraint break and fully absorbs ULP drift.

**This is itself a publishable methodology gotcha** for tabular constrained-attack benchmarks: any benchmark that evaluates integer-count inequality constraints on round-tripped (transform → inverse_transform) data systematically underestimates adversarial robustness on a per-seed basis tied to the train-set's specific scaler parameters. Worth a sentence in the paper's methodology section ("Evaluating constraints on integer-valued raw columns requires either bypassing the round-trip or applying a strictly-sub-unit tolerance to absolute inequalities").

---

## Paper narrative implications

### Reframe the "+55pp ADV/ADV+CTR gap" finding

`docs/tabularbench_comparison_findings.md` established that FraudBench produces a +55pp gap between ADV (unconstrained) and ADV+CTR (post-hoc constraint-filtered) recall on LCLD — the largest on TabularBench's leaderboard. That result was being read as "post-hoc filtering recovers most of the model's apparent robustness, so unconstrained CAPGD overstates vulnerability."

Phase 3 + the M1+g1 follow-up shows the correct reading:

> Post-hoc constraint filtering rejects ~99.9% of unconstrained CAPGD's output (only ~3 of ~2769 flipped positives survive per seed), but a constraint-aware attacker at the same ε-budget generates **2111 feasible-and-flipped attacks** with g1-projection alone, and **2888 (= 100% of flipped) with M1+g1** — a **~960× underestimate** of realistic attacker success when both formula and mutability are honored. The +55pp gap measured attack-generation inefficiency, not defender safety.

### This strengthens §1 of the paper

The current guidance-doc motivation ("unconstrained attacks overstate vulnerability on financial data") is correct at the *sample level* (individual infeasible adversarial examples) but misleading at the *attacker level* (can the attacker successfully attack the model?). The precise claim becomes:

> "Unconstrained CAPGD generates ~99.9% infeasible adversarial examples, but when those examples are replaced with constraint-aware equivalents (g1 projection + M1 mutability mask) the attack success rate rises ×960 and reaches 100% domain-feasibility while preserving and slightly increasing the flipped-prediction count. Feasibility filtering is not equivalent to constraint-aware attack generation, and papers that rely on the former vastly underestimate the threat."

This is a stronger position than the guidance doc currently takes and directly differentiates FraudBench from TabularBench's framing.

### Robust PR-AUC invariance is now load-bearing

PR-AUC on LCLD sits at 0.10506 ± 7×10⁻⁶ across:
- All 8 mask-ablation variants (`mask_ablation_findings.md`)
- All 3 seeds in `cross_dataset_feasibility_findings.md`
- Three attack regimes in this run — unconstrained, g1-projected, M1+g1 (σ ≤ 7×10⁻⁶ within each)

That's six independent experimental axes where PR-AUC is unmoved. This needs an explicit paper section: CAPGD at ε=0.1 on LCLD hits a hard attack-strength ceiling on rank ordering, regardless of mask, attack variant, or constraint regime. The implication is that **PR-AUC cannot distinguish constraint-aware from constraint-unaware attacks on LCLD at this ε**. Robust accuracy and flipped-prediction count discriminate (robust acc moves 0.036 → 0.039 → 0.153 across the three regimes); PR-AUC does not.

---

## Caveats

1. **g1 tolerance slack under projection.** `adv_g1_installment` = 1.0000 at G1_TOL=0.1 after projection (both g1-projected and M1+g1) — i.e., the float32/float64 round-trip is within 10 cents of the derived installment. Confirmed clean; no precision concerns.

2. **ε=0.1 is a single operating point.** A smaller ε might show attack weakening under projection if installment perturbations become load-bearing at tighter budgets. Not tested here.

3. **10 CAPGD steps.** Matches every prior LCLD run in the benchmark. Longer step budgets not tested; unclear whether the equivalence holds at, say, 50 steps.

4. **M1+g1 uses a different trained model than unconstrained / g1-projected.** Cell 16 retrains the MLP per seed (different random init from Cell 10's models). On clean PR-AUC the models match (~0.298 ± 0.007 vs ~0.304 ± 0.000), but the test-set decision boundary differs slightly — explains why M1+g1 flipped count (2888) differs from the other two (2769). The headline filtered-success comparison (0.11% / 76.5% / 100%) is a *ratio on each row's own model* and is therefore unaffected by this. Same caveat applies to robust-accuracy comparisons across the three regimes.

5. **EVAL_TOL = 1e-6 is a deliberate choice.** Raw LCLD integer counts have minimum non-zero gap 1.0; tolerance is 6 orders of magnitude smaller, mathematically incapable of masking any real attacker-induced violation. The fix is mechanism-level (correct evaluation under round-trip), not a result-massaging knob.

---

## What this means for the roadmap

Phase 3 and the M1+g1 follow-up are complete. The result is now stronger than the originally reported figures:

1. Constraint-aware attack generation and post-hoc constraint filtering are not interchangeable (g1-projection alone: 0.11% → 76.5% filtered success, no loss of flips).
2. Combining formula projection with mutability masks reaches **100% domain-feasibility** while quadrupling robust-accuracy damage compared to g1 alone — saturates the constrained envelope.
3. PR-AUC is insensitive to constraint regime at the LCLD ε=0.1 operating point — robust accuracy and flipped-and-feasible counts are the discriminating metrics.
4. The "seed-42 sparse-categorical OHE artifact" (referenced as a soft blocker in 5 findings docs) **does not exist**. The actual cause of the seed-42 anomaly was float64 round-trip drift, fixed by `EVAL_TOL = 1e-6` in commit `326483d`.

**Next step:** Apply the same `EVAL_TOL` fix to the other LCLD-touching notebooks (`mask_ablation`, `cross_dataset_feasibility`, `tabularbench_comparison`, `tabularbench_metric_analysis`) and refresh their findings docs. Remove the soft-blocker entry from `docs/ToDo.md` ICAIF §B. Then move on from g1/M1 to other roadmap items.
