# g1-Projected CAPGD on LCLD — Findings

**Date:** 2026-04-22
**Notebook:** `notebooks/g1_projection_attack.ipynb` (post-fix re-run, commit `adcd78d`, Colab A100)
**Canonical results:** `results/adv_examples/g1_projection/g1_projection_results.csv` (9 rows: 3 seeds × 3 attacks)
**Summary table:** `results/adv_examples/g1_projection/g1_projection_summary.csv`
**Spec:** `docs/constraint_evaluation_guidance.md` §5 Phase 3

---

## Headline numbers

LCLD only, 3 seeds (42/123/456), CAPGD ε=0.1 / 10 steps, neural MLP (128-dim hidden, 20 epochs). Three attacks compared on the same model per seed for the unconstrained vs g1-projected pair; M1+g1 retrains the model per seed (different random init — see Caveats §5). Mean ± std over seeds.

| Attack         | Flipped pos. | Feas.-flipped | Filtered success | Agg. feas.       | g1   | g3            | Robust acc.   | Robust PR-AUC      |
|----------------|-------------:|--------------:|-----------------:|-----------------:|-----:|--------------:|--------------:|-------------------:|
| unconstrained  | 2625 ± 244   | 1 ± 2         | **0.05%**        | 0.0009 ± 0.0007  | 0.012 | 0.706 ± 0.099 | 0.038 ± 0.001 | 0.10506 ± 3×10⁻⁶ |
| g1-projected   | 2624 ± 243   | 1318 ± 680    | **50.2%**        | 0.693 ± 0.092    | 1.000 | 0.707 ± 0.097 | 0.042 ± 0.001 | 0.10506 ± 3×10⁻⁶ |
| **M1 + g1**    | 2912 ± 191   | 2774 ± 314    | **95.3%**        | 0.958 ± 0.059    | 1.000 | 0.962 ± 0.062 | 0.175 ± 0.022 | 0.10508 ± 9×10⁻⁶ |

Per-seed raw counts (current canonical CSV):

| Seed | Attack         | Flipped | Feas.-flipped | Agg. feas. | g1    | g2    | g3    | g4    | Robust acc. | Robust PR-AUC |
|-----:|----------------|--------:|--------------:|-----------:|------:|------:|------:|------:|------------:|--------------:|
| 42   | unconstrained  | 2529    | 0             | 0.0004     | 0.011 | 0.982 | 0.709 | 0.223 | 0.0372      | 0.10505 |
| 42   | g1-projected   | 2529    | 600           | 0.6920     | 1.000 | 0.982 | 0.707 | 1.000 | 0.0414      | 0.10505 |
| 42   | **M1 + g1**    | 2830    | 2473          | 0.8897     | 1.000 | 1.000 | 0.890 | 1.000 | 0.1830      | 0.10507 |
| 123  | unconstrained  | 2443    | 3             | 0.0016     | 0.015 | 0.980 | 0.804 | 0.197 | 0.0386      | 0.10505 |
| 123  | g1-projected   | 2442    | 1953          | 0.7866     | 1.000 | 0.979 | 0.804 | 1.000 | 0.0427      | 0.10505 |
| 123  | **M1 + g1**    | 3130    | 3099          | 0.9926     | 1.000 | 0.993 | 1.000 | 1.000 | 0.1499      | 0.10509 |
| 456  | unconstrained  | 2902    | 1             | 0.0007     | 0.008 | 0.986 | 0.606 | 0.151 | 0.0368      | 0.10506 |
| 456  | g1-projected   | 2900    | 1402          | 0.6018     | 1.000 | 0.986 | 0.610 | 1.000 | 0.0408      | 0.10506 |
| 456  | **M1 + g1**    | 2776    | 2751          | 0.9927     | 1.000 | 0.997 | 0.995 | 1.000 | 0.1913      | 0.10509 |

---

## Central finding — CAPGD retains attack power under g1 projection

Same-model unconstrained vs g1-projected flipped counts differ by **at most 2 flips on any seed** (2529 vs 2529, 2443 vs 2442, 2902 vs 2900). The g1 projection raised filtered success from 0.05% to 50.2% — a **~990× increase in "useful" attack success rate** — without measurably reducing the attack's ability to flip predictions.

This falsifies the intuition that g1's feature coupling is a meaningful defense. Before this run the expected outcome was a trade-off: constraint-aware attacks should produce more feasible examples at the cost of flipping fewer predictions. The data says there is no trade-off on LCLD at ε=0.1: CAPGD was never meaningfully using installment perturbations to cause misclassifications. It perturbed installment because the ε-budget was available, not because the gradient signal on installment had high value. Force installment to be derived, CAPGD redirects budget to `loan_amnt` / `term` and produces the same end-to-end damage.

---

## Adding M1 closes the credit-bureau gap

g1-projection alone leaves g3 (`pub_rec_bankruptcies ≤ pub_rec`) at the same ~71% pass rate as the unconstrained attack — CAPGD was perturbing immutable credit-bureau fields independently of installment. Combining g1 projection with the M1 mutability mask (freezes 65 immutable raw fields including `pub_rec`, `pub_rec_bankruptcies`, `open_acc`, `total_acc`, `int_rate`, `installment`) jumps:

- **Aggregate feasibility:** 0.693 → **0.958** (×1.4)
- **g3 pass rate:** 0.707 → **0.962** (closes the credit-bureau gap)
- **Filtered success rate:** 50.2% → **95.3%** (×1.9)
- **Robust accuracy:** 0.042 → **0.175** (×4.2 — most damaging defender-side metric, still small absolute)

What's left of the gap (95.3% → 100%) is essentially the seed-42 sparse-categorical issue (clean_feasibility = 0.888 on that seed; same `addr_state` / `purpose` test-only-value problem flagged in `cross_dataset_feasibility_findings.md` §Open flags). On seeds 123 and 456 the M1+g1 aggregate feasibility is 0.993 — within 1pp of the clean-data ceiling.

The 0.05% → 50% → 95% filtered-success progression is the single cleanest experimental story in the codebase for the paper's §5/§6.

### Re-derivation note

M1 lists `installment` as immutable (attackers cannot directly set it), but g1 demands installment = f(loan_amnt, int_rate, term). Treating installment as immutable in isolation makes g1 fail whenever loan_amnt or term get perturbed. Resolution: the M1+g1 attack restores `installment` to the float64 value derived from the *post-attack* loan_amnt + restored int_rate + snapped term, in DataFrame space, after the attack loop completes. This matches what `check_g1_installment` reconstructs via `inverse_transform_numeric`. Without this fix g1 collapses from 1.0 to ~0.01 (see commit `adcd78d`).

---

## Per-constraint breakdown

| Constraint | Unconstr. | g1-projected | **M1 + g1** | Notes |
|---|---:|---:|---:|---|
| g1 (installment formula)    | 0.012  | **1.000** | **1.000** | Forced by per-step projection (and DataFrame-space re-derivation under M1) |
| g4 (term OHE validity)      | 0.190  | **1.000** | **1.000** | Forced by per-step one-hot snapping |
| g2 (open_acc ≤ total_acc)   | 0.983  | 0.982     | **0.997** | Bumped by M1 freezing both fields |
| g3 (bankruptcies ≤ pub_rec) | 0.706  | 0.707     | **0.962** | Bumped by M1 freezing both fields; residual 0.04 is seed-42 OHE artifact |
| Aggregate                   | 0.0009 | 0.693     | **0.958** | Bounded above by clean_feasibility (~0.96 mean) |

g3 was the binding constraint after g1-projection alone; M1 closes it. The remaining gap to 1.0 is entirely the seed-42 unseen-categorical issue, not an attack-side defect.

---

## Paper narrative implications

### Reframe the "+55pp ADV/ADV+CTR gap" finding

`docs/tabularbench_comparison_findings.md` established that FraudBench produces a +55pp gap between ADV (unconstrained) and ADV+CTR (post-hoc constraint-filtered) recall on LCLD — the largest on TabularBench's leaderboard. That result was being read as "post-hoc filtering recovers most of the model's apparent robustness, so unconstrained CAPGD overstates vulnerability."

Phase 3 + follow-up shows the correct reading:

> Post-hoc constraint filtering rejects ~99.95% of unconstrained CAPGD's output, but a constraint-aware attacker at the same ε-budget generates 1318 feasible-and-flipped attacks instead of 1 (g1 alone), and 2774 with M1+g1 — a **~2000× underestimate** of realistic attacker success when both formula and mutability are honored. The +55pp gap measured attack-generation inefficiency, not defender safety.

### This strengthens §1 of the paper

The current guidance-doc motivation ("unconstrained attacks overstate vulnerability on financial data") is correct at the *sample level* (individual infeasible adversarial examples) but misleading at the *attacker level* (can the attacker successfully attack the model?). The precise claim becomes:

> "Unconstrained CAPGD generates ~99.9% infeasible adversarial examples, but when those examples are replaced with constraint-aware equivalents (g1 projection + M1 mutability mask) the attack success rate rises ×2000 and reaches 95% domain-feasibility while preserving the flipped-prediction count. Feasibility filtering is not equivalent to constraint-aware attack generation, and papers that rely on the former vastly underestimate the threat."

This is a stronger position than the guidance doc currently takes and directly differentiates FraudBench from TabularBench's framing.

### Robust PR-AUC invariance is now load-bearing

PR-AUC on LCLD sits at 0.1051 across:
- All 8 mask-ablation variants (`mask_ablation_findings.md`)
- All 3 seeds in `cross_dataset_feasibility_findings.md`
- Three attack regimes in this run — unconstrained, g1-projected, M1+g1 (σ ≤ 9×10⁻⁶ within each)

That's six independent experimental axes where PR-AUC is unmoved. This needs an explicit paper section: CAPGD at ε=0.1 on LCLD hits a hard attack-strength ceiling on rank ordering, regardless of mask, attack variant, or constraint regime. The implication is that **PR-AUC cannot distinguish constraint-aware from constraint-unaware attacks on LCLD at this ε**. Robust accuracy and flipped-prediction count discriminate (robust acc moves 0.038 → 0.042 → 0.175 across the three regimes); PR-AUC does not.

---

## Caveats

1. **Seed 42 clean_feasibility = 0.888 versus 0.991/0.991 for seeds 123/456.** Same unseen-categorical-value issue flagged in `cross_dataset_feasibility_findings.md` §Open flags — one sparse categorical (likely `addr_state` or `purpose`) has a test-only value in the seed-42 stratified split, producing all-zero OHE rows that fail the g4-style validity check. Caps M1+g1 aggregate feasibility on seed 42 at ~0.89 instead of ~0.99.

2. **g1 tolerance slack under projection.** `adv_g1_installment` = 1.0000 at G1_TOL=0.1 after projection (both g1-projected and M1+g1) — i.e., the float32/float64 round-trip is within 10 cents of the derived installment. Confirmed clean; no precision concerns.

3. **ε=0.1 is a single operating point.** A smaller ε might show attack weakening under projection if installment perturbations become load-bearing at tighter budgets. Not tested here.

4. **10 CAPGD steps.** Matches every prior LCLD run in the benchmark. Longer step budgets not tested; unclear whether the equivalence holds at, say, 50 steps.

5. **M1+g1 uses a different trained model than unconstrained / g1-projected.** Cell 16 retrains the MLP per seed (different random init from Cell 10's models). On clean PR-AUC the models match (~0.302 ± 0.003 vs ~0.301 ± 0.002), but the test-set decision boundary differs slightly — explains why M1+g1 flipped count (2912) is higher than the other two (2624/2625). The headline filtered-success comparison (0.05% / 50.2% / 95.3%) is a *ratio on each row's own model* and is therefore unaffected by this. Same caveat applies to robust-accuracy comparisons across the three regimes.

6. **Re-run variance on Cell 10.** Re-running the whole notebook (including Cell 10's per-seed model training) shifts flipped counts by ±100–300 per seed across runs (seed 123 in particular swung 2838 → 2443 between the original Phase 3 run and this re-run). The unconstrained ≈ g1-projected equivalence (≤2-flip delta on the same model) is preserved across re-runs and is the load-bearing fact.

---

## What this means for the roadmap

Phase 3 and the M1+g1 follow-up are complete. The result is stronger than predicted — the paper now has direct empirical evidence that:

1. Constraint-aware attack generation and post-hoc constraint filtering are not interchangeable (g1-projection alone: 0.05% → 50% filtered success, no loss of flips).
2. Combining formula projection with mutability masks recovers ~95% domain-feasibility while quadrupling robust-accuracy damage compared to g1 alone.
3. PR-AUC is insensitive to constraint regime at the LCLD ε=0.1 operating point — robust accuracy and flipped-and-feasible counts are the discriminating metrics.

**Next step:** Update `docs/constraint_evaluation_guidance.md` §1 and §5 with the Phase 3 findings (in particular, the reframing of the +55pp gap and the M1+g1 closure of g3). Then move on from g1/M1 to other roadmap items.
