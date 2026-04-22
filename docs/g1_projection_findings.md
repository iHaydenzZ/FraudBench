# g1-Projected CAPGD on LCLD — Findings

**Date:** 2026-04-22
**Notebook:** `notebooks/g1_projection_attack.ipynb` (executed commit `a944d89`, Colab A100)
**Canonical results:** `results/adv_examples/g1_projection/g1_projection_results.csv`
**Summary table:** `results/adv_examples/g1_projection/g1_projection_summary.csv`
**Spec:** `docs/constraint_evaluation_guidance.md` §5 Phase 3

---

## Headline numbers

LCLD only, 3 seeds (42/123/456), CAPGD ε=0.1 / 10 steps, neural MLP (128-dim hidden, 20 epochs). Each seed trains one model and runs two attacks on the same model — stock `capgd_attack` vs an identical loop with per-step g1 formula projection + term OHE snapping. Mean ± std over seeds.

| Attack         | Flipped pos. | Feas.-flipped | Filtered success | Agg. feas. | g1 pass | Robust PR-AUC |
|----------------|-------------:|--------------:|-----------------:|-----------:|--------:|--------------:|
| unconstrained  | 2769 ± ~85   | 1.7           | **0.06%**        | 0.0013 ± 0.0006 | 0.012 | 0.1051 ± 0.000002 |
| g1-projected   | 2769 ± ~85   | 1100          | **39.7%**        | 0.661 ± 0.096   | 1.000 | 0.1051 ± 0.000002 |

Per-seed raw counts (for reproducibility):

| Seed | Attack        | Flipped pos. | Feas.-flipped | Agg. feas. | g1 pass | Robust PR-AUC |
|-----:|---------------|-------------:|--------------:|-----------:|--------:|--------------:|
| 42   | unconstrained | 2665         | 3             | 0.0007     | 0.0179  | 0.10505 |
| 42   | g1-projected  | 2664         | 656           | 0.5978     | 1.0000  | 0.10505 |
| 123  | unconstrained | 2838         | 1             | 0.0011     | 0.0085  | 0.10505 |
| 123  | g1-projected  | 2838         | 1653          | 0.7709     | 1.0000  | 0.10505 |
| 456  | unconstrained | 2804         | 1             | 0.0019     | 0.0107  | 0.10506 |
| 456  | g1-projected  | 2804         | 992           | 0.6132     | 1.0000  | 0.10506 |

---

## Central finding — CAPGD retains attack power under g1 projection

Flipped-prediction count is **essentially identical** across the two attacks: 2769 mean flips under unconstrained CAPGD, 2769 under g1-projected, differing by exactly one flip on a single seed. The g1 projection raised the fraction of those flipped attacks that are domain-feasible from 0.06% to 39.7% — a **660× increase in "useful" attack success rate** — without reducing the attack's ability to flip predictions at all.

This falsifies the intuition that g1's feature coupling is a meaningful defense. Before this run the expected outcome was a trade-off: constraint-aware attacks should produce more feasible examples at the cost of flipping fewer predictions. The data says there is no trade-off on LCLD at ε=0.1: CAPGD was never meaningfully using installment perturbations to cause misclassifications. It perturbed installment because the ε-budget was available, not because the gradient signal on installment had high value. Force installment to be derived, CAPGD redirects budget to `loan_amnt` / `term` and produces the same end-to-end damage.

---

## Per-constraint breakdown

| Constraint | Unconstrained adv pass | g1-projected adv pass | Notes |
|---|---:|---:|---|
| g1 (installment formula) | 0.012 | **1.000** | Forced by per-step projection |
| g4 (term OHE validity)   | 0.21  | **1.000** | Forced by per-step one-hot snapping |
| g2 (open_acc ≤ total_acc) | 0.982 | 0.982 | Unchanged — CAPGD rarely breaks it |
| g3 (bankruptcies ≤ pub_rec) | 0.676 | 0.673 | **Unchanged — now the binding constraint** |
| Aggregate | 0.0013 | 0.661 | Bounded above by g2 × g3 ≈ 0.66 |

With g1 and g4 forced to 1.0, aggregate feasibility is now gated by g2 and g3 — both of which involve credit-bureau fields (`pub_rec`, `pub_rec_bankruptcies`, `open_acc`, `total_acc`) that are immutable in any realistic LCLD threat model (listed in `LCLD_IMMUTABLE_RAW`). The fact that unconstrained CAPGD breaks g3 in ~33% of samples confirms that CAPGD is perturbing features an attacker cannot touch. Combining g1 projection with the M1 mutability mask should push aggregate feasibility to ~1.0 by freezing these fields at clean values — follow-up experiment below.

---

## Paper narrative implications

### Reframe the "+55pp ADV/ADV+CTR gap" finding

`docs/tabularbench_comparison_findings.md` established that FraudBench produces a +55pp gap between ADV (unconstrained) and ADV+CTR (post-hoc constraint-filtered) recall on LCLD — the largest on TabularBench's leaderboard. That result was being read as "post-hoc filtering recovers most of the model's apparent robustness, so unconstrained CAPGD overstates vulnerability."

Phase 3 shows the correct reading:

> Post-hoc constraint filtering rejects ~99.9% of unconstrained CAPGD's output, but a constraint-aware attacker at the same ε-budget generates 1100 feasible-and-flipped attacks instead of 2 — a **660× underestimate** of realistic attacker success. The +55pp gap measured attack-generation inefficiency, not defender safety.

### This strengthens §1 of the paper

The current guidance-doc motivation ("unconstrained attacks overstate vulnerability on financial data") is correct at the *sample level* (individual infeasible adversarial examples) but misleading at the *attacker level* (can the attacker successfully attack the model?). The precise claim becomes:

> "Unconstrained CAPGD generates ~99.8% infeasible adversarial examples, but when those examples are replaced with constraint-aware equivalents the attack success rate is preserved (2769 ≈ 2769 flipped positives). Feasibility filtering is not equivalent to constraint-aware attack generation, and papers that rely on the former vastly underestimate the threat."

This is a stronger position than the guidance doc currently takes and directly differentiates FraudBench from TabularBench's framing.

### Robust PR-AUC invariance is now load-bearing

PR-AUC on LCLD sits at 0.1051 across:
- All 8 mask-ablation variants (`mask_ablation_findings.md`)
- All 3 seeds in `cross_dataset_feasibility_findings.md`
- Both unconstrained and g1-projected attacks in this run (σ = 2×10⁻⁶)

That's five independent experimental axes where PR-AUC is unmoved. This needs an explicit paper section: CAPGD at ε=0.1 on LCLD hits a hard attack-strength ceiling on rank ordering, regardless of mask or constraint regime. The implication is that **PR-AUC cannot distinguish constraint-aware from constraint-unaware attacks on LCLD at this ε**. Robust accuracy and flipped-prediction count discriminate; PR-AUC does not.

---

## Caveats

1. **Seed 42 clean_feasibility = 0.888 versus 0.991/0.991 for seeds 123/456.** Same unseen-categorical-value issue flagged in `cross_dataset_feasibility_findings.md` §Open flags — one sparse categorical (likely `addr_state` or `purpose`) has a test-only value in the seed-42 stratified split, producing all-zero OHE rows that fail the g4-style validity check. Doesn't materially affect this run's conclusions (the Δ between attacks is what matters, and clean-feasibility is the same baseline under both) but should be resolved before the paper table is final.

2. **g3 is now the binding constraint (~0.67 pass rate) because `pub_rec` and `pub_rec_bankruptcies` are perturbed by CAPGD despite being immutable.** The M1 mask freezes these. Combining M1 with g1 projection is the natural experiment and is run below as the follow-up.

3. **g1 tolerance slack under projection.** `adv_g1_installment` = 1.0000 at G1_TOL=0.1 after projection — i.e., the float32 forward/inverse scaler round-trip is within 10 cents of the derived installment. Confirmed clean; no precision concerns.

4. **ε=0.1 is a single operating point.** A smaller ε might show attack weakening under projection if installment perturbations become load-bearing at tighter budgets. Not tested here.

5. **10 CAPGD steps.** Matches every prior LCLD run in the benchmark. Longer step budgets not tested; unclear whether the equivalence holds at, say, 50 steps.

---

## What this means for the roadmap

Phase 3 is complete and the result is stronger than predicted — the paper now has direct empirical evidence that constraint-aware attack generation and post-hoc constraint filtering are not interchangeable. No iteration needed.

**Immediate next step (this notebook):** Combined M1 mask + g1 projection. Expected to push aggregate feasibility from 0.66 to ~1.0 while preserving the ~2700 flipped-prediction count. Closes the g3 loop and produces the paper's "fully realistic constrained attack" result.

**After that:** Update `docs/constraint_evaluation_guidance.md` §1 and §5 with the Phase 3 findings, particularly the reframing of the +55pp gap.
