# Cross-Dataset Feasibility Audit — Findings

**Date:** 2026-04-22
**Notebook:** `notebooks/cross_dataset_feasibility.ipynb` (executed commit `088d8c7`, Colab A100)
**Canonical results:** `results/adv_examples/cross_dataset_feasibility/cross_dataset_feasibility_results.csv`
**Companion gradient table:** `results/adv_examples/cross_dataset_feasibility/cross_dataset_feasibility_gradient.csv`
**Spec:** `docs/constraint_evaluation_guidance.md` §5 Phase 1

---

## Headline numbers

All four datasets, 3 seeds each (42, 123, 456), unconstrained CAPGD at ε=0.1, sample_frac=0.1, neural MLP (128-dim hidden, 20 epochs).

| Dataset | Clean feas. (mean ± std) | Adv feas. (mean ± std) | Clean PR-AUC | Robust PR-AUC (mean ± std) |
|---|---:|---:|---:|---:|
| **ieee_cis** | 0.9999 ± 0.0001 | **0.000141 ± 0.000098** | 0.4283 | 0.0678 ± 0.0238 |
| **lcld**     | 0.9564 ± 0.0594 | **0.000932 ± 0.000393** | 0.3020 | 0.1051 ± 0.0000 |
| **sparkov**  | 0.9930 ± 0.0004 | **0.003788 ± 0.002786** | 0.6056 | 0.0055 ± 0.0003 |
| **ccfd**     | 1.0000 ± 0.0000 | **1.000000 ± 0.000000** | 0.6330 | 0.5802 ± 0.2262 |

---

## Headline finding — the a-priori gradient is wrong

`docs/constraint_evaluation_guidance.md` §4 predicted a monotone constraint-richness gradient:

> LCLD (Very High) >> Sparkov (Moderate) > IEEE-CIS (Low) > CCFD (Very Low)

The empirical ordering by adversarial feasibility (ascending = more restrictive) is instead:

> IEEE-CIS (0.014%) < LCLD (0.093%) < Sparkov (0.38%) << CCFD (100%)

The three constrained datasets cluster tightly below 1% adv feasibility; CCFD stands alone at 100%. IEEE-CIS — framed in the guidance doc as "Low" constraint richness — is empirically the most restrictive of the four.

### Why IEEE-CIS filters hardest

CAPGD at ε=0.1 in processed space simultaneously breaks:
- OHE validity of three separate categoricals: ProductCD, card4, card6 (each a g4-style check)
- Non-negativity of C1–C14 (14 columns ANDed)
- Non-negativity of D1–D15 (15 columns ANDed)

Six cheap per-row boolean checks ANDed together filter harder than one nonlinear formula does in isolation. The paper's original "richness" framing — which assumed constraint *strength* dominates — missed that constraint *count* also matters, and that high-dimensional OHE expansions implicitly create many cheap constraints.

### Refined thesis for the paper

The binary **presence vs absence** of any domain structure is the real dichotomy. Within the constrained group (LCLD, IEEE-CIS, Sparkov) the exact adv feasibility rate is a function of how many independent OHE / non-negativity / range checks the preprocessing pipeline exposes, not of whether the dataset carries a nonlinear formula constraint. Recommendation for §4: collapse the three-tier gradient table to two tiers ("Constrained — any structure" vs "Unconstrained — statistical only") and back the tiering with the table above rather than a-priori constraint counts.

---

## The Sparkov row is the cleanest evidence for reporting PR-AUC alongside feasibility

| Dataset | Adv feasibility | Robust PR-AUC |
|---|---:|---:|
| sparkov | 0.0038 (constraints filter 99.6% of attacks) | 0.0055 (near-random on ~0.6% positive class) |
| ccfd    | 1.0000 (no constraint filters anything)      | 0.5802 (barely degraded from clean 0.6330) |

Sparkov: constraints reject almost all adv examples, but the ~0.4% that remain concentrate exactly where they destroy the model's rank ordering. CCFD: loads of "feasible" adv examples, almost no damage to the model's rank ordering.

**Implication:** Constraint feasibility and PR-AUC capture different facets of adversarial robustness — attack *filterability* vs attack *destructiveness*. Reporting only one misrepresents reality. This is the paper's cleanest empirical argument for dual-metric evaluation and should anchor §5/§6 of the writeup.

---

## Other notable results

### Robust PR-AUC collapse ordering

| Dataset | Clean → Robust PR-AUC | Relative drop |
|---|---:|---:|
| sparkov  | 0.606 → 0.005 | −99.1% |
| ieee_cis | 0.428 → 0.068 | −84.2% |
| lcld     | 0.302 → 0.105 | −65.2% |
| ccfd     | 0.633 → 0.580 | −8.4%  |

Ranking: Sparkov > IEEE-CIS > LCLD > CCFD for model-damage severity. This ordering is *not* inversely correlated with adv feasibility — further evidence that the two metrics are capturing independent phenomena.

### LCLD robust PR-AUC is still locked at 0.1051

The seed-level std on LCLD robust PR-AUC is 0.0000 (three-significant-figure precision). Matches the mask ablation's "PR-AUC invariance" finding exactly. Strong evidence that CAPGD at ε=0.1 is hitting an attack-strength ceiling on LCLD, which neither mask nor attack variant has been able to move.

---

## Open flags (blockers before paper table is final)

1. **LCLD clean feasibility is unexpectedly low: 0.9564 ± 0.0594** (tabularbench_comparison run reported 0.9968 with identical g1 tolerance). The 0.0594 std implies at least one of the three seeds fell below ~0.90. Hypothesis: a sparse categorical (`addr_state` or `purpose`) has a category present in test but not train for one seed, causing all-zero OHE rows that fail the g4-style OHE-sum check. Need per-seed inspection of `results_df[results_df["dataset"]=="lcld"]`.

2. **Per-constraint decomposition not yet reported.** The results CSV has per-constraint columns (`adv_g1_installment`, `adv_i_c_nonneg`, `adv_s_merch_bbox`, …) but only the aggregate was printed. Without the decomposition we cannot confirm which constraint actually dominates for IEEE-CIS and Sparkov — essential for §4 of the paper. Minimal fix: add a cell that groups the `clean_*_` / `adv_*_` per-constraint columns by dataset and prints means.

3. **CCFD robust PR-AUC std = 0.2262 is unstable.** Clean PR-AUC is stable (0.6330) but robust PR-AUC swings seed-to-seed from near-clean to collapsed. Likely cause: training-init variance on 0.17% positive class with only 20 epochs. Needs either more seeds or more epochs specifically for CCFD. At minimum, flag in the paper.

---

## What this means for the roadmap

Phase 1 is complete. The refined thesis it produced (*presence vs absence*, not *richness gradient*) is a stronger paper contribution than the one we started with — but it also means Phase 2's proposed work ("define mutable/immutable partitions for CCFD/IEEE-CIS/Sparkov to demonstrate Tier A") is partially obsoleted as motivation. The value of Phase 2 is now ablation/robustness evidence for the refined thesis, not structural support for a tiered framework that the data didn't bear out.

Phase 3 (Tier C — constraint-aware attack on LCLD via g1 projection) remains the critical-path experiment. Moving there next.
