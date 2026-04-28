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

### Why IEEE-CIS filters hardest — and what actually does the work

Per-constraint decomposition (Cell 14, 2026-04-22 re-run) shows the per-constraint pass rates on IEEE-CIS are:

| Constraint        | Clean | Adv   | Drop  |
|-------------------|------:|------:|------:|
| `i_product_ohe`   | 1.000 | 0.018 | 0.982 |
| `i_card4_ohe`     | 1.000 | 0.048 | 0.952 |
| `i_card6_ohe`     | 1.000 | 0.206 | 0.794 |
| `i_d_nonneg`      | 1.000 | 0.486 | 0.514 |
| `i_amt_positive`  | 1.000 | 1.000 | 0.000 |
| `i_c_nonneg`      | 1.000 | **1.000** | 0.000 |

The dominance is **OHE validity, not non-negativity.** C1–C14 non-negativity is fully preserved (1.000) — StandardScaler centering means CAPGD's ε=0.1 perturbations in scaled space don't push raw-space C-counts negative. D1–D15 non-negativity drops modestly (0.486). The three OHE checks (ProductCD, card4, card6) are what produce the 0.014% aggregate. A-priori we expected "many cheap checks ANDed together" — that's still right, but the cheap checks that matter are OHE-validity, not non-negativity.

### Sparkov: same story

| Constraint        | Clean | Adv   | Drop   |
|-------------------|------:|------:|-------:|
| `s_state_ohe`     | 1.000 | 0.0002 | 0.9998 |
| `s_category_ohe`  | 1.000 | 0.017 | 0.983  |
| `s_gender_ohe`    | 1.000 | 0.265 | 0.735  |
| `s_merch_bbox`    | 0.993 | 0.992 | 0.001  |
| `s_city_pop_pos`  | 1.000 | 1.000 | 0.000  |
| `s_amt_positive`  | 1.000 | 1.000 | 0.000  |

The merchant bounding-box (the constraint we expected to dominate per `constraint_evaluation_guidance.md` §3.4) drops only 0.993 → 0.992 — CAPGD's ε in scaled space barely shifts continental-scale lat/long. The actual filter is again three OHE checks (state with 50 categories is the binding one at 0.0002 adv pass).

### LCLD per-constraint (for reference)

| Constraint        | Clean | Adv   | Drop  |
|-------------------|------:|------:|------:|
| `g1_installment`  | 0.998 | 0.012 | 0.986 |
| `g4_term_ohe`     | 1.000 | 0.185 | 0.815 |
| `g3_bankruptcy`   | 0.962 | 0.692 | 0.270 |
| `g2_open_total`   | 0.997 | 0.983 | 0.013 |

g1 (formula) leads but `g4_term_ohe` is the second-most-binding — same OHE pattern as IEEE-CIS and Sparkov.

### Refined thesis for the paper

The binary **presence vs absence** of any domain structure remains the real dichotomy at the dataset level. Within the constrained group, the per-constraint decomposition reveals a stronger claim: **OHE-validity is the universal binding constraint at ε=0.1.** It's the cheapest constraint to detect (single argmax check) and the hardest for unconstrained CAPGD to satisfy — because preprocessing distributes attack budget across all OHE columns equally, never producing a clean one-hot. Non-negativity, range, and bounding-box constraints are largely preserved because raw-space distributional centering (StandardScaler with mean ≫ 0 for counts) protects them.

Recommendation for §4: keep the binary "Constrained — any structure" vs "Unconstrained — statistical only" tiering, but in the per-dataset narrative, replace "C/D non-negativity" claims with OHE-validity claims; replace Sparkov's geo-bbox claim with OHE-validity. The single-most-paper-worthy fact in this audit is that **CAPGD breaks OHE validity on every constrained dataset** — an attack-mode finding that generalizes the LCLD g4 result.

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

1. ~~**LCLD clean feasibility is unexpectedly low: 0.9564 ± 0.0594**~~ — **MISDIAGNOSED, RESOLVED 2026-04-28**. The "sparse categorical / all-zero OHE" hypothesis was falsified by Phase A diagnostics (`scratch/diagnose_seed42.py`): no all-zero OHE rows exist on any seed. Actual root cause: float64 round-trip drift on the integer-valued `pub_rec` / `pub_rec_bankruptcies` columns during `(x-mean)/scale * scale + mean` inverse-transform. ~83% of LCLD test rows have raw `(pub_rec, bank) == (0, 0)`; the two columns drift by ~1 ULP in different directions, making strict `bank ≤ pub_rec` false on a per-seed-dependent fraction of the boundary cohort (seed 42: 2958 rows flip; seed 123: 0 flips by coincidence; seed 456: 124 flips). Fix: `EVAL_TOL = 1e-6` in `check_g2`/`check_g3` (commit `326483d` in `notebooks/g1_projection_attack.ipynb`). To be propagated to this notebook in a follow-up; expected post-fix LCLD clean feasibility ≈ 0.998 across all seeds. See `g1_projection_findings.md` §"Methodology fix" for the full mechanism.

2. ~~**Per-constraint decomposition not yet reported.**~~ **CLOSED 2026-04-22** by `notebooks/cross_dataset_feasibility.ipynb` Cell 14 (added in commit `c247aa4`). Output saved to `cross_dataset_per_constraint.csv`. Per-dataset binding constraints are: lcld→`g1_installment` (0.012), ieee_cis→`i_product_ohe` (0.018), sparkov→`s_state_ohe` (0.0002), ccfd→none (1.000). The decomposition revealed two corrections to the earlier narrative (OHE dominates IEEE-CIS, not C/D non-negativity; OHE dominates Sparkov, not geo-bbox) — incorporated into the "Why IEEE-CIS filters hardest" and "Sparkov: same story" sections above.

3. **CCFD robust PR-AUC std = 0.2262 is unstable.** Clean PR-AUC is stable (0.6330) but robust PR-AUC swings seed-to-seed from near-clean to collapsed. Likely cause: training-init variance on 0.17% positive class with only 20 epochs. Needs either more seeds or more epochs specifically for CCFD. At minimum, flag in the paper.

---

## What this means for the roadmap

Phase 1 is complete. The refined thesis it produced (*presence vs absence*, not *richness gradient*) is a stronger paper contribution than the one we started with — but it also means Phase 2's proposed work ("define mutable/immutable partitions for CCFD/IEEE-CIS/Sparkov to demonstrate Tier A") is partially obsoleted as motivation. The value of Phase 2 is now ablation/robustness evidence for the refined thesis, not structural support for a tiered framework that the data didn't bear out.

Phase 3 (Tier C — constraint-aware attack on LCLD via g1 projection) remains the critical-path experiment. Moving there next.
