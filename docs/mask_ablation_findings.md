# Mask Ablation — Findings

**Date:** 2026-04-15
**Notebook:** `notebooks/mask_ablation.ipynb` (executed commit `0799c5f`)
**Canonical results:** `results/adv_examples/mask_ablation/mask_ablation_summary.csv`
**Spec:** `docs/plans/mask_ablation_experiment_plan.md`
**Implementation plan:** `docs/plans/2026-04-15-mask-ablation-implementation.md`

---

## Headline numbers

All variants produced in the same seed loop against the same trained model per seed (no cross-run mixing). Robust metrics averaged over 3 seeds. Feasibility / g1 / g4 are seed=42 only.

| Variant | n_mut (mean) | Robust PR-AUC | Robust Acc (mean ± std) | Feasibility | g1 pass | g4 pass |
|---|---:|---:|---:|---:|---:|---:|
| M0 (no mask) | 187.7 | 0.1051 ± 0.0000 | 0.0367 ± 0.0070 | 0.0005 | 0.0140 | 0.1501 |
| M1 (binary mask) | 122.7 | 0.1051 ± 0.0000 | **0.1419 ± 0.0086** | 0.0018 | 0.0088 | 0.1673 |
| M2 (directional) | 122.7 | 0.1051 ± 0.0000 | 0.1421 ± 0.0089 | 0.0017 | 0.0088 | 0.1691 |
| M3 (+dti freeze) | 121.7 | 0.1051 ± 0.0000 | 0.1467 ± 0.0088 | 0.0018 | 0.0087 | 0.1694 |
| M4 (+term freeze) | 120.7 | 0.1051 ± 0.0000 | 0.1503 ± 0.0098 | 0.0092 | 0.0092 | **1.0000** |
| M5 (M3 ∪ M4) | 119.7 | 0.1051 ± 0.0000 | 0.1557 ± 0.0100 | 0.0078 | 0.0078 | 1.0000 |
| **M6-strict** | 73.7 | 0.1055 ± 0.0003 | **0.3147 ± 0.0103** | 0.0069 | 0.0069 | 1.0000 |
| M6-relaxed | 86.7 | 0.1053 ± 0.0002 | 0.2688 ± 0.0102 | 0.0082 | 0.0082 | 1.0000 |

---

## Sanity checks performed

- **M2 ≡ M1 confirmed.** Robust accuracy difference 0.1421 − 0.1419 = 0.0002, far below the 0.009 seed-std band. The `[M2 warning]` print confirmed the directional clip matched zero OHE-expanded columns for `emp_length`, so M2 degenerates into M1's attack. Validates spec §8.2's prediction.
- **Term freeze effect.** `term_ohe_max_abs_delta` for M4/M5/M6-strict/M6-relaxed = 0.000, for M0/M1/M2/M3 ≈ 0.079. Freeze works.
- **Dti freeze effect.** `dti_mean_abs_delta` for M3/M5/M6-strict/M6-relaxed ≈ 1.5e-7 (float noise), for M0/M1/M2/M4 ≈ 0.99. Freeze works.
- **E1 scale linearity.** Mean cost at `cost_scale=2.0` is exactly 2× `cost_scale=1.0` (M0: 0.458 → 0.916; M1: 0.470 → 0.939). Same linearity holds at 0.5× scale.
- **Feasibility aggregate math.** For every variant, `aggregate ≤ min(g1, g2, g3, g4)` held (trivially, since g2 = g3 = 1.0 under M1+ and g4 = 1.0 under M4+, leaving g1 as the AND bottleneck).
- **Model-training parity.** Clean PR-AUC / clean accuracy is identical across all 8 variant rows within a seed (e.g., seed=42: clean_pr_auc=0.2974 for every variant). Confirms the train-once-per-seed optimization is correct: all variants see the same model, same data split, same preprocessor.

---

## Caveats

- **Single-seed feasibility / perturbation stats / E1.** Computed on seed=42 only (per spec §4). Summary table's `feasibility_seed42` / `g1_pass_seed42` / `g4_pass_seed42` columns have no ± band.
- **M4/M5/M6 g4 = 1.0 is structural, not a capability improvement.** Freezing `term_*` OHE columns (M4/M5) or the entire raw `term` feature (M6 via exclusion from mutable set) guarantees the OHE validity check passes by construction. Do NOT frame this as "CAPGD learned to produce valid OHE encodings" — it did not; it was prevented from touching those columns at all. Correct framing: *freezing term recovers the portion of aggregate feasibility that was blocked solely by g4*. For LCLD the g4-contribution to aggregate was ~0.01, matching the M4 aggregate exactly.
- **E1 covers numeric features, `term`, and `emp_length`** (as binary changed/unchanged). `term` is reconstructed via `reconstruct_term_from_ohe` (gives numeric 36/60 — ordinally meaningful). `emp_length` is reconstructed via OHE-argmax but the argmax index is NOT ordinal (OHE column order is lexical: `1 year`, `10+ years`, `2 years`, ...), so `total_cost` treats it as a binary changed/unchanged indicator via `BINARY_COST_FEATURES`. Remaining categoricals (`purpose`, `home_ownership`, `addr_state`, `application_type`) are NOT reconstructed and contribute 0 — documented scope limitation.
- **E1 uses corrected binary cost for emp_length** (commit `a928eea`, Cell 13 re-run). The pre-fix version scaled `emp_length` cost by OHE-argmax index gap; the corrected version uses a binary changed/unchanged indicator. Effect on totals is small (M0 mean 0.458 vs pre-fix 0.460, M1 0.470 vs 0.471) because the OHE argmax rarely flips under the attack magnitude (term_ohe_max_abs_delta ≈ 0.08, too small to move argmax in most rows). Conclusion unchanged: M0 and M1 have near-identical attack costs.
- **E1 normalization.** Uses p1/p99 winsorized ranges to tame LCLD `annual_inc` tails. Sensitivity verified via ×2/×0.5 scale linearity check.
- **3-seed std is tight (0.007–0.010 on robust accuracy).** M5 − M1 = 0.014 is roughly one std band; M3/M4 differences from M1 are within the noise. Only M6-strict (+0.17) and M0 (−0.11) are robustly outside seed variance. Other pairs should be read as "indistinguishable at this sample size."
- **M2 is structurally a no-op on LCLD.** Only `emp_length` has a natural increase-only interpretation, and LCLD encodes it as a 12-column OHE. Directional clip is exact-match only (post-fix) and skips OHE-expanded features. To actually evaluate M2 would require an ordinal encoding of `emp_length` — deferred.
- **Undeclared-feature handling.** `build_processed_mutable_mask` defaults to mutable for any raw column not in the immutable set. The 11 raw features not listed in `LCLD_IMMUTABLE_RAW` ∪ `LCLD_MUTABLE_RAW` inherit this default — same behavior as M1 baseline, so comparisons remain internally consistent.
- **M6 immutable set** is `dataset.X.columns - MUTABLE_PROFILE`. Verified that `dataset.X.columns` excludes the target label.

---

## Answers to meeting questions (spec §9)

1. **Biggest marginal effect on robust accuracy?** M6-strict by a wide margin. Moving from M1 (122 mutable processed features) to M6-strict (74 mutable) lifts robust accuracy from 0.142 to 0.315 — a +17.3 point gap at 10× the seed-level std. The M1 → M4 → M5 progression yields only +0.014 total, roughly one std band — mask-layer fixes alone are not a strong defense on LCLD.
2. **Does M4 lift aggregate feasibility?** Yes, but only by the amount g4 was contributing. M1 aggregate = 0.0018; M4 aggregate = 0.0092 (+0.0074). The gain matches M4's g4=1.0 contribution exactly, because g1 is the remaining bottleneck (stays at 0.009 under every mask). Conclusion: freezing term cleans up a constraint artifact but does not produce substantively more feasible attacks because the installment formula (g1) still rejects them.
3. **Attacker-capability spectrum.** M1 (all 11 attacker-changeable features free) → 0.142. M6-relaxed (8 self-reported features) → 0.269. M6-strict (5 form-only features) → 0.315. Clear monotone spread. A low-capability attacker is meaningfully weaker on LCLD — a +17 point robustness gap vs M1.
4. **E1 takeaway.** Corrected M0/M1 costs are near-identical: M0 mean 0.458, median 0.507, p95 0.514; M1 mean 0.470, median 0.514, p95 0.514. M1 is marginally more expensive than M0 (+0.012 mean, +0.007 median) — counterintuitive at first since M1 moves fewer features, but consistent with the interpretation that M1's attack concentrates on high-weight features (annual_inc COST=8, dti COST=7) while M0 spreads across cheaper features (loan_amnt COST=1, term COST=1). Both variants' p95 saturates at 0.514 — attacks that succeed consume a similar "budget" regardless of which features are available. Weak differentiator for defender utility; not a strong talking point.
5. **Next experiment round.** g1 (installment formula) is the dominant unfixable barrier under any mask — it never moves above 0.015 across all variants, including M5 and M6-strict. This is strong evidence that **constraint-aware attacks** (e.g., TabularBench's CAPGDConstrained or CPGDL2) are the required next step on LCLD. Pure mask layering is saturating. Secondary: run M6-strict on CCFD/IEEE-CIS to confirm that realistic capability constraints generalize across fraud datasets.

---

## Notes on implementation corrections (during plan execution)

Issues caught and fixed before the canonical run:

1. **`compute_metrics` signature.** Plan called `compute_metrics(model, X, y)`; actual signature is `compute_metrics(y_true, y_probs)`. Fixed in commit `8dd7c8e`.
2. **`compute_aggregate_feasibility` return shape.** Plan unpacked `(agg, per_constraint)`; actual returns `(all_pass_series, agg_float, n_constraints)` with no per-constraint dict. Per-constraint rates now computed by calling `check_g*` functions directly. Fixed in commit `8b9d9eb`.
3. **E1 term recovery.** `inverse_transform_numeric` does not recover OHE-encoded features. Term is explicitly reconstructed via `reconstruct_term_from_ohe`. Fixed in commit `ba0d6b2`.
4. **Cell 10 missing constants.** `TOLERANCE`, `G1_TOL`, and `import re` were not extracted by the AST-only function-def copy. Added explicitly. Fixed in commit `8fae681`.
5. **M2 direction resolver applied to OHE.** `resolve_direction_indices` prefix-matched `emp_length` to all 12 OHE columns. Changed to exact-match only with a warning if no columns matched. Fixed in commit `99c224c`.
6. **M0/M1 apples-to-oranges.** Summary table previously mixed fresh M2–M6 rows with historical M0/M1 loaded from `comparison_*.csv`. M0/M1 now produced inside the same variant loop as M2–M6; preprocessor is always refit. Fixed in commit `9a7a634`.
7. **E1 emp_length cost scaled by OHE-index gap.** Initial `total_cost` used `abs(a − o) / range` on the argmax indices — meaningless, since OHE column order is lexical. Switched `emp_length` to binary `(a != o)` via `BINARY_COST_FEATURES`. Fixed in commit `a928eea`; Cell 13 re-executed and outputs refreshed.
