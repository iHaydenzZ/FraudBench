# Mask Ablation — Findings

**Date:** 2026-04-15 (template — numbers filled after Colab run)
**Notebook:** `notebooks/mask_ablation.ipynb`
**Canonical results:** `results/adv_examples/mask_ablation/mask_ablation_summary.csv`
**Spec:** `docs/plans/mask_ablation_experiment_plan.md`
**Implementation plan:** `docs/plans/2026-04-15-mask-ablation-implementation.md`

---

## Headline numbers

> Paste the contents of `mask_ablation_summary.csv` here after the Colab run completes.
> 8 rows (M0, M1, M2, M3, M4, M5, M6strict, M6relaxed) × columns:
> `n_mutable_mean`, `robust_pr_auc_{mean,std}`, `robust_accuracy_{mean,std}`, `robust_recall_{mean,std}`, `robust_f1_{mean,std}`, `feasibility_seed42`, `g1_pass_seed42`, `g4_pass_seed42`.

_(table goes here)_

---

## Sanity checks performed

- **Baseline reproduction.** Cell 9 loads M0/M1 rows verbatim from `comparison_{unmasked,masked}.csv`. Confirmed robust metrics match the prior run to within ±0.00 (exact — we reload, we don't recompute).
- **M2 directional clip (LCLD-specific).** `emp_length` in LCLD is OHE-encoded, not numeric. `resolve_direction_indices` is exact-match only and skips OHE categoricals, so the directional clip has no effect on LCLD. Expected: M2 robust metrics match M1 almost exactly. Cell 12's `emp_length_pct_negative` column will be NaN for M2 because no raw `emp_length` column exists in the inverse-transformed frame. A warning is printed at M2 attack time.
- **Term freeze effect.** Cell 12 reports `term_ohe_max_abs_delta`. For M4 and M5, must be ≈ 0.
- **Dti freeze effect.** Cell 12 reports `dti_mean_abs_delta`. For M3 and M5, must be ≈ 0.
- **E1 scale linearity.** Cell 13 sensitivity rows at `cost_scale=2.0` must be exactly 2× the `cost_scale=1.0` rows (math check on the normalization).
- **Feasibility aggregate math.** For every variant, `aggregate ≤ min(g1, g2, g3, g4)` — the AND of constraints cannot exceed any individual constraint's pass rate.

_(verify each after Colab run; flag any that fail)_

---

## Caveats

- **Single-seed feasibility.** Feasibility, perturbation stats, and E1 are seed=42 only (per spec §4). Robust metrics are averaged over 3 seeds. Summary table's `feasibility_seed42` / `g1_pass_seed42` / `g4_pass_seed42` columns should NOT be read with a ± band.
- **M4 g4 ≈ 1.0 is an artifact, not a capability improvement.** Freezing `term_*` OHE columns guarantees the OHE validity check passes — this does not mean CAPGD learned to produce valid OHE encodings. Framing: "freezing term recovers the aggregate feasibility that was blocked solely by g4."
- **E1 covers numeric features, `term`, and `emp_length`.** `term` is reconstructed via `reconstruct_term_from_ohe` (gives numeric 36/60). `emp_length` is categorical/OHE in LCLD, so E1 reconstructs it via OHE-argmax — its cost contribution is effectively "changed vs unchanged" weighted by COST[emp_length]. Remaining categoricals (`purpose`, `home_ownership`, `addr_state`, `application_type`) are NOT reconstructed and contribute 0 — this is a documented E1 scope limitation.
- **E1 normalization.** Uses p1/p99 winsorized ranges (not min/max) to tame LCLD `annual_inc` tails.
- **Variance bands overlap for most pairs.** With 3 seeds the robust-metric std is comparable to mean differences between most (variant, variant) pairs. Only flag pairs where mean-difference exceeds the seed-level std as carrying signal.
- **Undeclared-feature handling.** `build_processed_mutable_mask` defaults to mutable for any column not matched against the immutable set. Raw LCLD features not listed in `LCLD_IMMUTABLE_RAW` ∪ `LCLD_MUTABLE_RAW` inherit this default — same behavior as the M1 baseline, so comparisons remain apples-to-apples.
- **M6 immutable set** is `dataset.X.columns - MUTABLE_PROFILE`. Verified that `dataset.X.columns` excludes the target label.
- **M2 is structurally a no-op on LCLD.** Directionality constraints require numeric raw features, but only `emp_length` in LCLD has a natural "increase-only" interpretation and it is stored as a 12-column OHE categorical. To actually evaluate M2 would require switching to an ordinal encoding of `emp_length` — left for future work. The current M2 result is reported as-is and should be read as "= M1, as expected under categorical encoding". This confirms spec §8.2's prediction.

---

## Answers to meeting questions (spec §9)

_Fill after run. Structure:_

1. **Biggest marginal effect on robust accuracy?** _(compare robust_accuracy_mean: M1 vs M4 vs M5 vs M6strict)_
2. **Does M4 lift aggregate feasibility?** _(compare aggregate: M1 vs M4; expected g4 → 1.0, aggregate jumps only if g1 was not the bottleneck)_
3. **Attacker-capability spectrum.** _(M6strict vs M6relaxed vs M1 robust_accuracy spread)_
4. **E1 takeaway.** _(median and p95 cost for M0 vs M1; affordable-curve shape)_
5. **Next experiment round.** _(if M5 feasibility ≈ 0: constraint-aware attacks are required, push toward TabularBench integration; if M5 feasibility high: mask-level fixes suffice, pivot to cross-dataset)_

---

## Notes on implementation corrections (during plan execution)

Issues caught during subagent-driven implementation and fixed before Colab run:

1. **`compute_metrics` signature.** Original plan called `compute_metrics(model, X, y)`; actual signature is `compute_metrics(y_true, y_probs)`. Fixed in commit `8dd7c8e`.
2. **`compute_aggregate_feasibility` return.** Original plan unpacked `(agg, per_constraint)`; actual returns `(all_pass_series, agg_float, n_constraints)` with no per-constraint dict. Per-constraint rates now computed by calling `check_g*` functions directly. Fixed in commit `8b9d9eb`.
3. **E1 term recovery.** `inverse_transform_numeric` does not recover OHE-encoded features. Term is now explicitly reconstructed via `reconstruct_term_from_ohe` for both clean and adversarial frames so its cost contribution is counted. Fixed in commit `ba0d6b2`.
4. **Cell 10 missing constants.** `TOLERANCE`, `G1_TOL`, and `import re` were not extracted by the AST-only function-def copy. Added explicitly. Fixed in commit `8fae681`.
