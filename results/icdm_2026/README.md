# `results/icdm_2026/` — ICDM 2026 Evidence Base (what each file means)

This directory is the consolidated deliverable of the three-notebook ICDM pipeline
defined in [`docs/icdm_experiment_spec_v3.md`](../../docs/icdm_experiment_spec_v3.md):

| Notebook | Role | Raw outputs here |
| --- | --- | --- |
| NB1 `icdm_capgd_protocol_grid` | White-box CAPGD protocol grid (MLP; defences; ε sweep) | `capgd_grid_*.csv` |
| NB2 `icdm_square_model_family` | Black-box Square across model families (MLP/XGBoost/Ensemble) | `square_family_*.csv` |
| NB3 `icdm_consolidate_and_diagnose` | Merge + diagnostics + figures (no new compute) | everything else |

The headline claim all of this supports (**protocol-sensitivity**, spec §0): the same
fraud model can look robust or vulnerable depending on whether evaluation uses an
unconstrained attack (Protocol A), post-hoc feasibility filtering (Protocol B), or
in-attack constraint integration (Protocol C).

---

## 1. Key concepts needed to read any CSV

**Protocols** (spec §1.5):

| `protocol` | Meaning |
| --- | --- |
| `A_unconstrained` | attack with no projection, no mask — the naive convention |
| `B_posthoc_filter` | A's adversarial examples, then infeasible ones rejected (reverted to clean). **Derived from A's saved Parquet, never a new attack** — hence `attack_runtime_sec = 0` and a note in `notes`. |
| `C1_projection` | attack with per-step constraint projection (g1 on LCLD; OHE blocks on IEEE-CIS/Sparkov) |
| `C2_mask_projection` | C1 plus the attacker-capability mutability mask |
| `not_applicable` | placeholder rows in NB2 for XGBoost/Ensemble Protocol C (black-box in-attack C is future work, Cartella-style) |

CCFD has no binding constraints (PCA features) → only A and B (= A); it is the
**negative control**.

**Feasibility metrics** (spec §3.1):

- `flipped_count` — attacked instances whose prediction flipped.
- `feasible_count` — instances passing the full constraint conjunction within `EVAL_TOL = 1e-6` (all OHE blocks folded in, spec §1.6).
- `feasible_flipped_count` — flipped **and** feasible: the only flips a real attacker could deploy.
- `fsr` — feasible_flipped / flipped. **Blank when `flipped_count = 0`** (0/0 undefined, not 0%).
- `aggregate_feasibility` — fraction of attacked instances passing the full conjunction.
- `main_failed_constraint` — the binding constraint (lowest adversarial pass rate).

**Same-model guarantee** (spec §1.9): within a `same_model_group_id`
(`{dataset}__{model}__{defence}__s{seed}`) all protocol/ε rows share one
`model_weight_hash` — the protocol gap is never a model-difference artefact.

**Settings**: seeds {42, 123, 456}; ε = 0.1 everywhere except the NB1 no-defence
MLP ε sweep {0.01, 0.05, 0.1, 0.15, 0.2}; CAPGD L∞ 10 steps; Square 100-query budget.

---

## 2. Raw per-run results (one row = one attack evaluation)

### `icdm_master_registry.csv` — 390 rows
**The canonical table.** Union of NB1 (294 rows) + NB2 (96 rows) in the spec §3.1
schema (26 columns, primary key `run_id`). This is the Google-Sheets import and the
source for every paper table. Use this unless you specifically need only one notebook's slice.

### `capgd_grid_results.csv` — 294 rows
NB1 slice: CAPGD, MLP only, 4 datasets × 3 defences ({none, adversarial_training,
input_validation}) × 3 seeds × protocols (A/B/C1/C2; CCFD A/B only) × ε
(sweep for no-defence, 0.1 for defended).

### `square_family_results.csv` — 96 rows
NB2 slice: Square, defence = none, 4 datasets × 3 models × 3 seeds × {A, B},
plus 24 `protocol = not_applicable` placeholders documenting that Protocol C for
XGBoost/Ensemble is deferred. Square is weaker than white-box CAPGD, so an MLP's
Square robust PR-AUC is generally higher than its NB1 CAPGD value — different
attack, not a contradiction (spec §5.3).

### `capgd_grid_per_constraint.csv` — 1,344 rows
One row per (run, constraint): `clean_pass_rate`, `adversarial_pass_rate`,
`is_binding`. This is where you see *which* constraint kills feasibility —
e.g. `s_state_ohe` binding on Sparkov, OHE validity on IEEE-CIS, `g1`
(instalment) on LCLD.

---

## 3. Aggregations (one row = mean ± std over 3 seeds)

- **`icdm_summary_table.csv`** (122 rows) — seed-aggregated master registry: group by
  (dataset, model, defence, attack, protocol, ε) with `mean_*`/`std_*` for the seven
  headline metrics and `n_seeds = 3`. The direct source for paper tables and figure bars.
- **`capgd_grid_summary.csv`** (98 rows) / **`square_family_summary.csv`** (24 rows) —
  the same aggregation restricted to each notebook's slice.
- **`icdm_coverage_table.csv`** (74 rows) — one row per experimental cell with
  `n_rows`, `n_seeds`, `n_eps`, a `status` (`measured` / `derived from A` /
  `not_applicable`), and `design_axes` mapping the cell onto the benchmark's seven
  design axes (spec §2). Read this to verify nothing in the claimed matrix is missing.

---

## 4. Diagnostics (the NB3 analyses)

### `kendall_tau_protocol_ranking.csv` — 10 rows
Does the *ranking* of alternatives change when the evaluation protocol changes?
Each row ranks the `items` by mean robust PR-AUC under `protocol_left` vs
`protocol_right` and reports `kendall_tau`, `weighted_tau`, and
`kt_distance = (1 − weighted_tau) / 2` (0 = identical ranking, 1 = fully reversed).
Two comparison families:

- **`strong_defence_B_vs_C`** — ranks the 3 MLP defence configs under B vs C1/C2
  (CAPGD), per constrained dataset. Result: distance **0.0 everywhere** — defence
  rankings are protocol-stable (adversarial_training > none > input_validation).
- **`free_model_A_vs_B`** — ranks the 3 model families under Square A vs B, per
  dataset. Result: CCFD identical (negative control behaves), but **IEEE-CIS fully
  inverts (τ = −1.0)** and LCLD/Sparkov partially reorder — choosing a model by
  naive Protocol A picks a different winner than feasibility-aware Protocol B.
  This is the ranking-level protocol-sensitivity evidence.

### `prauc_vs_rocauc.csv` — 24 rows
Per (dataset, attack, ε), no-defence MLP: clean/robust PR-AUC vs ROC-AUC side by
side, with `pr_drop`, `roc_drop`, and two flags — `rocauc_hides_collapse` (ROC-AUC
stays high while PR-AUC collapses on the rare class) and `rocauc_inverted`
(ROC-AUC *rises* under attack). Justifies reporting PR-AUC instead of the ROC-AUC
convention of prior fraud benchmarks (spec §6.4).

### `thesis_consistency_check.csv` — 4 rows
Cross-check of the fresh Protocol-A / no-defence / CAPGD / ε=0.1 robust PR-AUC
against the old thesis registry (`results/registry_clean.csv`, read-only), per
dataset. `delta = new_mean − old_mean`; `noise_envelope` is the seed-noise bound the
delta is judged against (combined seed std, floored at 0.01); `flag = ok` iff
|delta| ≤ envelope. All 4 datasets pass. Also reconciles the known CCFD std
discrepancy: fresh 0.598 ± 0.262 (vs thesis 0.581 ± 0.102; the ICDM draft's ± 0.225
matched neither and is superseded).

### `golden_reference_anchors.csv` — 18 rows
Self-check of the fresh no-defence MLP ε=0.1 means against the spec §4.4 acceptance
anchors (`anchor_value` ± `tolerance` vs `fresh_value` → `status`). **All 18 ok**,
including the two designed-to-be-striking cells: IEEE-CIS C2 robust accuracy 0.886
(anchor 0.883) and Sparkov A aggregate feasibility ≈ 0 (anchor 1e-4).

### `experiment_status.md`
Human-readable run log: scope decisions (HopSkipJump dropped; full regenerate),
completion state per notebook, findings to confirm in the paper text, and the
deferred future-work list.

---

## 5. `figures/` (each as `.pdf` vector + `.png` 200 dpi)

| Figure | What it shows | Takeaway |
| --- | --- | --- |
| `fig_protocol_core` | **Headline.** LCLD & IEEE-CIS, no-defence, ε=0.1. Top row: FSR (%) by protocol A/B/C1/C2 with mean feasible-flipped counts labelled; bottom row: robust PR-AUC. ±1 std bars; single weight hash per seed. | Protocol A reports thousands of flips of which ~0 are feasible; B looks near-clean; C1/C2 expose the deployable threat. Same model, three different verdicts. |
| `fig_epsilon_sweep` | LCLD & IEEE-CIS, no-defence, A/C1/C2 vs ε ∈ {0.01…0.2}. Top: feasible-flipped count (log y, 0 floored to 0.5 and annotated); bottom: FSR. Dashed line at ε=0.1. | The protocol gap persists across the whole ε range — not an ε=0.1 artefact. |
| `fig_model_family` | Robust PR-AUC (Square) by model (MLP/XGBoost/Ensemble), one panel per dataset, A vs B bars, ±std. | Model-family coverage TabularBench can't produce (deep-only); CCFD shows A ≈ B (negative control). |
| `fig_kendall_tau` | Bump/slope chart: defence-config rank under B (left) → under C (right) per dataset, annotated with weighted-KT distance (strong comparison). | Defence rankings are protocol-stable (all distances 0.0). |
| `fig_kendall_tau_free` | Same chart for the free comparison: model-family rank under Square A → B. | Model rankings are *not* protocol-stable — IEEE-CIS fully inverts. |

---

## 6. Provenance & reproducibility

- Every number here was regenerated fresh under spec v3 (no reuse of the old
  registry except as the read-only cross-check in `thesis_consistency_check.csv`).
- Raw adversarial examples behind every row are saved as Parquet, keyed by `run_id`,
  under `results/adv_examples/icdm_capgd_grid/` and
  `results/adv_examples/icdm_square_family/` — Protocol B and all feasibility
  metrics are re-derivable from them without re-running attacks.
- The old thesis-era analysis should keep using `results/registry_clean.csv`;
  everything ICDM-facing should use `icdm_master_registry.csv` from this directory.
