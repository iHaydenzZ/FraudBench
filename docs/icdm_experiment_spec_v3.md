# FraudBench — ICDM 2026 Experiment Specification (v3)

**Audience:** Claude Code (implementation agent).
**Owner:** Xitong (Hayden), experiments role.
**Status:** Ready to implement. Supersedes `icdm_experiment_spec_v2.md` and the pre-flight gating in `registry_preflight_check.md`.

### What changed from v2 (two decisions)

1. **HopSkipJump is dropped from the ICDM scope.** Black-box coverage is Square Attack only. (Timing from the existing registry: HSJ ≈ 7.1 h per run vs Square ≈ 10.6 min and CAPGD ≈ 0.1 s; HSJ was 93% of all attack compute and buys only a partial cross-check. The 6 cached HSJ rows in `registry_clean.csv` exist but are **not** used here; HSJ remains future work.)
2. **Full regenerate — no reuse of prior results except nothing.** Every number is produced fresh under the schema in §3 (which carries the feasibility-aware columns the old `registry_clean.csv` lacks). The old registry is used only once, read-only, as a consistency cross-check in NB3. Rationale: the old registry is the naive-evaluation (Protocol A) layer only — it has no protocol axis and no `feasible_flipped` / `fsr` / `aggregate_feasibility` — and the protocol-gap layer must be regenerated anyway (same-model requirement); regenerating gives one self-consistent, single-provenance dataset and avoids hunting/validating old `adv_examples` files. CAPGD being ~free makes this nearly costless.

Consequences of these two decisions:
- The Sparkov OHE-aggregate fix is **no longer a separate notebook** — it is achieved by requiring the constraint evaluator to fold all OHE blocks into the aggregate conjunction for every dataset (§1.6).
- The ε sweep is **promoted from optional to regular** (CAPGD ≈ 0.1 s/run makes it free), folded into NB1 on the no-defence MLP.
- The strong B-vs-C Kendall-Tau (over MLP defence configurations) is now produced directly, because regenerating the defence × protocol layer with CAPGD is trivial; no separate gating decision.

Net structure: **three notebooks** — NB1 (CAPGD protocol grid, ~30 min), NB2 (Square model-family, ~6 h), NB3 (consolidate + diagnose, ~0).

Place this file at `docs/icdm_experiment_spec.md`.

---

## 0. Purpose and positioning (unchanged)

FraudBench is a **deployability-oriented robustness diagnosis pipeline** for financial fraud models. Headline claim — **protocol-sensitivity**:

> The same fraud model can look robust or vulnerable depending on whether evaluation uses an unconstrained attack (Protocol A), post-hoc feasibility filtering (Protocol B), or deployment-aware in-attack constraint integration (Protocol C). FraudBench makes that protocol dependence measurable.

Not "a more financial TabularBench / a complement," not "a new constrained attack."

**In scope:** regenerate the full evidence base fresh — CAPGD protocol grid (incl. defences and ε sweep), Square model-family layer, and the consolidated registry + diagnostics/figures.
**Out of scope:** HopSkipJump, CAA/MOEVA head-to-head, FA-AT, CTGAN, CCFD extra seeds, M2 ordinal, black-box in-attack Protocol C for tree/ensemble. All future work.
**Reuse, do not reimplement:** CAPGD, Square, the mutability mask, per-step projection operators, the constraint evaluator, and metric functions already exist in the repo; call them with new configs.

---

## 1. Conventions

### 1.1 Repository and paths

- Repo (local): `/Users/xitong/Local_Document/githubClone/Capstone_FraudBench`

```text
results/adv_examples/icdm_capgd_grid/        # NB1 outputs (incl. saved adversarial parquet)
results/adv_examples/icdm_square_family/     # NB2 outputs (incl. saved adversarial parquet)
results/icdm_2026/                           # NB3 deliverable (CSVs + figures + status)
```

### 1.2 Environment

- `uv` for dependencies; Colab GPU. Seed `PYTHONHASHSEED`, NumPy, Torch.
- Attacks: reuse repo CAPGD / Square. ART questions → `https://adversarial-robustness-toolbox.readthedocs.io/en/latest/`.

### 1.3 Models

- **MLP** (NB1, and the MLP rows of NB2), thesis Appendix D: two hidden layers, 128 units, ReLU, class-weighted BCE, Adam, 20 epochs; processed-space tensors (StandardScaler numeric, one-hot categorical, dataset-specific imputation), stratified subsample preserving imbalance.
- **XGBoost, Ensemble** (NB2 only): reuse repo configs (`scale_pos_weight`; soft-voting LR+XGBoost+MLP).

### 1.4 Attacks (CAPGD and Square only — NO HopSkipJump)

- **CAPGD** (NB1): `L_inf`, adaptive APGD schedule, `steps = 10`. ε per §1.7. Constraint-aware variants add per-step projection.
- **Square** (NB2): score-based black-box, 100-query budget, ε = 0.1. Model-agnostic — the only cross-model-comparable robustness axis.

### 1.5 Protocols

| `protocol` | Meaning | Produced by |
| --- | --- | --- |
| `A_unconstrained` | attack, no projection, no mask | attack run |
| `B_posthoc_filter` | feasibility filter on A's adversarial examples (TabularBench ADV+CTR convention); infeasible → rejected (revert to clean) | **derived from A's saved examples — not a new attack** |
| `C1_projection` | attack + per-step projection (g1 on LCLD; OHE blocks on IEEE-CIS / Sparkov) | attack run |
| `C2_mask_projection` | C1 + attacker-capability mutability mask | attack run |

Datasets without binding constraints (CCFD) have only `A_unconstrained` and `B_posthoc_filter` (which equals A); no C rows.

### 1.6 Constraints, tolerance, and the folded aggregate (replaces the Sparkov fix notebook)

```python
EVAL_TOL = 1e-6   # NAMED_CONSTANT, not a function argument (thesis §3.10 / Appendix E)
```

**The constraint evaluator must fold ALL one-hot blocks into the aggregate feasibility conjunction for every dataset** (this is what the old code missed for Sparkov, thesis Appendix G L9; doing it globally here removes the need for a separate fix notebook). `aggregate_feasibility` = fraction of attacked instances passing the full conjunction within `EVAL_TOL`; `main_failed_constraint` = the binding (lowest adversarial pass rate) constraint.

| Dataset | Constraints | Projected by C1 | Binding (expected) |
| --- | --- | --- | --- |
| LCLD | `g1` (instalment), `g2`, `g3`, `g4` | `g1` | `g1` |
| IEEE-CIS | `i_product_ohe`, `i_card4_ohe`, `i_card6_ohe`, `i_d_nonneg`, `i_c_nonneg`, `i_amt_positive` | the 3 OHE blocks | OHE validity |
| Sparkov | `s_state_ohe`, `s_category_ohe`, `s_gender_ohe`, `s_merch_bbox`, `s_city_pop_pos`, `s_amt_positive` | the 3 OHE blocks | `s_state_ohe` |
| CCFD | none (PCA features) | — | none → negative control |

C2 mask freezes immutable cells (LCLD: M1 mask; IEEE-CIS: the repo's documented mutable allow-list `{TransactionAmt, ProductCD, addr1/2, dist1/2}` — i.e. everything else (V*/C*/D*/card*/identity blocks) is frozen, per `ieee_cis_ohe_projection_attack.ipynb` Cell 14 / `constraint_evaluation_guidance.md` §3.3. NB: "D-fields" alone is an inaccurate shorthand — the D-fields are merely the block highlighted for the D-non-negativity variance-reduction point; the actual C2 mask is far wider, which is what reproduces the §4.4 anchor robust-accuracy ≈ 0.88).

### 1.7 ε grid

- **No-defence MLP:** ε ∈ {0.01, 0.05, 0.1, 0.15, 0.2} for A / C1 / C2 (the ε sweep).
- **Defence configs:** ε = 0.1 only (the protocol/Kendall-Tau layer).

### 1.8 Seeds

`SEEDS = [42, 123, 456]`.

### 1.9 Reproducibility, caching, same-model proof

1. Train **one** model per `(dataset, model, defence, seed)`; persist weights and compute a stable hash:

   ```python
   import hashlib
   def weight_hash(model) -> str:
       buf = b"".join(p.detach().cpu().numpy().tobytes()
                      for p in model.state_dict().values())
       return hashlib.sha256(buf).hexdigest()[:16]
   ```

   Store in `model_weight_hash`. Within a `same_model_group_id`, all protocol/ε rows must share one hash.

2. **Decouple mask from training (critical):** the repo's default retrains when the mask cell pattern changes (source of thesis §5.10 L2). Build the C2 attack against the *already-fitted* model — mask enters the attack only — and assert the hash is unchanged. Same requirement for defended models (each defence config is its own fitted model, but A/C1/C2 within a `(dataset, defence, seed)` share it).

3. Save every batch of adversarial examples as Parquet under the notebook's `adv_examples` dir, keyed by `run_id`. Re-deriving Protocol B and all feasibility metrics must read these Parquet files (no GPU re-run). This is required for B and for `feasible_flipped` / `fsr` on the Square layer.

4. Append to the results CSV immediately after each attack (`mode='a'`, header first write only); the loop must be resumable.

---

## 2. Benchmark design axes (NB3 maps every result onto these)

| # | Axis | Filled by |
| --- | --- | --- |
| 1 | Scenario + constraint audit (datasets; binding constraint per dataset; constraint-richness is binary) | NB1, NB3 |
| 2 | Model-family coverage (MLP / XGBoost / Ensemble) | NB2 |
| 3 | Three-protocol diagnostic (A / B / C) — core | NB1 |
| 4 | Attacker-capability branch (mask; C1 vs C2 separation) | NB1 |
| 5 | Reporting lens (PR-AUC + feasibility dual reporting; FSR, feasible-flipped) | NB1, NB3 |
| 6 | Defence loop (none / AT / input-validation, and protocol-sensitivity → Kendall-Tau) | NB1, NB3 |
| 7 | Perturbation budget (ε sweep, no-defence) | NB1 |

---

## 3. Output data contract (table formats)

Single canonical long-format registry; imports into Google Sheets and pivots into the paper tables.

### 3.1 `*_results.csv` — one row per (model, dataset, defence, seed, ε, protocol)

| # | Column | dtype | Notes |
| --- | --- | --- | --- |
| 1 | `run_id` | str | `{nb}__{dataset}__{model}__{defence}__s{seed}__e{eps}__{protocol}`; primary key |
| 2 | `notebook` | str | `nb1_capgd_grid` \| `nb2_square_family` |
| 3 | `dataset` | str | `CCFD` \| `IEEE-CIS` \| `LCLD` \| `Sparkov` |
| 4 | `model` | str | `MLP` \| `XGBoost` \| `Ensemble` |
| 5 | `defence` | str | `none` \| `adversarial_training` \| `input_validation` |
| 6 | `attack` | str | `CAPGD` \| `Square` |
| 7 | `protocol` | str | `A_unconstrained` \| `B_posthoc_filter` \| `C1_projection` \| `C2_mask_projection` |
| 8 | `seed` | int | 42 \| 123 \| 456 |
| 9 | `epsilon` | float | per §1.7 |
| 10 | `same_model_group_id` | str | `{dataset}__{model}__{defence}__s{seed}` |
| 11 | `model_weight_hash` | str | 16-hex; identical within a group |
| 12 | `n_test` | int | evaluated test-split size |
| 13 | `clean_pr_auc` | float | [0,1] |
| 14 | `robust_pr_auc` | float | [0,1] |
| 15 | `clean_roc_auc` | float | for PR-AUC vs ROC-AUC comparison |
| 16 | `robust_roc_auc` | float | |
| 17 | `clean_accuracy` | float | |
| 18 | `robust_accuracy` | float | |
| 19 | `flipped_count` | int | |
| 20 | `feasible_count` | int | passes full conjunction |
| 21 | `feasible_flipped_count` | int | flipped AND feasible |
| 22 | `fsr` | float | feasible_flipped / flipped |
| 23 | `aggregate_feasibility` | float | full-conjunction pass fraction |
| 24 | `main_failed_constraint` | str | binding constraint |
| 25 | `attack_runtime_sec` | float | |
| 26 | `notes` | str | |

### 3.2 `*_per_constraint.csv`
`run_id, dataset, seed, epsilon, model, defence, protocol, constraint_name, clean_pass_rate, adversarial_pass_rate, is_binding`.

### 3.3 `*_summary.csv`
Group by `(dataset, model, defence, protocol, epsilon)`; `n_seeds=3`; `mean_<m>` and `std_<m>` for `m ∈ {robust_pr_auc, robust_roc_auc, flipped_count, feasible_flipped_count, fsr, aggregate_feasibility, robust_accuracy}`.

### 3.4 Deliverable layout

```text
results/icdm_2026/
├── icdm_master_registry.csv          # union of NB1 + NB2 results (the Google-Sheets import)
├── capgd_grid_results.csv / _per_constraint.csv / _summary.csv
├── square_family_results.csv / _summary.csv
├── kendall_tau_protocol_ranking.csv
├── prauc_vs_rocauc.csv
├── thesis_consistency_check.csv      # NB3 cross-check vs old registry_clean.csv
├── golden_reference_anchors.csv
├── figures/
│   ├── fig_protocol_core.pdf / .png
│   ├── fig_epsilon_sweep.pdf / .png
│   ├── fig_model_family.pdf / .png
│   └── fig_kendall_tau.pdf / .png
└── experiment_status.md
```

---

## 4. NB1 — `icdm_capgd_protocol_grid.ipynb`

**The protocol-sensitivity core. ~30 min total (training-bound; CAPGD attacks are seconds).**

### 4.1 Goal

Regenerate the entire CAPGD-based evidence base fresh and same-model: the headline A/B/C result (no-defence), the defence × protocol layer (for the strong Kendall-Tau), and the ε sweep (no-defence). Splitting C into C1 (projection) and C2 (projection + mask) separates feasibility from attacker capability. The corrected folded-OHE aggregate (§1.6) makes Sparkov correct by construction.

### 4.2 Matrix

- Model: MLP only (CAPGD is white-box).
- Datasets: CCFD, IEEE-CIS, LCLD, Sparkov.
- Defences: `none`, `adversarial_training`, `input_validation`.
- Seeds: 42, 123, 456.
- Protocols: A, B(derived), C1, C2 — except CCFD has only A, B(=A); no C.
- ε: no-defence → {0.01,0.05,0.1,0.15,0.2}; defended → 0.1 only.

Trainings: 4 datasets × 3 defences × 3 seeds = 36 MLP. CAPGD attacks: a few hundred runs at ~0.1 s each (plus projection overhead) → minutes.

### 4.3 Run loop (same-model + weight-hash assert)

```python
for dataset in ["CCFD", "IEEE-CIS", "LCLD", "Sparkov"]:
    for defence in ["none", "adversarial_training", "input_validation"]:
        for seed in SEEDS:
            set_all_seeds(seed)
            model = train_mlp(dataset, defence, seed)     # train ONCE per group
            h = weight_hash(model); clean = cache_clean_predictions(model, dataset)
            eps_list = EPS_SWEEP if defence == "none" else [0.1]
            protocols = ["A_unconstrained"] if dataset == "CCFD" \
                        else ["A_unconstrained", "C1_projection", "C2_mask_projection"]
            for eps in eps_list:
                for protocol in protocols:
                    attack = build_capgd(model, dataset, protocol, epsilon=eps, steps=10)
                    adv = attack.generate(...)            # SAME fitted model for all protocols
                    save_parquet(adv, run_id)
                    row = evaluate(model, clean, adv, dataset, defence, protocol, eps)  # + ROC-AUC + folded-OHE aggregate
                    assert row["model_weight_hash"] == h
                    append_csv(row)
                emit_protocol_B_row_from_A(...)           # feasibility filter on A's parquet; no new attack
```

### 4.4 Acceptance anchors (no-defence MLP, ε = 0.1, means over 3 seeds)

| Dataset | Protocol | Clean PR-AUC | Robust PR-AUC | Robust Acc | Flipped | Feas-flipped | FSR | Agg.feas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LCLD | A | 0.302 | 0.105 | 0.036 | 2769±130 | **3±1** | 0.11% | 0.0012 |
| LCLD | B (derived) | 0.302 | ≈0.30 | ≈clean | = A | **3±1** | 0.11% | — |
| LCLD | C1 g1 | 0.302 | ≈0.105 | 0.039 | 2769±129 | **2111±121** | 76.5% | 0.791 |
| LCLD | C2 M1+g1 | 0.302 | ≈0.105 | 0.153 | 2888±140 | **2888±140** | 100% | 1.000 |
| IEEE-CIS | A | 0.428 | 0.068 | 0.170 | 204±6 | **0** | 0% | 0.0002 |
| IEEE-CIS | B (derived) | 0.428 | ≈0.43 | ≈clean | = A | **0** | 0% | — |
| IEEE-CIS | C1 OHE | 0.428 | 0.065 | 0.170 | 204±6 | **81±54** | 39.6% | 0.507 |
| IEEE-CIS | C2 M+OHE | 0.428 | 0.409 | 0.883 | 7±5 | **7±5** | 100% | 0.9999 |

Also expect (cross-checks): CCFD A robust PR-AUC ≈ 0.58 (negative control, A≈B, no C); Sparkov A robust PR-AUC ≈ 0.005 with `s_state_ohe` binding; the §1.6 folded aggregate must yield Sparkov A `aggregate_feasibility` ≈ 0.0001.

**Caveat:** C2 here is same-model (thesis Tables 11–12 were between-model), so C2 may shift from the anchors. Record any material shift in `notes` and `experiment_status.md`; do not force-fit to the old values.

### 4.5 Figures

**`fig_protocol_core` (headline).** Grouped bars, 2 cols (LCLD, IEEE-CIS) × 2 rows; no-defence, ε=0.1. Row 1: `fsr` (%) grouped by A/B/C1/C2, with mean `feasible_flipped_count` labelled above each bar. Row 2: `robust_pr_auc` (shows B near clean while A/C1/C2 are low). ±1 std error bars. Subtitle "single trained model per seed (identical weight hash)."

**`fig_epsilon_sweep`.** 2 cols (LCLD, IEEE-CIS); no-defence. Lines vs ε ∈ {0.01..0.2} for A/C1/C2: top panel `feasible_flipped_count` (log y, floor 0→0.5 and annotate), bottom panel `fsr`. Shaded ±1 std. Vertical dashed line at ε=0.1. Demonstrates the gap is not an ε=0.1 artefact.

---

## 5. NB2 — `icdm_square_model_family.ipynb`

**Model-family axis. ~6 h (the only real compute: 36 Square runs × ~10 min).**

### 5.1 Goal

Cover MLP / XGBoost / Ensemble on a model-agnostic robustness axis (Square), at Protocol A and B — the comparable table TabularBench cannot produce (it is deep-only); the tree model is the production differentiator. Save adversarial examples so Protocol B + feasibility metrics are computable.

### 5.2 Matrix

`4 datasets × 3 models × {A_unconstrained, B_posthoc_filter}`, Square, ε=0.1, 3 seeds, defence = `none`.
36 Square runs (one per dataset×model×seed); B derived from A's saved examples. Save adversarial parquet (needed for B + feasible-flipped/FSR — the old registry only had scalars, so this layer is regenerated).

**Do NOT fill Protocol C for XGBoost / Ensemble:**

```text
protocol = N/A
notes = "Cartella-style black-box in-attack Protocol C deferred to future work"
```

(Cartella et al. 2021 show black-box in-attack constraint enforcement on XGBoost is feasible — editability vector = mask, hot-encoded argmax = OHE projection, per-step correction = in-attack — so this is a concrete future direction, not an impossibility.)

### 5.3 Expected shape (values to-be-measured)

| Model | Metric | CCFD | IEEE-CIS | LCLD | Sparkov |
| --- | --- | --- | --- | --- | --- |
| XGBoost | A / B robust PR-AUC (Square) | measure | measure | measure | measure |
| MLP | clean PR-AUC (anchor) | 0.633 | 0.428 | 0.302 | 0.606 |
| MLP | A / B robust PR-AUC (Square) | measure | measure | measure | measure |
| Ensemble | A / B robust PR-AUC (Square) | measure | measure | measure | measure |

Notes: CCFD is the **negative control** (no binding constraint → A ≈ B, gap ≈ 1×). Square is black-box and weaker than white-box CAPGD, so an MLP's Square robust PR-AUC will generally be higher (degrades less) than its CAPGD value in NB1 — different attack, not a contradiction.

### 5.4 Figure — `fig_model_family`

Grouped bars: `robust_pr_auc` (Square) on y, grouped by model (XGBoost/MLP/Ensemble), one panel per dataset (4 panels), two bars per group (A vs B), ±std. Caption: coverage across production-relevant model families; CCFD shows A≈B.

---

## 6. NB3 — `icdm_consolidate_and_diagnose.ipynb`

**No new compute. Run last.**

Tasks:
1. **Merge** NB1 + NB2 `*_results.csv` → `icdm_master_registry.csv` (already in §3 schema, single provenance).
2. **Coverage table + ICDM summary table** (pivots over seeds), each cell tagged with its design axis (§2).
3. **Kendall-Tau** → `kendall_tau_protocol_ranking.csv`:
   - **Strong (now available):** rank the MLP defence configs that are CAPGD-evaluable `{none, adversarial_training, input_validation}` by robust PR-AUC under Protocol B, then under Protocol C (C1; or C2 — report both), per dataset; report weighted Kendall-Tau distance B-vs-C. (Ensemble defence is excluded from the C ranking — its in-attack C is ill-defined like tree.)
   - **Free:** rank the 3 model families by Square robust PR-AUC under A, then under B (from NB2), per dataset; report A-vs-B Kendall-Tau.
4. **PR-AUC vs ROC-AUC** → `prauc_vs_rocauc.csv`: per dataset, no-defence MLP, show where ROC-AUC stays high while PR-AUC reveals rare-class collapse (justifies the PR-AUC choice vs the ROC-AUC convention of prior fraud benchmarks).
5. **Thesis-consistency cross-check** → `thesis_consistency_check.csv`: load the old `registry_clean.csv` (read-only) and compare the freshly regenerated **Protocol-A, no-defence, CAPGD, ε=0.1** robust PR-AUC against it per dataset (expect agreement: CCFD≈0.58, IEEE-CIS≈0.071, LCLD≈0.105, Sparkov≈0.005). Flag any deviation beyond seed noise. Also compute the 3-seed std for CCFD to reconcile the known discrepancy (thesis 0.581±0.102 vs ICDM draft 0.580±0.225).
6. **Golden-reference self-check** against `golden_reference_anchors.csv` (the §4.4 values); warn on deviations.
7. **Figures:** `fig_kendall_tau` — bump/slope chart of each item's rank under B (left) → rank under C (right), per dataset, annotated with the weighted Kendall-Tau distance (strong version; also emit the free A-vs-B version).

---

## 7. Global figure conventions

matplotlib; save `.pdf` (vector) + `.png` (dpi=200), `bbox_inches='tight'`; serif font, size 9–10. Bars → ±1 std error bars; lines → shaded ±1 std band. Consistent protocol colours across all figures (A / B / C1 / C2). Log axes: floor 0→0.5 and annotate the true 0. One-line title, labelled axes, no chart-junk.

---

## 8. Execution order and compute budget

| Order | Notebook | New compute (real estimate) | Required |
| --- | --- | --- | --- |
| 1 | `icdm_capgd_protocol_grid` | 36 MLP trainings (~19 min) + few-hundred CAPGD runs (~minutes) ≈ **~30 min** | yes |
| 2 | `icdm_square_model_family` | 36 Square runs × ~10.6 min ≈ **~6 h** (split across Colab sessions; the only real cost) | yes |
| 3 | `icdm_consolidate_and_diagnose` | **~0** (post-processing) | yes |

Total ≈ **~7 GPU-hours**, dominated entirely by NB2's Square runs. (No HopSkipJump → none of the ~48 h HSJ cost.)

### `experiment_status.md` template

```text
Decisions: HopSkipJump dropped from ICDM scope (future work); all results regenerated fresh.

Completed:
- NB1 CAPGD protocol grid: MLP x 4 datasets x {none,AT,input_validation} x 3 seeds,
  protocols A/B/C1/C2 (CCFD A/B only), eps sweep on no-defence; folded-OHE aggregate.
- NB2 Square model-family: 4 datasets x 3 models x {A,B}, adversarial examples saved.
- NB3 consolidation: master registry, coverage + summary, Kendall-Tau (strong + free),
  PR-AUC vs ROC-AUC, thesis-consistency cross-check, figures.

Findings to confirm in writing:
- Same-model C2 vs between-model thesis Tables 11-12: <matches / differs - detail>.
- Sparkov folded aggregate: binding constraint <s_state_ohe ?>; aggregate ~0.0001 <yes/no>.
- CCFD robust PR-AUC reconciled to: <value +/- std> (vs thesis 0.581+/-0.102, ICDM draft 0.580+/-0.225).
- Strong B-vs-C Kendall-Tau distances per dataset: <values>.

Deferred (future work):
- HopSkipJump (6 cached rows exist, unused); black-box in-attack Protocol C for tree/ensemble (Cartella-style);
  CAA/MOEVA head-to-head; FA-AT; CTGAN; CCFD extra seeds; M2 ordinal.
```

---

## 9. Out of scope for this spec (paper-writing tasks, not code)

Handle separately (not in the notebooks): positioning prose ("diagnosis pipeline"; remove "complement to TabularBench" wording including the ICDM §VI.A heading), the Cartella lineage paragraph in related work, the PR-AUC-over-ROC-AUC justification, the temporal/non-IID scoped-limitation paragraph, and the one-line confirmation to the supervisor of the ICDM positioning. The notebooks produce the evidence (ROC-AUC columns, Kendall-Tau, consistency check) those paragraphs rely on.
