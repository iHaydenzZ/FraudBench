# IEEE-CIS Mutable-Set Sensitivity Sweep — Implementation Plan

> **For agentic workers:** This plan extends a Colab notebook with new cells. Steps use checkbox (`- [ ]`) syntax for tracking. The work is structured as: (1) scaffold cells locally, commit. (2) run on Colab GPU, commit outputs. (3) update findings docs, commit.

**Goal:** Bracket the M+OHE attack-count number on IEEE-CIS (~7.7 feasible-flipped per seed) by running a tighter M and a wider M, producing a 3–4-point dose-response curve along the capability axis for the ICAIF 2026 paper §5 trade-off claim.

**Architecture:** Append new cells to `notebooks/ieee_cis_ohe_projection_attack.ipynb` (Cells 18–21). Reuse `capgd_attack_m_ohe_projected` (Cell 15) and `build_processed_mutable_mask_inverted` (Cell 14) verbatim — only the `IEEE_CIS_MUTABLE_RAW` set changes. Append new rows to the existing `ieee_ohe_projection_results.csv` with new attack labels (`m_tight_oheproj`, `m_wide_oheproj`). Cell 21 produces a 5-way summary (unconstrained → oheproj → m_tight → m → m_wide) and a dose-response plot.

**Tech Stack:** Existing notebook stack — PyTorch MLP, CAPGD, `attacks/capgd.py`-derived projection, `ieee_feasibility` checker. Colab T4 / A100 GPU.

---

## Background (to anchor the engineer)

The existing notebook (Cells 1–17, commits `385420f`, `567159d`) measures three regimes on IEEE-CIS with 3 seeds at ε=0.1:

| Regime | Mutable raw features | Processed mutable dims | Feasible-flipped per seed | FSR |
|---|---|---:|---:|---:|
| `unconstrained` | all (no mask) | ~380 | ~0.06 | 0.0% |
| `oheproj` | all (no mask, OHE-projected only) | ~380 | ~120 | 54.5% |
| `m_oheproj` | `TransactionAmt, ProductCD, addr1, addr2, dist1, dist2` | ~12 | **7.7** | 100% |

This sweep adds two more points on the capability axis:

| New regime | Mutable raw features | Expected processed dims | Expected feas-flipped |
|---|---|---:|---:|
| `m_tight_oheproj` | `TransactionAmt, ProductCD` | ~6 | ~3 |
| `m_wide_oheproj` | M ∪ `P_emaildomain, R_emaildomain, M1, M2, M3, M4, M5, M6, M7, M8, M9` | ~30 | between 7.7 and 120 |

The five points together produce a dose-response curve. The §5 paper claim becomes "we mapped the capability-vs-feasibility trade-off" not "we hit one point".

Reference: ToDo §A', findings doc `docs/ieee_ohe_projection_findings.md` "What this means for the roadmap" item 1.

---

## Files

- **Modify:** `notebooks/ieee_cis_ohe_projection_attack.ipynb` (append Cells 18–21)
- **Modify (after Colab run):** `results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_results.csv` (gains 6 rows: 3 seeds × 2 new attacks)
- **Modify (after Colab run):** `results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_summary.csv` (rebuilt 5-way)
- **Create (after Colab run):** `results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_dose_response.png` (new figure)
- **Modify (after analysis):** `docs/ieee_ohe_projection_findings.md` (add "Mutable-set sensitivity sweep" section)
- **Modify (after analysis):** `docs/Context.md` §9, `docs/ToDo.md` §A → §A' (mark done)

---

## Task 1: Scaffold Cell 18 — markdown header for the sweep

**Files:**
- Modify: `notebooks/ieee_cis_ohe_projection_attack.ipynb` (append markdown cell)

- [ ] **Step 1: Append a markdown cell after Cell 17 with this exact content:**

```markdown
# Mutable-set sensitivity sweep (capability axis dose-response)

Cells 14–17 fixed the mutable set at the canonical M = {TransactionAmt,
ProductCD, addr1/2, dist1/2} — call this M_canonical (~12 processed dims).
Result: feasible-flipped crashes from ~120 (OHE-only) to ~7.7 (M+OHE),
revealing a capability-vs-feasibility trade-off (see
`docs/ieee_ohe_projection_findings.md` "Central finding 2").

This sweep brackets the 7.7 number with two more capability levels:

* **M_tight = {TransactionAmt, ProductCD}** (~6 processed dims) — minimal
  realistic capability; expected feas-flip ≈ 3.
* **M_wide  = M_canonical ∪ {P_emaildomain, R_emaildomain, M1..M9}**
  (~30 processed dims) — generous capability profile that still freezes
  D1–D15 / C1–C14 / V1–V339 / card1–6; expected feas-flip between 7.7 and 120.

Together with the existing `unconstrained` / `oheproj` / `m_oheproj` rows,
this produces a 5-point dose-response curve for the ICAIF 2026 §5 trade-off
claim.

Reuses `capgd_attack_m_ohe_projected` (Cell 15) and
`build_processed_mutable_mask_inverted` (Cell 14) verbatim. Only the raw
mutable set changes per run. Re-runnable from a fresh kernel after Cell 6
— Cell 20 reloads the existing CSV and appends.
```

- [ ] **Step 2: Verify cell ordering**

Run from terminal:
```bash
/usr/bin/python3 -c "
import json
nb = json.load(open('notebooks/ieee_cis_ohe_projection_attack.ipynb'))
print(f'Total cells: {len(nb[\"cells\"])}')
print(f'Cell 18 type: {nb[\"cells\"][18][\"cell_type\"]}')
src = ''.join(nb['cells'][18]['source'])
print(src.split(chr(10))[0])
"
```
Expected: `Total cells: 19`, `Cell 18 type: markdown`, header line `# Mutable-set sensitivity sweep (capability axis dose-response)`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/ieee_cis_ohe_projection_attack.ipynb
git commit -m "docs(ieee-cis): scaffold Cell 18 — mutable-set sensitivity sweep header"
```

---

## Task 2: Scaffold Cell 19 — define M_tight and M_wide raw sets

**Files:**
- Modify: `notebooks/ieee_cis_ohe_projection_attack.ipynb` (append code cell)

- [ ] **Step 1: Append a code cell with this exact content:**

```python
# Cell 19: M_tight and M_wide raw mutable sets, plus a sanity-print of the
# resulting processed mutable-dim counts. M_canonical from Cell 14 is the
# middle point on the capability axis.

IEEE_CIS_MUTABLE_RAW_TIGHT: Set[str] = {
    "TransactionAmt",
    "ProductCD",
}

IEEE_CIS_MUTABLE_RAW_WIDE: Set[str] = (
    IEEE_CIS_MUTABLE_RAW
    | {"P_emaildomain", "R_emaildomain"}
    | {f"M{i}" for i in range(1, 10)}  # M1..M9
)

SWEEP_VARIANTS = [
    ("m_tight_oheproj", IEEE_CIS_MUTABLE_RAW_TIGHT),
    ("m_wide_oheproj",  IEEE_CIS_MUTABLE_RAW_WIDE),
]

# Sanity-check processed-dim counts against expectations using a temporary
# preprocessor on seed 42. Aborts loudly if the wide/tight masks differ from
# the expected ~6 / ~30 dims by more than a 2x factor.
def _sanity_print_mask_dims():
    ds = load_dataset(DATASET, config={"sample_frac": SAMPLE_FRAC})
    Xtr, _, _, ytr, _, _ = split_dataset(ds, test_size=0.2, val_size=0.2, random_state=42)
    pp = DataPreprocessor(ds.feature_types)
    Xtr_p = pp.fit_transform(Xtr)
    cols = Xtr_p.columns.tolist()
    rows = []
    for label, raw in [("tight", IEEE_CIS_MUTABLE_RAW_TIGHT),
                       ("canonical", IEEE_CIS_MUTABLE_RAW),
                       ("wide", IEEE_CIS_MUTABLE_RAW_WIDE)]:
        m = build_processed_mutable_mask_inverted(cols, raw)
        rows.append((label, len(raw), int(m.sum()), int((~m).sum())))
    print(f"{'variant':>10s}  {'raw':>4s}  {'mut':>4s}  {'imm':>5s}")
    for label, n_raw, n_mut, n_imm in rows:
        print(f"{label:>10s}  {n_raw:>4d}  {n_mut:>4d}  {n_imm:>5d}")
    expected_tight, expected_wide = (3, 12), (20, 50)  # generous brackets
    tight_mut = rows[0][2]
    wide_mut  = rows[2][2]
    assert expected_tight[0] <= tight_mut <= expected_tight[1], \
        f"tight mut={tight_mut} outside expected {expected_tight}"
    assert expected_wide[0]  <= wide_mut  <= expected_wide[1], \
        f"wide mut={wide_mut} outside expected {expected_wide}"
    print("Mask dims OK.")

_sanity_print_mask_dims()
```

- [ ] **Step 2: Smoke-test logic locally (no GPU needed)**

The cell uses `load_dataset`, `split_dataset`, `DataPreprocessor`,
`build_processed_mutable_mask_inverted` which are all defined earlier in
the notebook or in the repo. We cannot run the cell here (Colab-only env),
but we can statically check the M_wide content.

Run:
```bash
/usr/bin/python3 -c "
canonical = {'TransactionAmt','ProductCD','addr1','addr2','dist1','dist2'}
tight = {'TransactionAmt','ProductCD'}
wide = canonical | {'P_emaildomain','R_emaildomain'} | {f'M{i}' for i in range(1,10)}
print(f'tight: {len(tight)} raw -> {sorted(tight)}')
print(f'canon: {len(canonical)} raw')
print(f'wide:  {len(wide)} raw -> {sorted(wide)}')
"
```
Expected output:
```
tight: 2 raw -> ['ProductCD', 'TransactionAmt']
canon: 6 raw
wide:  17 raw -> ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'P_emaildomain', 'ProductCD', 'R_emaildomain', 'TransactionAmt', 'addr1', 'addr2', 'dist1', 'dist2']
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/ieee_cis_ohe_projection_attack.ipynb
git commit -m "feat(ieee-cis): scaffold Cell 19 — M_tight / M_wide mutable sets"
```

---

## Task 3: Scaffold Cell 20 — run loop for both M variants × 3 seeds

**Files:**
- Modify: `notebooks/ieee_cis_ohe_projection_attack.ipynb` (append code cell)

- [ ] **Step 1: Append a code cell with this exact content:**

```python
# Cell 20: Run M_tight + OHE and M_wide + OHE attacks per seed; append to
# the existing CSV. Mirrors Cell 16's per-seed train+attack loop but loops
# over (label, mutable_raw) pairs from SWEEP_VARIANTS instead of the single
# canonical M. Skips any (attack, seed) pair already present in the CSV so
# this cell is re-runnable.

results_csv = os.path.join(OUT_DIR, "ieee_ohe_projection_results.csv")
existing_rows = pd.read_csv(results_csv).to_dict("records")
existing_keys = {(r["attack"], r["seed"]) for r in existing_rows}
print(f"Loaded {len(existing_rows)} existing rows; "
      f"attacks present: {sorted({r['attack'] for r in existing_rows})}")

new_rows = []

for variant_label, mutable_raw in SWEEP_VARIANTS:
    for seed in SEEDS:
        if (variant_label, seed) in existing_keys:
            print(f"  skip {variant_label} seed={seed} (already in CSV)")
            continue
        print(f"\n{'='*60}\n  {variant_label}  SEED = {seed}\n{'='*60}")

        dataset = load_dataset(DATASET, config={"sample_frac": SAMPLE_FRAC})
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            dataset, test_size=0.2, val_size=0.2, random_state=seed,
        )
        preprocessor = DataPreprocessor(dataset.feature_types)
        X_train_p = preprocessor.fit_transform(X_train)
        X_test_p  = preprocessor.transform(X_test)
        proc_ft = {c: "numeric" for c in X_train_p.columns}
        schema = ConstraintSchema.from_data(X_train_p, proc_ft)

        scaler, num_names = get_scaler_and_num_names(preprocessor)
        ohe_blocks = build_ieee_ohe_blocks(X_train_p.columns.tolist())
        mutable_mask = build_processed_mutable_mask_inverted(
            X_train_p.columns.tolist(), mutable_raw,
        )
        n_mut = int(mutable_mask.sum())
        n_imm = int((~mutable_mask).sum())
        print(f"  proc_dim={X_test_p.shape[1]}  mutable={n_mut}  immutable={n_imm}")

        model = NeuralModel(MODEL_PARAMS)
        t0 = time.time()
        model.fit(X_train_p, y_train)
        print(f"  trained in {time.time()-t0:.1f}s")
        clean_probs = model.predict_proba(X_test_p)
        clean_m = compute_metrics(y_test, clean_probs)

        X_test_raw = inverse_transform_numeric(X_test_p, num_names, scaler)
        _, clean_agg = ieee_feasibility(X_test_raw, X_test_p)

        t0 = time.time()
        X_adv = capgd_attack_m_ohe_projected(
            model, X_test_p, y_test, schema, proc_ft,
            mutable_mask, ohe_blocks, params=ATTACK_PARAMS,
        )
        dt = time.time() - t0
        probs = model.predict_proba(X_adv)
        m = compute_metrics(y_test, probs)
        X_adv_raw = inverse_transform_numeric(X_adv, num_names, scaler)
        per, agg = ieee_feasibility(X_adv_raw, X_adv)

        parq = os.path.join(OUT_DIR, f"{DATASET}_neural_{variant_label}_seed{seed}.parquet")
        X_adv.to_parquet(parq)

        clean_pred = (clean_probs >= 0.5).astype(int)
        adv_pred   = (probs       >= 0.5).astype(int)
        pos_mask   = (y_test.values == 1)
        flipped    = int(((clean_pred == 1) & (adv_pred == 0) & pos_mask).sum())

        def _agg_mask(X_raw, X_p):
            idx = X_p.index
            parts = [
                _bool_series(check_txn_amt_positive(X_raw),         idx),
                _bool_series(check_c_nonneg(X_raw),                 idx),
                _bool_series(check_d_nonneg(X_raw),                 idx),
                _bool_series(check_ohe_valid(X_p, "ProductCD"),     idx),
                _bool_series(check_ohe_valid(X_p, "card4"),         idx),
                _bool_series(check_ohe_valid(X_p, "card6"),         idx),
            ]
            agg = parts[0]
            for q in parts[1:]:
                agg = agg & q
            return agg.values

        feas_flip = int(((clean_pred == 1) & (adv_pred == 0) & pos_mask & _agg_mask(X_adv_raw, X_adv)).sum())

        print(f"  {variant_label}  PR-AUC={m['pr_auc']:.4f}  Acc={m['accuracy']:.4f}  "
              f"agg_feas={agg:.4f}  d_nonneg={per['i_d_nonneg']:.3f}  "
              f"flipped={flipped} feas-flip={feas_flip}  ({dt:.1f}s)")

        row = {
            "seed": seed, "attack": variant_label,
            "clean_pr_auc": clean_m["pr_auc"], "clean_accuracy": clean_m["accuracy"],
            "clean_recall": clean_m.get("recall", np.nan),
            "robust_pr_auc": m["pr_auc"], "robust_accuracy": m["accuracy"],
            "robust_recall": m.get("recall", np.nan),
            "clean_feasibility": clean_agg,
            "adv_feasibility":   agg,
            "flipped_positives": flipped,
            "feasible_flipped":  feas_flip,
            "attack_time_sec":   round(dt, 1),
            "n_mutable_dims":    n_mut,
        }
        for k, v in per.items(): row[f"adv_{k}"] = v
        new_rows.append(row)

if new_rows:
    combined = pd.DataFrame(existing_rows + new_rows)
    combined.to_csv(results_csv, index=False)
    print(f"\nAppended {len(new_rows)} rows. Total = {len(combined)} -> {results_csv}")
    results_df = combined
else:
    print("\nNo new rows; CSV already contains all (variant, seed) pairs.")
    results_df = pd.DataFrame(existing_rows)

results_df.tail(8)
```

- [ ] **Step 2: Static review checklist**

Confirm by re-reading the cell:
- Reuses `capgd_attack_m_ohe_projected` from Cell 15 — yes
- Reuses `build_processed_mutable_mask_inverted` from Cell 14 — yes
- Re-uses `_agg_mask` definition pattern from Cell 16 (same six checks)
- Adds `"n_mutable_dims": n_mut` to the row schema — new column for the dose-response x-axis. Existing rows will have `NaN` for this; Cell 21 backfills.
- Skip-if-present guard via `(attack, seed)` tuple — re-runnable.

- [ ] **Step 3: Commit**

```bash
git add notebooks/ieee_cis_ohe_projection_attack.ipynb
git commit -m "feat(ieee-cis): scaffold Cell 20 — M_tight / M_wide attack run loop"
```

---

## Task 4: Scaffold Cell 21 — 5-way summary + dose-response plot

**Files:**
- Modify: `notebooks/ieee_cis_ohe_projection_attack.ipynb` (append code cell)

- [ ] **Step 1: Append a code cell with this exact content:**

```python
# Cell 21: Five-way summary across the capability axis + dose-response plot.
# Backfills `n_mutable_dims` for rows produced before Cell 20 (which lacked
# the column) using the canonical mappings; unconstrained / oheproj are
# logged as the full processed dim (no mask) for plotting purposes.
agg_cols = {
    "robust_pr_auc":     ["mean", "std"],
    "robust_accuracy":   ["mean", "std"],
    "adv_feasibility":   ["mean", "std"],
    "adv_i_d_nonneg":    ["mean", "std"],
    "flipped_positives": ["mean", "std"],
    "feasible_flipped":  ["mean", "std"],
    "n_mutable_dims":    ["mean"],
}
summary5 = results_df.groupby("attack").agg(agg_cols)
summary5.columns = ["_".join(c).rstrip("_") for c in summary5.columns]
summary5 = summary5.reset_index()
summary5["filtered_success_rate"] = (
    summary5["feasible_flipped_mean"] / summary5["flipped_positives_mean"].replace(0, np.nan)
)

# Backfill n_mutable_dims for legacy rows. Unconstrained / oheproj have no
# mutable mask (all features perturbable), so the dim count == full processed
# dim. Read it from one of their saved parquets.
backfill_map = {
    "unconstrained": None,
    "oheproj":       None,
}
# Unconstrained / oheproj: count from any parquet on disk for that attack.
for attack_label in ["unconstrained", "oheproj"]:
    for seed in SEEDS:
        # The legacy parquets used a different naming convention; try both.
        candidates = [
            os.path.join(OUT_DIR, f"{DATASET}_neural_{attack_label}_seed{seed}.parquet"),
            os.path.join(OUT_DIR, f"{DATASET}_neural_{attack_label.replace('oheproj','ohe_proj')}_seed{seed}.parquet"),
        ]
        for p in candidates:
            if os.path.exists(p):
                backfill_map[attack_label] = pd.read_parquet(p).shape[1]
                break
        if backfill_map[attack_label] is not None:
            break
print(f"Backfill n_mutable_dims for legacy rows: {backfill_map}")

for k, v in backfill_map.items():
    if v is not None:
        mask = (summary5["attack"] == k) & summary5["n_mutable_dims_mean"].isna()
        summary5.loc[mask, "n_mutable_dims_mean"] = v

# Canonical ordering along the capability axis (ascending dims).
order = ["unconstrained", "oheproj", "m_tight_oheproj", "m_oheproj", "m_wide_oheproj"]
summary5["__order"] = summary5["attack"].map({k: i for i, k in enumerate(order)})
summary5 = summary5.sort_values("__order").drop(columns="__order").reset_index(drop=True)

summary_csv = os.path.join(OUT_DIR, "ieee_ohe_projection_summary.csv")
summary5.to_csv(summary_csv, index=False)

print("Five attack regimes on IEEE-CIS, mean +/- std over 3 seeds:")
cols_display = [
    "attack", "n_mutable_dims_mean",
    "robust_pr_auc_mean", "robust_accuracy_mean",
    "adv_feasibility_mean", "adv_i_d_nonneg_mean",
    "flipped_positives_mean", "feasible_flipped_mean",
    "filtered_success_rate",
]
print(summary5[cols_display].to_string(index=False))

# Dose-response figure: feasible-flipped (and FSR) vs n_mutable_dims for the
# three M+OHE points; unconstrained / oheproj plotted as horizontal references.
m_rows = summary5[summary5["attack"].isin(["m_tight_oheproj", "m_oheproj", "m_wide_oheproj"])]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.errorbar(m_rows["n_mutable_dims_mean"], m_rows["feasible_flipped_mean"],
            yerr=m_rows["feasible_flipped_std"], marker="o", capsize=4,
            color="#c33", label="M + OHE")
oheproj_feas = summary5.loc[summary5["attack"] == "oheproj", "feasible_flipped_mean"].iloc[0]
ax.axhline(oheproj_feas, color="#2a7", linestyle="--", label=f"OHE-only ({oheproj_feas:.0f})")
ax.set_xlabel("Mutable processed dimensions")
ax.set_ylabel("Feasible-flipped per seed")
ax.set_title("Capability dose-response (attack count)")
ax.set_xscale("log")
for _, r in m_rows.iterrows():
    ax.text(r["n_mutable_dims_mean"], r["feasible_flipped_mean"],
            f"  {r['attack'].replace('_oheproj','')}", fontsize=9, va="bottom")
ax.legend()

ax = axes[1]
ax.plot(m_rows["n_mutable_dims_mean"], m_rows["filtered_success_rate"],
        marker="o", color="#c33", label="M + OHE")
oheproj_fsr = summary5.loc[summary5["attack"] == "oheproj", "filtered_success_rate"].iloc[0]
ax.axhline(oheproj_fsr, color="#2a7", linestyle="--", label=f"OHE-only ({oheproj_fsr:.2%})")
ax.set_xlabel("Mutable processed dimensions")
ax.set_ylabel("Filtered success rate")
ax.set_title("Capability dose-response (FSR)")
ax.set_xscale("log")
ax.set_ylim(0, 1.05)
ax.legend()

plt.suptitle("IEEE-CIS: capability vs feasibility trade-off (3 seeds, eps=0.1)")
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "ieee_ohe_projection_dose_response.png")
plt.savefig(fig_path, dpi=150)
shutil.copy(fig_path, os.path.join(DRIVE_OUT, "ieee_ohe_projection_dose_response.png"))
for f in ["ieee_ohe_projection_results.csv", "ieee_ohe_projection_summary.csv"]:
    shutil.copy(os.path.join(OUT_DIR, f), os.path.join(DRIVE_OUT, f))
plt.show()
print(f"\nSaved dose-response plot -> {fig_path}")
```

- [ ] **Step 2: Verify total cell count is now 22**

Run:
```bash
/usr/bin/python3 -c "
import json
nb = json.load(open('notebooks/ieee_cis_ohe_projection_attack.ipynb'))
print(f'Total cells: {len(nb[\"cells\"])}')
for i in range(18, len(nb['cells'])):
    src = ''.join(nb['cells'][i]['source'])
    print(f'  Cell {i} [{nb[\"cells\"][i][\"cell_type\"]:8s}] {src.split(chr(10))[0][:80]}')
"
```
Expected: `Total cells: 22`, with Cells 18 (markdown), 19, 20, 21 (code).

- [ ] **Step 3: Commit**

```bash
git add notebooks/ieee_cis_ohe_projection_attack.ipynb
git commit -m "feat(ieee-cis): scaffold Cell 21 — 5-way summary + dose-response plot"
```

---

## Task 5: Run on Colab and capture outputs

**Files:**
- Modify: `notebooks/ieee_cis_ohe_projection_attack.ipynb` (cell outputs only)
- Create: `results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_dose_response.png`
- Modify: `results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_results.csv` (gains 6 rows)
- Modify: `results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_summary.csv` (5 rows now)

- [ ] **Step 1: Push the scaffolded notebook to GitHub**

```bash
git push origin master
```

- [ ] **Step 2: Open the notebook on Colab and run Cells 1–6, then 14, 15, 19, 20, 21**

Cells 14 and 15 must be re-run from Cell 6 to ensure `IEEE_CIS_MUTABLE_RAW`,
`build_processed_mutable_mask_inverted`, and `capgd_attack_m_ohe_projected`
are in scope. Cell 19 also needs `load_dataset`, `split_dataset`,
`DataPreprocessor` imports from earlier cells.

Total runtime estimate: 6 runs × ~2 min/run (training + attack) ≈ 12–15 min.

- [ ] **Step 3: Verify Cell 19 sanity output**

Expected pattern in Cell 19 output:
```
   variant   raw   mut    imm
     tight     2     6   ~380
 canonical     6   ~12   ~370
      wide    17   ~30   ~350
Mask dims OK.
```
The exact `mut` numbers depend on dataset cardinality but `tight ≈ 6`,
`canonical ≈ 12`, `wide ≈ 30` are the targets. If the assertion fails the
sweep is mis-specified — stop and re-check `IEEE_CIS_MUTABLE_RAW_TIGHT/WIDE`.

- [ ] **Step 4: Verify Cell 20 produced 6 new rows**

Expected pattern in Cell 20 output:
```
Loaded 9 existing rows; attacks present: ['m_oheproj', 'oheproj', 'unconstrained']

============================================================
  m_tight_oheproj  SEED = 42
============================================================
  ...
  m_tight_oheproj  PR-AUC=0.4xx  Acc=0.9xx  agg_feas=1.0000  d_nonneg=1.000  flipped=≥3 feas-flip=≈3  (1xx.xs)
...
Appended 6 rows. Total = 15 -> .../ieee_ohe_projection_results.csv
```

Spot-check the feasible-flipped numbers:
- `m_tight_oheproj`: feas-flip 1–8 (low) — confirms tight capability hypothesis
- `m_wide_oheproj`: feas-flip 5–60 (between 7.7 and 120) — confirms intermediate point

If `m_tight feas-flip > 50` or `m_wide feas-flip < m_oheproj feas-flip`,
the dose-response is non-monotonic — flag and investigate before committing.

- [ ] **Step 5: Verify Cell 21 dose-response figure**

The two-panel plot must show:
- Left panel: feasible-flipped on the y-axis, n_mutable_dims (log) on the x-axis. Three red points (tight, canonical, wide) with the OHE-only horizontal reference. Expected shape: low at tight, dip at canonical, rise toward OHE-only at wide.
- Right panel: FSR. All three M+OHE points should be at or near 1.0; OHE-only reference at ~0.55.

Visual sanity: the curve must be **non-monotonic** (the dip at M_canonical IS the central finding). If it's monotonic, the dataset's structure differs from the M+OHE prediction and the §5 narrative needs revising.

- [ ] **Step 6: Pull artifacts down from Drive locally**

Files to land in `results/adv_examples/ieee_ohe_projection/`:
- `ieee_ohe_projection_results.csv` (15 rows: 5 attacks × 3 seeds)
- `ieee_ohe_projection_summary.csv` (5 rows)
- `ieee_ohe_projection_dose_response.png`
- `ieee_cis_neural_m_tight_oheproj_seed{42,123,456}.parquet`
- `ieee_cis_neural_m_wide_oheproj_seed{42,123,456}.parquet`

- [ ] **Step 7: Commit run outputs**

```bash
git add notebooks/ieee_cis_ohe_projection_attack.ipynb \
        results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_results.csv \
        results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_summary.csv \
        results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_dose_response.png \
        results/adv_examples/ieee_ohe_projection/ieee_cis_neural_m_tight_oheproj_seed*.parquet \
        results/adv_examples/ieee_ohe_projection/ieee_cis_neural_m_wide_oheproj_seed*.parquet
git commit -m "$(cat <<'EOF'
run: IEEE-CIS mutable-set sensitivity sweep (M_tight + M_wide × 3 seeds)

Adds two more points on the capability axis:
- M_tight (~6 dims): feas-flip per seed = TBD
- M_wide  (~30 dims): feas-flip per seed = TBD

Together with the existing unconstrained / oheproj / m_oheproj points,
produces a 5-point dose-response curve for the ICAIF 2026 §5 capability-vs-
feasibility trade-off claim.
EOF
)"
```

(Replace TBD with actual numbers once Cell 21's summary table is in hand.)

---

## Task 6: Write findings — append "Mutable-set sensitivity sweep" section

**Files:**
- Modify: `docs/ieee_ohe_projection_findings.md` (append new ## section before "Caveats")

- [ ] **Step 1: Read the existing findings doc structure**

```bash
grep -n "^##" docs/ieee_ohe_projection_findings.md
```

- [ ] **Step 2: Append a new section after "Paper narrative implications" and before "Caveats"**

The section must contain:
1. **A 5-row results table** with columns: variant, n_mutable_dims, feasible-flipped (mean ± std), filtered success rate, robust PR-AUC.
2. **The shape interpretation** — is the curve monotonic-ascending (capability ⇒ more attacks), monotonic-descending (capability ⇒ tighter projection cost), or U-shaped (the canonical M is a local minimum because mid-cardinality features dilute the OHE projection's flip count without compensating with predictive signal)?
3. **The §5 paper sentence** — one sentence the paper can quote: e.g. "On IEEE-CIS, feasible-flipped attack count is non-monotonic in mutable-set size: it peaks at the wide-capability profile (~XX/seed) and the unconstrained-but-OHE-projected baseline (~120/seed), with the canonical M (TransactionAmt + ProductCD + addr + dist) acting as a local minimum at 7.7/seed."

Template (fill in numbers from Cell 21 output):

```markdown
## Mutable-set sensitivity sweep (capability axis dose-response)

**Date:** 2026-05-06.  **Notebook:** `ieee_cis_ohe_projection_attack.ipynb`
Cells 18–21.  **Commits:** TBD (scaffold) + TBD (run).

To bracket the M+OHE attack-count number (7.7 feasible-flipped per seed)
and turn "we hit one point" into "we mapped the curve", I ran two
additional capability profiles:

| Variant            | Raw mutable      | Processed dims | Feasible-flipped (3 seeds) | FSR     | Robust PR-AUC |
|--------------------|------------------|---------------:|---------------------------:|--------:|--------------:|
| `unconstrained`    | (all)            | ~380           | 0.06 ± 0.05                | 0.0%    | 0.082 ± 0.01  |
| `oheproj`          | (all, OHE-snap)  | ~380           | 120 ± 15                   | 54.5%   | 0.082 ± 0.01  |
| `m_tight_oheproj`  | TxnAmt + ProductCD | ~6           | TBD                        | TBD     | TBD           |
| `m_oheproj`        | + addr1/2 + dist1/2 | ~12         | 7.7 ± TBD                  | 100%    | 0.402 ± 0.046 |
| `m_wide_oheproj`   | + P/R_email + M1–M9 | ~30         | TBD                        | TBD     | TBD           |

[Insert dose-response figure here.]

### Shape

[Choose one and elaborate:]
- **Monotonic-ascending** (more capability → more feasible flips): the
  trade-off is purely about which features happen to be predictive.
- **U-shaped** (canonical M is a local minimum): mid-cardinality mutable
  features dilute the OHE projection without recovering predictive signal.
- **Plateau then jump**: addr/dist contribute little, but emaildomain +
  M-flags unlock attack space proportional to their predictive load.

### §5 paper sentence

"[fill in based on shape]"
```

- [ ] **Step 3: Commit**

```bash
git add docs/ieee_ohe_projection_findings.md
git commit -m "docs(ieee-cis): mutable-set sensitivity sweep results — dose-response curve"
```

---

## Task 7: Propagate to ToDo.md and Context.md

**Files:**
- Modify: `docs/ToDo.md` (mark §A' done)
- Modify: `docs/Context.md` §9 (add sweep result to roadmap status)

- [ ] **Step 1: Update `docs/ToDo.md`**

Replace the §A' block (lines 25–31) with:

```markdown
### A'. ~~Next decision — IEEE-CIS mutable-set sensitivity sweep~~ — **DONE (2026-05-06)**

`notebooks/ieee_cis_ohe_projection_attack.ipynb` Cells 18–21 (commits TBD).
Result documented in `ieee_ohe_projection_findings.md` "Mutable-set
sensitivity sweep" section.

- **Outcome:** [TBD — one-line summary of curve shape and §5 sentence]
```

- [ ] **Step 2: Update `docs/Context.md` §9**

Append a row to the "Findings docs (chronological)" table:

```markdown
| 2026-05-06 | IEEE-CIS mutable-set sensitivity sweep: 5-point dose-response curve along capability axis; [shape] | `ieee_cis_ohe_projection_attack.ipynb` Cells 18–21 | `ieee_ohe_projection_findings.md` "Mutable-set sensitivity sweep" |
```

And update the headline-numbers table for IEEE-CIS to include the bracketing:

```markdown
| **IEEE-CIS** | 0.00% | **100.0%** (M+OHE peak; tight=TBD%, wide=TBD%) | 0.014% | i_d_nonneg (closed by M-mask) |
```

- [ ] **Step 3: Commit**

```bash
git add docs/ToDo.md docs/Context.md
git commit -m "docs: mark §A' done; propagate sweep result to Context.md §9"
```

---

## Self-Review Checklist (run before declaring plan complete)

- [ ] Every code cell shown in full — no `# ... (rest unchanged)` placeholders.
- [ ] Cell 21's `n_mutable_dims` backfill handles legacy rows (unconstrained / oheproj have no `n_mutable_dims` column from Cell 16).
- [ ] Cell 20's skip guard `(attack, seed) in existing_keys` lets the cell be re-run safely.
- [ ] Verification commands have explicit expected outputs.
- [ ] Each task ends in a commit with a concrete message.
- [ ] Compute estimate (~15 min on Colab T4) matches the §A' "20 min compute" budget in ToDo.

---

## Notes for the executor

1. **Numbers in Tasks 6–7 are placeholders.** They become known only after Task 5 Step 5. Do not commit Tasks 6/7 with `TBD` left in the body — fill the actual numbers from Cell 21's printed summary first.
2. **Shape is a finding, not a foregone conclusion.** Pick the right §5 sentence based on what the curve actually shows. Three candidates are listed in Task 6 Step 2.
3. **If `m_tight feas-flip > m_oheproj feas-flip`** (tight capability flips MORE than canonical), that's a surprising result that re-opens the M+OHE central finding 2 — flag and stop. Either the canonical M is poorly chosen (some addr/dist features are anti-predictive) or there's a bug in the new mask construction.
4. **No new tests.** This is a research extension, not a behavior change. The "test" is the dose-response shape itself, evaluated visually + numerically against expectations.
