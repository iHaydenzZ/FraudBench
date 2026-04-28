# LCLD Seed-42 Sparse-Categorical OHE Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift LCLD seed-42 clean feasibility from 0.888 → ≥0.99 (and M1+g1 aggregate from 0.890 → ≥0.99) by fixing the test-only-category issue that produces all-zero OHE rows; cascade updated numbers into 3 findings docs + Context + ToDo.

**Architecture:** This is a small surgical fix in research-pipeline code, but the root cause is uncertain. The "obvious" fix proposed in 5 findings docs (`OneHotEncoder(handle_unknown="ignore")`) is **already applied** at `preprocessing/processor.py:38` — and is itself the cause: `handle_unknown="ignore"` produces all-zero OHE rows for unknown categories, which fails the OHE-sum validity check. Phase A runs a diagnostic to confirm the root cause and quantify it; Phase B applies the conditional fix (default path: `min_frequency` to absorb rare/unknown categories into a single bucket); Phase C reruns affected notebooks and cascades numbers.

**Tech Stack:** scikit-learn `OneHotEncoder` (sklearn ≥1.1 for `handle_unknown="infrequent_if_exist"`); `train_test_split` (currently stratifies on `y` only); pytest for regression test; Colab A100 for notebook reruns.

**Time estimate:** Phase A ~30 min (local). Phase B ~30 min code + test (local). Phase C ~3-4h Colab compute + ~1h doc cascade.

**Affected datasets / notebooks:**
- LCLD: primary fix target. Notebooks: `cross_dataset_feasibility.ipynb`, `g1_projection_attack.ipynb`, possibly `mask_ablation.ipynb` (verify in Phase A)
- IEEE-CIS: regression check only. Notebook: `ieee_cis_ohe_projection_attack.ipynb`
- Sparkov, CCFD: regression check via Phase C1 (cross-dataset notebook covers all 4)

---

## Pre-task context

**What we already know:**
- `preprocessing/processor.py:38` already has `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` → unknown categories produce all-zero rows by design (this is the bug, not the fix)
- `datasets/splitter.py:111,116` stratifies on `y` only → high-cardinality categoricals (`addr_state` 50 values, `purpose` ~14 values) can have rare values land entirely in test
- LCLD seed-42 affected: clean_feas = 0.888 vs ~0.991 on seeds 123/456 (`g1_projection_findings.md` Caveat #1)
- LCLD M1+g1 seed-42 aggregate = 0.890 vs ~0.993 on other seeds
- IEEE-CIS clean_feas = 0.9999 ± 0.0001 → presumably **not affected**, but verify since fix is global

**What changes about the headline numbers if Phase B succeeds:**
- LCLD M1+g1 FSR: 95.3% → ~99% (driven by seed-42 climbing from 0.890 to ~0.99)
- LCLD g1-projected FSR: 50.2% → likely ~50-55% (smaller change since g1-only doesn't depend on M-mask freezes)
- IEEE-CIS OHE-projected FSR: should stay 59.7% (regression check)

**Why fix now (not after M+OHE):**
- Cascade is cheaper now: only LCLD findings docs reference the affected numbers
- After M+OHE, the headline cross-dataset table will mix pre-fix LCLD with post-fix IEEE-CIS — bad form for a paper
- Same `min_frequency` setting is global; if IEEE-CIS rows shift, easier to catch with smaller blast radius

---

## Phase A: Diagnose root cause (~30 min, local)

### Task A1: Quantify all-zero OHE rows on LCLD per seed × per categorical column

**Files:**
- Investigate: `preprocessing/processor.py`, `datasets/splitter.py`
- Output: paste diagnostic numbers into Phase A "Diagnostic results" section of this plan (added at execution time)

- [ ] **Step A1.1: Run a one-off diagnostic script**

Create temp file `scratch/diagnose_seed42.py`:

```python
"""Diagnose all-zero OHE rows in LCLD splits per seed × per categorical column."""
from datasets.loader import load_dataset
from datasets.splitter import split_dataset
from preprocessing.processor import DataPreprocessor

print(f"{'seed':<6}{'col':<25}{'all_zero_count':<18}{'all_zero_pct':<14}")
for seed in [42, 123, 456]:
    ds = load_dataset("lcld", sample_frac=0.1)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(ds, random_state=seed)
    pp = DataPreprocessor(ds.feature_types).fit(X_train)
    X_test_p = pp.transform(X_test)
    for cat_col, ftype in ds.feature_types.items():
        if ftype not in ("categorical", "binary"):
            continue
        ohe_cols = [c for c in X_test_p.columns if c.startswith(cat_col + "_")]
        if not ohe_cols:
            continue
        block = X_test_p[ohe_cols]
        all_zero_count = int((block.sum(axis=1) == 0).sum())
        all_zero_pct = all_zero_count / len(block) * 100
        if all_zero_count > 0:
            print(f"{seed:<6}{cat_col:<25}{all_zero_count:<18}{all_zero_pct:<14.2f}")
```

Run: `uv run python scratch/diagnose_seed42.py`
Note: if `uv` fails on macOS ARM (CUDA torch), use `.venv/bin/python scratch/diagnose_seed42.py` per CLAUDE.md.

Expected: at least one column on seed=42 shows ≥1% all-zero rows in test set; seeds 123/456 show <0.1%.

- [ ] **Step A1.2: For each offending column, identify the specific test-only categories**

Append to `scratch/diagnose_seed42.py`:

```python
print("\n--- Test-only categories per offending column ---")
for seed in [42, 123, 456]:
    ds = load_dataset("lcld", sample_frac=0.1)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(ds, random_state=seed)
    for cat_col in ds.feature_types:
        if ds.feature_types[cat_col] not in ("categorical", "binary"):
            continue
        train_cats = set(X_train[cat_col].dropna().unique())
        test_cats = set(X_test[cat_col].dropna().unique())
        test_only = test_cats - train_cats
        if test_only:
            print(f"seed={seed}, col={cat_col}: test-only = {test_only}")
            for cat in test_only:
                n_test = (X_test[cat_col] == cat).sum()
                n_train_total = (X_train[cat_col] == cat).sum()
                print(f"  '{cat}': {n_test} in test, {n_train_total} in train")
```

Run: `.venv/bin/python scratch/diagnose_seed42.py`
Expected output format: `seed=42, col=addr_state: test-only = {'IA', 'ME'}` with row counts.

- [ ] **Step A1.3: For the offending column(s), check minimum train-frequency of all categories**

Append:

```python
print("\n--- Train-frequency distribution for offending columns ---")
for cat_col in OFFENDING_COLS_FROM_A1_2:  # replace with actual list
    ds = load_dataset("lcld", sample_frac=0.1)
    X_train, _, _, _, _, _ = split_dataset(ds, random_state=42)
    counts = X_train[cat_col].value_counts()
    print(f"\n{cat_col} train counts (bottom 10):")
    print(counts.tail(10))
    print(f"  min={counts.min()}, count<10={int((counts < 10).sum())}, count<5={int((counts < 5).sum())}")
```

Expected: bottom-10 categories have counts in single digits or low double digits. This determines the `min_frequency` threshold for Phase B.

- [ ] **Step A1.4: Record diagnostic findings as a new section in this plan**

Edit this plan, after the Phase A header, add:

```markdown
### Phase A diagnostic results (filled at execution time)

**Offending column(s):** [e.g., addr_state, purpose]
**Seed-42 all-zero count:** [e.g., addr_state: 268 rows / 2.4%]
**Test-only categories:** [e.g., addr_state: IA (12 rows), ME (8 rows)]
**Bottom train-frequency:** [e.g., addr_state min count = 3]
**Recommended min_frequency:** [e.g., 10 — absorbs all categories with <10 train occurrences plus all unknowns]
**Phase B path selected:** [B-α / B-β / B-γ / B-δ]
```

- [ ] **Step A1.5: Commit the diagnostic script + plan update**

```bash
git add scratch/diagnose_seed42.py docs/plans/2026-04-28-lcld-seed42-ohe-fix.md
git commit -m "diag(seed42): quantify LCLD all-zero OHE rows root cause"
```

### Task A2: Verify IEEE-CIS is unaffected (microadjustment per Plan-3 ordering)

**Files:**
- Investigate: same script extended to IEEE-CIS

- [ ] **Step A2.1: Repeat diagnostic on IEEE-CIS for the 3 binding OHE blocks**

Append to `scratch/diagnose_seed42.py`:

```python
print("\n--- IEEE-CIS regression check ---")
for seed in [42, 123, 456]:
    ds = load_dataset("ieee_cis", sample_frac=0.1)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(ds, random_state=seed)
    pp = DataPreprocessor(ds.feature_types).fit(X_train)
    X_test_p = pp.transform(X_test)
    for cat_col in ["ProductCD", "card4", "card6"]:
        ohe_cols = [c for c in X_test_p.columns if c.startswith(cat_col + "_")]
        if not ohe_cols:
            continue
        block = X_test_p[ohe_cols]
        all_zero_pct = (block.sum(axis=1) == 0).mean() * 100
        print(f"seed={seed}, col={cat_col}: {all_zero_pct:.4f}% all-zero")
```

Run: `.venv/bin/python scratch/diagnose_seed42.py`
Expected: <0.01% all-zero on every seed × column. Confirms IEEE-CIS doesn't independently need the fix; the global change in Phase B is a no-op for IEEE-CIS.

- [ ] **Step A2.2: Decision point — choose Phase B path**

Based on A1 + A2 results, mark Phase B path in the diagnostic results section. Default expected:

- **Path B-α (default):** Test-only categories are rare → use `OneHotEncoder(min_frequency=N, handle_unknown="infrequent_if_exist")`. Proceed to Task B1 below.
- **Path B-β:** Test-only categories are common → stratify split on `y + offending_column_bucketed`. Replace Task B1 with split-stratification work in `datasets/splitter.py`.
- **Path B-γ:** OHE-sum check is semantically wrong for `handle_unknown="ignore"` outputs → modify validator to allow sum=0 as "missing". Replace Task B1 with validator changes in `constraints/validator.py`.
- **Path B-δ:** Mix — apply B-α with conservative `min_frequency` and accept residual all-zero rate. Same Task B1 with smaller `min_frequency`.

If A2 shows IEEE-CIS would regress under B-α (>1pp shift), switch to B-γ (which is dataset-agnostic).

---

## Phase B: Apply the fix (default path B-α; ~30 min, local)

> **Note:** Tasks below are written for the default path (B-α: rare-value `min_frequency`). If Task A2.2 selected a different path, replace this section with the appropriate alternative before executing. Most likely path is B-α with `min_frequency=10`.

### Task B1: Add `min_frequency` to OneHotEncoder

**Files:**
- Modify: `preprocessing/processor.py:38`
- Test: `tests/test_dataset.py` (add new test)

- [ ] **Step B1.1: Write a failing regression test**

Add to `tests/test_dataset.py`:

```python
import pytest
from datasets.loader import load_dataset
from datasets.splitter import split_dataset
from preprocessing.processor import DataPreprocessor

def _lcld_available():
    try:
        load_dataset("lcld", sample_frac=0.01)
        return True
    except (FileNotFoundError, OSError):
        return False

@pytest.mark.skipif(not _lcld_available(), reason="LCLD data not available")
def test_lcld_seed42_no_all_zero_ohe_rows():
    """Regression: seed-42 split must not produce >1% all-zero OHE rows for any categorical."""
    ds = load_dataset("lcld", sample_frac=0.1)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(ds, random_state=42)
    pp = DataPreprocessor(ds.feature_types).fit(X_train)
    X_test_p = pp.transform(X_test)
    offenders = []
    for cat_col, ftype in ds.feature_types.items():
        if ftype not in ("categorical", "binary"):
            continue
        ohe_cols = [c for c in X_test_p.columns if c.startswith(cat_col + "_")]
        if not ohe_cols:
            continue
        block = X_test_p[ohe_cols]
        all_zero_pct = (block.sum(axis=1) == 0).mean()
        if all_zero_pct >= 0.01:
            offenders.append((cat_col, all_zero_pct))
    assert not offenders, f"Columns with ≥1% all-zero OHE rows in seed-42 test set: {offenders}"
```

- [ ] **Step B1.2: Run test to verify it fails (pre-fix)**

Run: `.venv/bin/pytest tests/test_dataset.py::test_lcld_seed42_no_all_zero_ohe_rows -v`
Expected: FAIL with `Columns with ≥1% all-zero OHE rows in seed-42 test set: [('addr_state', 0.024...)]` (or whichever column A1 identified).

- [ ] **Step B1.3: Apply the fix in `preprocessing/processor.py:38`**

Replace line 38 exactly:

```python
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
```

With:

```python
                ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False)),
```

(Use the `min_frequency` value chosen in A1.4 — `10` is the default suggested unless A1.3 shows otherwise.)

`handle_unknown="infrequent_if_exist"` requires sklearn ≥1.1 — check `pyproject.toml` if uncertain. The combination routes both rare-train and unknown-test categories to a single `*_infrequent_sklearn` column, eliminating all-zero rows.

- [ ] **Step B1.4: Run test to verify it passes (post-fix)**

Run: `.venv/bin/pytest tests/test_dataset.py::test_lcld_seed42_no_all_zero_ohe_rows -v`
Expected: PASS.

- [ ] **Step B1.5: Run full preprocessing-related test suite to check nothing else broke**

Run: `.venv/bin/pytest tests/test_dataset.py tests/test_constraints.py -v -m "not slow"`
Expected: ALL PASS. If any tests reference specific OHE column counts or names, they may need updates — fix per failure.

- [ ] **Step B1.6: Invalidate preprocessor cache (delete stale joblib files)**

```bash
rm -f results/preprocessor_lcld_*.joblib
rm -f results/preprocessor_ieee_cis_*.joblib
rm -f results/preprocessor_sparkov_*.joblib
rm -f results/preprocessor_ccfd_*.joblib
```

The `min_frequency` change adds a new `*_infrequent_sklearn` output column for some categoricals — cached preprocessors won't have it.

- [ ] **Step B1.7: Commit the fix**

```bash
git add preprocessing/processor.py tests/test_dataset.py
git commit -m "fix(ohe): use min_frequency to absorb test-only categories

LCLD seed-42 had ~Y% all-zero OHE rows in addr_state due to test-only
categories. handle_unknown='ignore' was producing them by design, which
failed the OHE-sum validity check. Switching to
handle_unknown='infrequent_if_exist' + min_frequency=10 routes both
rare and unknown categories to a single bucket."
```

(Replace `Y%` with the actual percentage from A1.1.)

---

## Phase C: Rerun affected experiments + cascade docs (~3-4h Colab + ~1h docs)

### Task C1: Rerun `cross_dataset_feasibility.ipynb` on Colab A100

**Files:**
- Run: `notebooks/cross_dataset_feasibility.ipynb`
- Output: `results/adv_examples/cross_dataset_feasibility/cross_dataset_feasibility_results.csv` (overwrite)
- Output: `results/adv_examples/cross_dataset_feasibility/cross_dataset_per_constraint.csv` (Cell 14, overwrite)

- [ ] **Step C1.1: Pull latest commit on Colab + restart runtime**

In Colab notebook, run setup cell that does `git pull` + reinstalls deps.

- [ ] **Step C1.2: Execute all cells with seeds [42, 123, 456] × 4 datasets**

Expected runtime: ~1.5h on A100 (4 datasets × 3 seeds).

- [ ] **Step C1.3: Verify seed-42 LCLD clean_feas now ≥0.99**

In Cell 14 output, check the LCLD row mean ± std:
- Old: `0.9564 ± 0.0594`
- New target: `≥0.99X ± ≤0.005`

If the new value is still <0.99, the Phase B fix didn't fully resolve it — return to Phase A and consider B-β (stratify) or B-γ (validator change). Do NOT proceed to C2.

- [ ] **Step C1.4: Verify per-constraint pass rates haven't shifted on the unaffected datasets**

Compare Cell 14 per-constraint output to the old values in `cross_dataset_feasibility_findings.md`:
- IEEE-CIS: `i_product_ohe`, `i_card4_ohe`, `i_card6_ohe`, `i_d_nonneg`, `i_amt_positive`, `i_c_nonneg` — all should be within ±0.5pp of pre-fix
- Sparkov: `s_state_ohe`, `s_category_ohe`, `s_gender_ohe`, `s_merch_bbox`, `s_city_pop_pos`, `s_amt_positive` — all should be within ±0.5pp
- CCFD: trivially unchanged (no categoricals)

If any non-LCLD value shifted >0.5pp, investigate before proceeding.

- [ ] **Step C1.5: Commit the updated CSVs**

```bash
git add results/adv_examples/cross_dataset_feasibility/
git commit -m "run(cross_dataset): rerun after OHE min_frequency fix"
```

### Task C2: Rerun `g1_projection_attack.ipynb` on Colab A100

**Files:**
- Run: `notebooks/g1_projection_attack.ipynb`
- Output: `results/adv_examples/g1_projection/g1_projection_results.csv` (overwrite)
- Output: `results/adv_examples/g1_projection/g1_projection_summary.csv` (overwrite)

- [ ] **Step C2.1: Restart runtime, run all cells**

Expected runtime: ~30-45 min on A100 (3 seeds × 3 attack regimes).

- [ ] **Step C2.2: Verify seed-42 M1+g1 aggregate now ≥0.99**

Check the per-seed table:
- Old: seed-42 M1+g1 aggregate = `0.8897`, g3 pass = `0.890`
- New target: aggregate ≥ 0.99, g3 pass ≥ 0.99

- [ ] **Step C2.3: Record the new headline 3-seed mean ± std for M1+g1 FSR**

Old: `95.3% ± [unstated, driven down by seed-42's 0.890]`
New target: ~98-99% with tight std (≤0.5pp).

Save the new mean + std somewhere visible (notebook output cell + or a markdown cell at top of notebook) — Phase C5 will reference these numbers.

- [ ] **Step C2.4: Commit**

```bash
git add results/adv_examples/g1_projection/
git commit -m "run(g1_projection): rerun after OHE min_frequency fix; M1+g1 FSR ~99%"
```

### Task C3: Rerun `ieee_cis_ohe_projection_attack.ipynb` on Colab A100 (regression check)

**Files:**
- Run: `notebooks/ieee_cis_ohe_projection_attack.ipynb`
- Output: `results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_results.csv` (regression check; should be unchanged)

- [ ] **Step C3.1: Restart runtime, run all cells**

Expected runtime: ~30 min on A100 (3 seeds × 2 attacks).

- [ ] **Step C3.2: Verify FSR mean ± std stays at 59.7% (within ±2pp tolerance)**

If FSR moved by >2pp, the `min_frequency=10` change had unintended IEEE-CIS effects (likely on `M1`–`M9` match-flag columns or low-cardinality `card4`/`card6`). Diagnose and either:
- Lower `min_frequency` to 5 or 3
- Switch to per-dataset preprocessor config (more invasive, defer)

- [ ] **Step C3.3: Commit only if numbers changed materially (>0.5pp shift)**

```bash
# Conditional — only if numbers shifted
git add results/adv_examples/ieee_ohe_projection/
git commit -m "run(ieee_ohe_projection): regression rerun post-OHE fix; FSR=Y%"
```

If numbers are stable, no commit needed for this task.

### Task C4: Update `cross_dataset_feasibility_findings.md`

**Files:**
- Modify: `docs/cross_dataset_feasibility_findings.md`

- [ ] **Step C4.1: Update §"Headline numbers" LCLD row**

Find the table:
```
| **lcld**     | 0.9564 ± 0.0594 | **0.000932 ± 0.000393** | 0.3020 | 0.1051 ± 0.0000 |
```

Replace 0.9564 ± 0.0594 with the actual post-fix value from C1.3. Adv feas may also shift slightly — use C1.5 actual numbers.

- [ ] **Step C4.2: Resolve §"Open flags" #1**

Find the section starting with:
```
1. **LCLD clean feasibility is unexpectedly low: 0.9564 ± 0.0594**
```

Replace with:
```
1. ~~**LCLD clean feasibility is unexpectedly low: 0.9564 ± 0.0594**~~ **CLOSED 2026-04-28** by `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)` in `preprocessing/processor.py:38`. Root cause: `addr_state` had test-only categories on seed-42 (specifically: [list from A1.2]); `handle_unknown="ignore"` produced all-zero OHE rows that failed the OHE-sum validity check. New clean_feas: 0.99X ± 0.00Y across 3 seeds. See `docs/plans/2026-04-28-lcld-seed42-ohe-fix.md`.
```

- [ ] **Step C4.3: Update §"Headline finding" or any other passages that quote the 0.9564 number**

Search the file for "0.9564" / "0.0594" and update each instance. Likely 1-2 mentions outside the table.

- [ ] **Step C4.4: Commit**

```bash
git add docs/cross_dataset_feasibility_findings.md
git commit -m "docs(cross_dataset): refresh LCLD numbers post-OHE fix; close open flag #1"
```

### Task C5: Update `g1_projection_findings.md`

**Files:**
- Modify: `docs/g1_projection_findings.md`

- [ ] **Step C5.1: Update headline mean ± std table (§"Headline numbers")**

Find the M1+g1 row:
```
| **M1 + g1**    | 2912 ± 191   | 2774 ± 314    | **95.3%**        | 0.958 ± 0.059    | 1.000 | 0.962 ± 0.062 | 0.175 ± 0.022 | 0.10508 ± 9×10⁻⁶ |
```

Replace with C2.3 actual numbers. Likely changes: filtered success ~98-99%, agg feas ~0.99, g3 pass ~0.99 (std much tighter).

- [ ] **Step C5.2: Update per-seed table (seed=42 M1+g1 row)**

Find:
```
| 42   | **M1 + g1**    | 2830    | 2473          | 0.8897     | 1.000 | 1.000 | 0.890 | 1.000 | 0.1830      | 0.10507 |
```

Replace with C2.2 actual numbers (likely agg ≥0.99, g3 ≥0.99).

- [ ] **Step C5.3: Update §"Adding M1 closes the credit-bureau gap"**

Find the bullet `- **Filtered success rate:** 50.2% → **95.3%** (×1.9)`. Update both numbers.
Find the paragraph `What's left of the gap (95.3% → 100%) is essentially the seed-42 sparse-categorical issue...`. Replace with: `Resolved 2026-04-28 by OHE min_frequency fix; M1+g1 now sits at ~99% on all 3 seeds. See ` + plan link.

Find the line `The 0.05% → 50% → 95% filtered-success progression is the single cleanest experimental story...`. Update 95% → new value.

- [ ] **Step C5.4: Update Caveat #1 (or whichever caveat references seed-42)**

Mark as resolved with cross-reference to plan doc.

- [ ] **Step C5.5: Commit**

```bash
git add docs/g1_projection_findings.md
git commit -m "docs(g1_projection): refresh M1+g1 numbers post-OHE fix"
```

### Task C6: Update `constraint_evaluation_guidance.md`

**Files:**
- Modify: `docs/constraint_evaluation_guidance.md`

- [ ] **Step C6.1: Update §1 result #1 headline numbers**

Find:
```
Adding the M1 mutability mask raises it further to **95.3%** — 2774 feasible-and-flipped attacks per seed vs 1 under stock CAPGD, a **~2000× underestimate** of realistic attacker success.
```

Update 95.3% → C2.3 value, 2774 → new feas-flipped count, recompute the multiplicative factor.

- [ ] **Step C6.2: Update §4 last note**

Find:
```
On LCLD, replacing stock CAPGD with g1-projected CAPGD raises adv feasibility from 0.093% to 69.3% at the same ε, and adding the M1 mutability mask raises it to 95.8%
```

Update 69.3% → new value (if g1-projected adv feas changed), 95.8% → new value.

- [ ] **Step C6.3: Update §5 Soft blockers — mark item #1 as RESOLVED**

Find:
```
1. **Seed-42 sparse-categorical issue on LCLD** ...
```

Replace with:
```
1. ~~**Seed-42 sparse-categorical issue on LCLD**~~ **RESOLVED 2026-04-28** by `OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist")`. See `docs/plans/2026-04-28-lcld-seed42-ohe-fix.md`. LCLD M1+g1 now ~99% on all seeds.
```

- [ ] **Step C6.4: Update §6 Caveat 2 — update the LCLD progression number**

Find:
```
LCLD has via M1+g1 (FSR = 95.3%)
```

Update 95.3% → new value.

- [ ] **Step C6.5: Add §8 Decision log entry**

Append to the Decision log table:
```
| OHE min_frequency=10 to absorb test-only categories | LCLD seed-42 had Y% all-zero OHE rows in `addr_state` due to rare values landing only in test split. `handle_unknown="ignore"` produced them by design, failing the OHE-sum validity check. Setting `min_frequency=10` + `handle_unknown="infrequent_if_exist"` routes both rare and unknown categories to a single bucket. Lifted seed-42 clean_feas 0.888→0.99X and M1+g1 0.890→0.99X. See `docs/plans/2026-04-28-lcld-seed42-ohe-fix.md`. | 2026-04-28 |
```

(Replace Y% and 0.99X with actual numbers from Phase A and Phase C.)

- [ ] **Step C6.6: Commit**

```bash
git add docs/constraint_evaluation_guidance.md
git commit -m "docs(guidance): cascade post-OHE-fix numbers; close soft blocker #1"
```

### Task C7: Update `Context.md` §9 + `ToDo.md` ICAIF section

**Files:**
- Modify: `docs/Context.md`
- Modify: `docs/ToDo.md`

- [ ] **Step C7.1: Update Context.md §9 headline numbers table — LCLD row**

Find:
```
| **LCLD** | 0.05% | **95.3%** (M1+g1) | 0.093% | g3 (closed by M1) |
```

Update 95.3% → new value, possibly update 0.093% if cross-dataset adv feas shifted.

- [ ] **Step C7.2: Update Context.md §9 outstanding soft blockers — strike #1**

Find:
```
1. **LCLD seed-42 sparse-categorical issue** ...
```

Replace with:
```
1. ~~**LCLD seed-42 sparse-categorical issue**~~ ✅ **RESOLVED 2026-04-28** — see `docs/plans/2026-04-28-lcld-seed42-ohe-fix.md`.
```

- [ ] **Step C7.3: Update ToDo.md ICAIF §B — mark first item DONE**

Find:
```
- [ ] LCLD seed-42 sparse-categorical fix (`OneHotEncoder(handle_unknown="ignore")` or stratify split). <1 day.
```

Replace with:
```
- [x] LCLD seed-42 sparse-categorical fix — **DONE 2026-04-28** via `min_frequency=10`. See `docs/plans/2026-04-28-lcld-seed42-ohe-fix.md`.
```

- [ ] **Step C7.4: Commit**

```bash
git add docs/Context.md docs/ToDo.md
git commit -m "docs: cascade OHE-fix numbers into Context + ToDo"
```

### Task C8: Clean up scratch files

- [ ] **Step C8.1: Decide whether to keep `scratch/diagnose_seed42.py`**

Either:
- Promote to `scripts/diagnose_ohe_validity.py` (general-purpose diagnostic for future datasets), then commit
- Delete and rely on the `tests/test_dataset.py::test_lcld_seed42_no_all_zero_ohe_rows` regression test

Recommendation: promote, since the same diagnostic will be useful for IEEE-CIS / Sparkov / future datasets.

- [ ] **Step C8.2: Commit**

```bash
git add scripts/diagnose_ohe_validity.py  # if promoted
git rm scratch/diagnose_seed42.py  # always
git commit -m "chore(scripts): promote OHE all-zero diagnostic to scripts/"
```

---

## Verification checklist (run before declaring complete)

- [ ] `.venv/bin/pytest tests/test_dataset.py tests/test_constraints.py -v` all pass
- [ ] LCLD seed-42 clean_feas ≥ 0.99 in current `cross_dataset_feasibility_results.csv`
- [ ] LCLD seed-42 M1+g1 aggregate ≥ 0.99 in current `g1_projection_results.csv`
- [ ] IEEE-CIS FSR mean ± std unchanged from pre-fix value (within 2pp; regression check via Task C3)
- [ ] Sparkov + CCFD per-constraint pass rates unchanged from pre-fix (regression check via Task C1.4)
- [ ] `cross_dataset_feasibility_findings.md` Open flag #1 marked CLOSED with date + plan link
- [ ] `g1_projection_findings.md` Caveat #1 marked RESOLVED
- [ ] `constraint_evaluation_guidance.md` Soft blocker #1 marked RESOLVED + Decision log entry added
- [ ] `Context.md` §9 LCLD headline FSR + soft blocker section updated
- [ ] `ToDo.md` ICAIF §B item marked DONE

---

## Rollback plan (if Phase B causes more harm than good)

If C3 shows IEEE-CIS regression >2pp OR C1.4 shows Sparkov / CCFD regression OR C1.3 shows LCLD seed-42 still <0.99:

1. Revert `preprocessing/processor.py:38` to pre-fix (`handle_unknown="ignore"` only)
2. Delete cached preprocessors again
3. Switch to Path B-γ (modify validator to allow sum=0 as "missing"): edit `constraints/validator.py` and any per-constraint OHE-sum check in the notebooks to treat sum=0 as a separate category rather than a violation
4. Rerun Phase C from C1

The regression test from B1.1 should be updated to allow `min_frequency=0` (no fix) but require validator-level handling instead.
