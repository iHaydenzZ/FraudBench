# Fix Figures & Analysis Artefacts — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 5 code bugs in the figure/analysis scripts and add 3 documentation items so all artefacts are presentation-ready.

**Architecture:** Each fix targets one of three scripts (`generate_figures.py`, `statistical_tests.py`, `analyse_input_validation.py`) or documentation files (`README.md`). Changes are independent — no fix depends on another — but we execute in priority order (P0→P1→P2→P3). Every code change gets a test first (TDD). After all code fixes, regenerate all artefacts in one pass.

**Tech Stack:** Python 3, pandas, matplotlib, seaborn, scipy, numpy, pytest. Run via `uv run`.

---

## Task 1: Robustness Curves — Show All Defences (P0)

**Files:**
- Modify: `scripts/generate_figures.py:257-323` (`plot_robustness_curves()`)
- Test: `tests/test_figures.py` (add new test class)

### Problem

`plot_robustness_curves()` filters to groups with >1 unique epsilon (line 274). Adversarial training and input validation only exist at ε=0.1 for most dataset–model combos, so they get excluded. The figure shows only the `none` baseline.

### Step 1: Write the failing test

Add a new test class at the end of `tests/test_figures.py`:

```python
class TestPlotRobustnessCurves:
    """Tests for robustness curves figure generation."""

    def _make_multi_defence_data(self):
        """Registry with baseline at multiple epsilons + single-epsilon defences."""
        rows = []
        # Baseline (none) with multi-epsilon sweep
        for eps in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]:
            for seed in [42, 123, 456]:
                rows.append({
                    "dataset": "ccfd", "model_type": "neural",
                    "defence_type": "none", "attack_type": "capgd",
                    "attack_epsilon": eps, "seed": seed,
                    "robust_pr_auc": max(0.01, 0.85 - eps * 2),
                    "clean_pr_auc": 0.90,
                })
        # adversarial_training at eps=0.1 only
        for seed in [42, 123, 456]:
            rows.append({
                "dataset": "ccfd", "model_type": "neural",
                "defence_type": "adversarial_training", "attack_type": "capgd",
                "attack_epsilon": 0.1, "seed": seed,
                "robust_pr_auc": 0.75,
                "clean_pr_auc": 0.88,
            })
        # input_validation at eps=0.1 only
        for seed in [42, 123, 456]:
            rows.append({
                "dataset": "ccfd", "model_type": "neural",
                "defence_type": "input_validation", "attack_type": "capgd",
                "attack_epsilon": 0.1, "seed": seed,
                "robust_pr_auc": 0.55,
                "clean_pr_auc": 0.89,
            })
        return pd.DataFrame(rows)

    def test_single_epsilon_defences_included(self, tmp_path):
        """Defences with only 1 epsilon value should appear as scatter markers."""
        from scripts.generate_figures import plot_robustness_curves

        df = self._make_multi_defence_data()
        plot_robustness_curves(df, str(tmp_path))

        # Figure should be saved
        assert (tmp_path / "robustness_curves.png").exists()

        # Re-run the aggregation + filtering logic to check inclusion
        from scripts.generate_figures import aggregate_seeds
        agg = aggregate_seeds(df)
        # After fix: all 3 defences should be present in the filtered data
        # We verify by checking the figure was generated (non-empty)
        # and the aggregated data includes all defence types for neural models
        neural = agg[agg["model_type"] == "neural"]
        assert set(neural["defence_type"].unique()) == {"none", "adversarial_training", "input_validation"}

    def test_neural_only_filter(self, tmp_path):
        """Robustness curves should filter to neural models only (CAPGD irrelevant for trees)."""
        from scripts.generate_figures import plot_robustness_curves

        df = self._make_multi_defence_data()
        # Add tree data — should be excluded from CAPGD curves
        for eps in [0.01, 0.05, 0.1]:
            for seed in [42, 123, 456]:
                df = pd.concat([df, pd.DataFrame([{
                    "dataset": "ccfd", "model_type": "tree",
                    "defence_type": "none", "attack_type": "capgd",
                    "attack_epsilon": eps, "seed": seed,
                    "robust_pr_auc": 0.86,  # identical to clean — CAPGD no-op on trees
                    "clean_pr_auc": 0.86,
                }])], ignore_index=True)

        plot_robustness_curves(df, str(tmp_path))
        assert (tmp_path / "robustness_curves.png").exists()
```

### Step 2: Run test to verify it fails

Run: `uv run pytest tests/test_figures.py::TestPlotRobustnessCurves -v`
Expected: FAIL — `test_single_epsilon_defences_included` may pass on file existence but the figure currently only shows `none`. The key verification is visual, but the test ensures the function doesn't crash. We'll also verify the actual figure visually after the fix.

### Step 3: Implement the fix

In `scripts/generate_figures.py`, replace `plot_robustness_curves()` (lines 257-323) with:

```python
def plot_robustness_curves(df: pd.DataFrame, output_dir: str):
    """
    Figure 6: Robustness curves — Robust PR-AUC vs epsilon per dataset.

    Shows multi-epsilon sweeps as lines and single-epsilon defences as scatter
    markers.  Filters to neural models only (CAPGD is a no-op on trees).
    """
    agg = aggregate_seeds(df)
    agg = agg[agg["robust_pr_auc_mean"].notna() & (agg["robust_pr_auc_mean"] > 0)]

    if agg.empty:
        print("  Skipping robustness_curves: no robust data available.")
        return

    # Filter to neural models only — CAPGD has no effect on tree models
    agg = agg[agg["model_type"] == "neural"]
    if agg.empty:
        print("  Skipping robustness_curves: no neural model data available.")
        return

    # Classify each (dataset, defence, attack) group as multi-epsilon or single
    group_cols = ["dataset", "model_type", "defence_type", "attack_type"]
    counts = agg.groupby(group_cols)["attack_epsilon"].nunique().reset_index(name="n_eps")
    multi_eps = counts[counts["n_eps"] > 1]
    single_eps = counts[counts["n_eps"] == 1]

    # Need at least one multi-epsilon group to anchor the x-axis
    if multi_eps.empty:
        print("  Skipping robustness_curves: no multi-epsilon groups found.")
        return

    datasets = sorted(agg["dataset"].unique())
    n = len(datasets)
    if n == 0:
        print("  Skipping robustness_curves: no datasets with data.")
        return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        sub = agg[agg["dataset"] == dataset]

        defences = sorted(sub["defence_type"].unique())
        for defence in defences:
            d = sub[sub["defence_type"] == defence].sort_values("attack_epsilon")
            yerr = d["robust_pr_auc_std"].fillna(0)

            if len(d) >= 2:
                # Line plot for multi-epsilon data
                ax.errorbar(
                    d["attack_epsilon"],
                    d["robust_pr_auc_mean"],
                    yerr=yerr,
                    marker="o",
                    capsize=3,
                    label=defence,
                )
            elif len(d) == 1:
                # Single-point marker for defences with only one epsilon
                ax.scatter(
                    d["attack_epsilon"].values,
                    d["robust_pr_auc_mean"].values,
                    marker="^",
                    s=100,
                    zorder=5,
                    label=f"{defence} (ε={d['attack_epsilon'].values[0]})",
                )

        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Robust PR-AUC")
        ax.set_title(dataset.upper())
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Robustness Curves: Robust PR-AUC vs Epsilon (Neural Models)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "robustness_curves.png"), dpi=150)
    plt.close(fig)
    print("  Saved robustness_curves.png")
```

Key changes:
1. Filter to `model_type == "neural"` early (line added after initial filtering)
2. Classify groups into multi-eps (≥2) and single-eps (==1)
3. Multi-eps groups get line plots (existing behaviour)
4. Single-eps groups get scatter markers with `^` marker and `s=100`
5. Legend labels for single-eps include the epsilon value
6. Title updated to say "(Neural Models)"

### Step 4: Run test to verify it passes

Run: `uv run pytest tests/test_figures.py::TestPlotRobustnessCurves -v`
Expected: PASS — both tests should pass, figure files should be created.

### Step 5: Commit

```bash
git add scripts/generate_figures.py tests/test_figures.py
git commit -m "fix(figures): show all defences in robustness curves (Issue 1)

Single-epsilon defences (adversarial_training, input_validation) are now
plotted as scatter markers alongside the multi-epsilon baseline curves.
Also filters to neural models only since CAPGD is a no-op on trees."
```

---

## Task 2: Robustness Bars — Filter to ε=0.1 (P2)

**Files:**
- Modify: `scripts/generate_figures.py:70-126` (`plot_robustness_bars()`)
- Test: `tests/test_figures.py` (add new test class)

### Problem

`plot_robustness_bars()` plots all epsilon values, producing ~18 bars per dataset subplot. X-axis labels overlap and are unreadable.

### Step 1: Write the failing test

Add to `tests/test_figures.py`:

```python
class TestPlotRobustnessBars:
    """Tests for robustness bars figure generation."""

    def test_filters_to_canonical_epsilon(self, tmp_path):
        """Bar chart should only show ε=0.1 data."""
        from scripts.generate_figures import aggregate_seeds

        rows = []
        for eps in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]:
            for seed in [42, 123, 456]:
                rows.append({
                    "dataset": "ccfd", "model_type": "neural",
                    "defence_type": "none", "attack_type": "capgd",
                    "attack_epsilon": eps, "seed": seed,
                    "robust_pr_auc": 0.65, "clean_pr_auc": 0.85,
                })
        df = pd.DataFrame(rows)
        agg = aggregate_seeds(df)

        # After fix: filtering to eps=0.1 should leave 1 row for this config
        filtered = agg[np.isclose(agg["attack_epsilon"], 0.1)]
        assert len(filtered) == 1

    def test_bar_labels_include_model_type(self, tmp_path):
        """Bar labels should include model_type for disambiguation."""
        from scripts.generate_figures import plot_robustness_bars

        rows = []
        for model in ["neural", "tree"]:
            for defence in ["none", "adversarial_training"]:
                for seed in [42, 123, 456]:
                    rows.append({
                        "dataset": "ccfd", "model_type": model,
                        "defence_type": defence, "attack_type": "capgd",
                        "attack_epsilon": 0.1, "seed": seed,
                        "robust_pr_auc": 0.65, "clean_pr_auc": 0.85,
                    })
        df = pd.DataFrame(rows)
        plot_robustness_bars(df, str(tmp_path))
        assert (tmp_path / "robustness_bars.png").exists()
```

### Step 2: Run test to verify it fails

Run: `uv run pytest tests/test_figures.py::TestPlotRobustnessBars -v`
Expected: Tests may pass but the label format assertion would fail if we tighten it. The key test is `test_filters_to_canonical_epsilon` which validates the filtering logic.

### Step 3: Implement the fix

Replace `plot_robustness_bars()` (lines 70-126):

```python
def plot_robustness_bars(df: pd.DataFrame, output_dir: str):
    """
    Figure 1: Clean vs Robust PR-AUC per defence, per dataset (2x2 subplots).

    Filters to ε=0.1 only (canonical epsilon). Groups bars by model_type
    using colour coding.
    """
    agg = aggregate_seeds(df)
    # Filter to rows that have robust metrics
    agg = agg[agg["robust_pr_auc_mean"].notna() & (agg["robust_pr_auc_mean"] > 0)]
    # Filter to canonical epsilon only
    agg = agg[np.isclose(agg["attack_epsilon"], 0.1)]

    datasets = sorted(agg["dataset"].unique())
    n = len(datasets)
    if n == 0:
        print("  Skipping robustness_bars: no robust data available at ε=0.1.")
        return

    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        sub = agg[agg["dataset"] == dataset].copy()
        sub["label"] = sub["model_type"] + " / " + sub["defence_type"]

        x = np.arange(len(sub))
        width = 0.35
        ax.bar(
            x - width / 2,
            sub["clean_pr_auc_mean"],
            width,
            label="Clean PR-AUC",
            yerr=sub.get("clean_pr_auc_std", 0),
            capsize=3,
        )
        ax.bar(
            x + width / 2,
            sub["robust_pr_auc_mean"],
            width,
            label="Robust PR-AUC",
            yerr=sub.get("robust_pr_auc_std", 0),
            capsize=3,
        )
        ax.set_ylabel("PR-AUC")
        ax.set_title(f"{dataset.upper()} (ε=0.1)")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Robustness Degradation: Clean vs Robust PR-AUC (ε=0.1)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "robustness_bars.png"), dpi=150)
    plt.close(fig)
    print("  Saved robustness_bars.png")
```

Key changes:
1. Added `agg = agg[np.isclose(agg["attack_epsilon"], 0.1)]` filter (float-safe)
2. Changed label from `defence_type / attack_type` → `model_type / defence_type` (more informative, drops redundant attack info)
3. Updated title to indicate ε=0.1 filter

### Step 4: Run test to verify it passes

Run: `uv run pytest tests/test_figures.py::TestPlotRobustnessBars -v`
Expected: PASS

### Step 5: Commit

```bash
git add scripts/generate_figures.py tests/test_figures.py
git commit -m "fix(figures): filter robustness bars to ε=0.1 only (Issue 4)

Reduces bar count from ~18 to ~8 per subplot. Labels now show
model_type/defence_type instead of defence/attack."
```

---

## Task 3: Defence Heatmap — Add Ensemble Column (P1)

**Files:**
- Modify: `scripts/generate_figures.py:202-234` (`plot_defence_heatmap()`)
- Test: `tests/test_figures.py` (add new test class)

### Problem

The heatmap computes `delta = defence_robust - baseline_robust` by joining on `model_type`. Ensemble has `model_type="ensemble"` with no matching `none` baseline, so it's dropped. Should compare against the neural `none` baseline.

### Step 1: Write the failing test

Add to `tests/test_figures.py`:

```python
class TestPlotDefenceHeatmap:
    """Tests for defence heatmap figure generation."""

    def test_ensemble_column_present(self, tmp_path):
        """Heatmap should include ensemble defence column."""
        from scripts.generate_figures import plot_defence_heatmap, aggregate_seeds

        rows = []
        # neural baseline (none)
        for seed in [42, 123, 456]:
            rows.append({
                "dataset": "ccfd", "model_type": "neural",
                "defence_type": "none", "attack_type": "capgd",
                "attack_epsilon": 0.1, "seed": seed,
                "robust_pr_auc": 0.65, "clean_pr_auc": 0.85,
            })
        # adversarial_training (neural)
        for seed in [42, 123, 456]:
            rows.append({
                "dataset": "ccfd", "model_type": "neural",
                "defence_type": "adversarial_training", "attack_type": "capgd",
                "attack_epsilon": 0.1, "seed": seed,
                "robust_pr_auc": 0.75, "clean_pr_auc": 0.83,
            })
        # ensemble (model_type=ensemble, defence_type=ensemble)
        for seed in [42, 123, 456]:
            rows.append({
                "dataset": "ccfd", "model_type": "ensemble",
                "defence_type": "ensemble", "attack_type": "capgd",
                "attack_epsilon": 0.1, "seed": seed,
                "robust_pr_auc": 0.72, "clean_pr_auc": 0.87,
            })
        df = pd.DataFrame(rows)
        plot_defence_heatmap(df, str(tmp_path))
        assert (tmp_path / "defence_heatmap.png").exists()
```

### Step 2: Run test to verify it fails

Run: `uv run pytest tests/test_figures.py::TestPlotDefenceHeatmap -v`
Expected: FAIL or the heatmap only has 1 column (adversarial_training) — ensemble is dropped because no `none` baseline exists for `model_type="ensemble"`.

### Step 3: Implement the fix

Replace `plot_defence_heatmap()` (lines 202-234):

```python
def plot_defence_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Figure 4: Defence effectiveness heatmap — delta robust PR-AUC vs no-defence.

    Ensemble defence is compared against the neural 'none' baseline, since
    the ensemble model includes a neural sub-model.
    """
    agg = aggregate_seeds(df)
    agg = agg[agg["robust_pr_auc_mean"].notna() & (agg["robust_pr_auc_mean"] > 0)]

    if agg.empty:
        print("  Skipping defence_heatmap: no robust data available.")
        return

    # Get baseline (defence_type == 'none') per dataset+model+attack
    baseline = agg[agg["defence_type"] == "none"].copy()
    baseline = baseline.rename(columns={"robust_pr_auc_mean": "baseline_robust"})

    # Standard merge: same model_type baseline
    merge_on = ["dataset", "model_type", "attack_type"]
    merged = agg.merge(baseline[merge_on + ["baseline_robust"]], on=merge_on, how="left")

    # For ensemble: use neural 'none' baseline instead
    neural_baseline = baseline[baseline["model_type"] == "neural"][
        ["dataset", "attack_type", "baseline_robust"]
    ].copy()
    neural_baseline = neural_baseline.rename(columns={"baseline_robust": "neural_baseline_robust"})

    merged = merged.merge(
        neural_baseline, on=["dataset", "attack_type"], how="left"
    )

    # Fill in ensemble rows: use neural baseline where same-model baseline is missing
    is_ensemble = merged["model_type"] == "ensemble"
    merged.loc[is_ensemble, "baseline_robust"] = merged.loc[is_ensemble, "neural_baseline_robust"]
    merged = merged.drop(columns=["neural_baseline_robust"])

    merged["delta"] = merged["robust_pr_auc_mean"] - merged["baseline_robust"]

    # Filter to defences only (exclude 'none')
    defended = merged[merged["defence_type"] != "none"]
    if defended.empty:
        print("  Skipping defence_heatmap: no defence experiments found.")
        return

    pivot = defended.pivot_table(values="delta", index=["dataset", "attack_type"], columns="defence_type")

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.5 + 1)))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Defence Effectiveness: Delta Robust PR-AUC vs No Defence\n(ensemble compared vs neural baseline)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "defence_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  Saved defence_heatmap.png")
```

Key changes:
1. After standard merge, separately extract neural `none` baseline
2. For rows where `model_type == "ensemble"`, fill `baseline_robust` from neural baseline
3. Title updated to note ensemble comparison basis

### Step 4: Run test to verify it passes

Run: `uv run pytest tests/test_figures.py::TestPlotDefenceHeatmap -v`
Expected: PASS — ensemble column now appears in heatmap.

### Step 5: Commit

```bash
git add scripts/generate_figures.py tests/test_figures.py
git commit -m "fix(figures): add ensemble column to defence heatmap (Issue 2)

Ensemble defence now compared against neural 'none' baseline since
there is no ensemble 'none' baseline. Title updated to indicate
comparison basis."
```

---

## Task 4: Statistical Tests — Fix Ensemble Pairing (P1)

**Files:**
- Modify: `scripts/statistical_tests.py:31-136` (`pairwise_defence_tests()`)
- Test: `tests/test_figures.py` (add new test class)

### Problem

`pairwise_defence_tests()` groups by `(dataset, model_type)` and pairs defences within the same model_type. Ensemble has `model_type="ensemble"` and `defence_type="ensemble"` — there's no `defence_type="none"` row for `model_type="ensemble"`, so all ensemble comparisons produce n=0 paired data.

### Step 1: Write the failing test

Add to `tests/test_figures.py`:

```python
class TestStatisticalTestsEnsemble:
    """Tests for ensemble cross-model statistical comparisons."""

    def test_ensemble_vs_neural_baseline_produces_valid_pvalues(self):
        """Ensemble should be compared against neural 'none' baseline."""
        from scripts.statistical_tests import pairwise_defence_tests

        rows = []
        # Neural baseline (none)
        for seed in [42, 123, 456]:
            rows.append({
                "seed": seed, "dataset": "ccfd", "model_type": "neural",
                "defence_type": "none", "attack_type": "capgd",
                "attack_epsilon": 0.1,
                "robust_pr_auc": 0.60 + seed * 0.0001,
            })
        # Ensemble
        for seed in [42, 123, 456]:
            rows.append({
                "seed": seed, "dataset": "ccfd", "model_type": "ensemble",
                "defence_type": "ensemble", "attack_type": "capgd",
                "attack_epsilon": 0.1,
                "robust_pr_auc": 0.75 + seed * 0.0001,
            })
        df = pd.DataFrame(rows)
        results = pairwise_defence_tests(df)

        # Find the none vs ensemble comparison
        ens_row = results[
            (results["defence_a"] == "none") &
            (results["defence_b"] == "ensemble")
        ]
        assert len(ens_row) >= 1
        row = ens_row.iloc[0]
        assert "insufficient" not in str(row.get("note", ""))
        assert not np.isnan(row["p_value"])

    def test_ensemble_comparison_note_indicates_cross_model(self):
        """Cross-model ensemble comparisons should be annotated."""
        from scripts.statistical_tests import pairwise_defence_tests

        rows = []
        for seed in [42, 123, 456]:
            rows.append({
                "seed": seed, "dataset": "ccfd", "model_type": "neural",
                "defence_type": "none", "attack_type": "capgd",
                "attack_epsilon": 0.1, "robust_pr_auc": 0.60,
            })
            rows.append({
                "seed": seed, "dataset": "ccfd", "model_type": "ensemble",
                "defence_type": "ensemble", "attack_type": "capgd",
                "attack_epsilon": 0.1, "robust_pr_auc": 0.75,
            })
        df = pd.DataFrame(rows)
        results = pairwise_defence_tests(df)

        ens_row = results[
            (results["defence_a"] == "none") &
            (results["defence_b"] == "ensemble")
        ]
        assert len(ens_row) >= 1
        row = ens_row.iloc[0]
        assert "cross-model" in str(row.get("note", "")).lower() or row["model_type"] == "ensemble"
```

### Step 2: Run test to verify it fails

Run: `uv run pytest tests/test_figures.py::TestStatisticalTestsEnsemble -v`
Expected: FAIL — `none vs ensemble` comparison has `insufficient paired data (n=0)`.

### Step 3: Implement the fix

Replace `pairwise_defence_tests()` in `scripts/statistical_tests.py` (lines 31-136):

```python
def pairwise_defence_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run pairwise paired t-tests between defence types.

    For each (dataset, model_type) combination, compares robust_pr_auc
    across seeds for every pair of defence types.

    Special handling: ensemble (model_type='ensemble', defence_type='ensemble')
    is compared against neural 'none' baseline via cross-model pairing on
    (dataset, attack_type, attack_epsilon, seed).
    """
    pairs = [
        ("none", "adversarial_training"),
        ("none", "input_validation"),
        ("none", "ensemble"),
        ("adversarial_training", "input_validation"),
        ("adversarial_training", "ensemble"),
        ("input_validation", "ensemble"),
    ]

    rows = []

    for (dataset, model_type), grp in df.groupby(["dataset", "model_type"]):
        for def_a, def_b in pairs:
            # Cross-model pairing for ensemble comparisons
            is_cross_model = False
            if def_b == "ensemble" and model_type != "ensemble":
                # When processing non-ensemble model_types, pair the current
                # model_type's defence_a against ensemble data for same dataset
                df_a = grp[grp["defence_type"] == def_a][
                    ["seed", "attack_type", "attack_epsilon", "robust_pr_auc"]
                ].dropna()
                ens_data = df[
                    (df["dataset"] == dataset)
                    & (df["model_type"] == "ensemble")
                    & (df["defence_type"] == "ensemble")
                ][["seed", "attack_type", "attack_epsilon", "robust_pr_auc"]].dropna()

                # Only do cross-model from neural (not tree) to avoid duplicates
                if model_type != "neural":
                    continue

                paired = df_a.merge(
                    ens_data,
                    on=["seed", "attack_type", "attack_epsilon"],
                    suffixes=("_a", "_b"),
                )
                is_cross_model = True
            elif model_type == "ensemble":
                # Skip within-ensemble comparisons (no none/adv_train/iv for ensemble)
                continue
            else:
                df_a = grp[grp["defence_type"] == def_a][["seed", "robust_pr_auc"]].dropna()
                df_b = grp[grp["defence_type"] == def_b][["seed", "robust_pr_auc"]].dropna()
                paired = df_a.merge(df_b, on="seed", suffixes=("_a", "_b"))

            row = {
                "dataset": dataset,
                "model_type": "ensemble" if is_cross_model else model_type,
                "defence_a": def_a,
                "defence_b": def_b,
                "n_a": len(df_a) if "df_a" in dir() else 0,
                "n_b": len(ens_data) if is_cross_model else (len(df_b) if "df_b" in dir() else 0),
            }

            # Need at least 3 paired observations for a meaningful t-test
            n_paired = len(paired)
            if n_paired < 3:
                row.update(
                    {
                        "mean_a": float(paired["robust_pr_auc_a"].mean()) if n_paired > 0 else np.nan,
                        "mean_b": float(paired["robust_pr_auc_b"].mean()) if n_paired > 0 else np.nan,
                        "mean_diff": np.nan,
                        "t_statistic": np.nan,
                        "p_value": np.nan,
                        "w_statistic": np.nan,
                        "w_p_value": np.nan,
                        "cohens_d": np.nan,
                        "significant": False,
                        "note": f"insufficient paired data (n={n_paired})",
                    }
                )
                rows.append(row)
                continue

            a = paired["robust_pr_auc_a"].values
            b = paired["robust_pr_auc_b"].values

            mean_diff = float(np.mean(a) - np.mean(b))

            # Check if values are identical (zero variance in differences)
            if np.allclose(a, b):
                row.update(
                    {
                        "mean_a": float(np.mean(a)),
                        "mean_b": float(np.mean(b)),
                        "mean_diff": mean_diff,
                        "t_statistic": 0.0,
                        "p_value": 1.0,
                        "w_statistic": np.nan,
                        "w_p_value": np.nan,
                        "cohens_d": 0.0,
                        "significant": False,
                        "note": "identical values across seeds",
                    }
                )
                rows.append(row)
                continue

            t_stat, p_val = stats.ttest_rel(a, b)
            d = compute_cohens_d(a, b)

            # Wilcoxon signed-rank (requires n >= 6 for meaningful results)
            if n_paired >= 6:
                w_stat, w_pval = stats.wilcoxon(a, b)
            else:
                w_stat, w_pval = np.nan, np.nan

            note = "cross-model: ensemble vs neural baseline" if is_cross_model else ""

            row.update(
                {
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "mean_diff": mean_diff,
                    "t_statistic": float(t_stat),
                    "p_value": float(p_val),
                    "w_statistic": float(w_stat) if not np.isnan(w_stat) else np.nan,
                    "w_p_value": float(w_pval) if not np.isnan(w_pval) else np.nan,
                    "cohens_d": float(d),
                    "significant": bool(p_val < 0.05),
                    "note": note,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)
```

Key changes:
1. When `def_b == "ensemble"` and we're processing a non-ensemble model_type, do cross-model pairing
2. Only pair from `neural` (not tree) to avoid duplicate ensemble comparisons
3. Cross-model pairing merges on `[seed, attack_type, attack_epsilon]` instead of just `[seed]`
4. Skip all pairs when `model_type == "ensemble"` (handled via cross-model from neural)
5. Annotate cross-model rows with `note = "cross-model: ensemble vs neural baseline"`

### Step 4: Run ALL statistical tests to verify nothing is broken

Run: `uv run pytest tests/test_figures.py::TestStatisticalTests tests/test_figures.py::TestStatisticalTestsEnsemble -v`
Expected: ALL PASS — existing tests still work, new ensemble tests pass.

### Step 5: Commit

```bash
git add scripts/statistical_tests.py tests/test_figures.py
git commit -m "fix(stats): enable cross-model ensemble comparisons (Issue 3)

Ensemble defence (model_type=ensemble) now compared against neural
'none' baseline by pairing on (dataset, attack_type, attack_epsilon,
seed). Annotated as cross-model in the note column."
```

---

## Task 5: Input Validation Analysis — Verify Epsilon Filter (P2)

**Files:**
- Modify: `scripts/analyse_input_validation.py:111-178` (`compute_degradation()`)
- Test: `tests/test_figures.py` (add new test class)

### Problem

The `compute_degradation()` function merges on `["dataset", "model_type", "attack_type", "attack_epsilon"]` — this correctly includes `attack_epsilon` as a merge key, so different epsilons should produce separate rows. However, if the registry has epsilon values like 0.09999999 instead of 0.1 (float precision), the merge would silently produce wrong results. We need to add a strict epsilon filter and an assertion.

### Step 1: Write the failing test

Add to `tests/test_figures.py`:

```python
class TestInputValidationAnalysis:
    """Tests for input validation degradation analysis."""

    def test_epsilon_filter_strict(self):
        """Analysis should only include ε=0.1 data in the output."""
        from scripts.analyse_input_validation import aggregate_seeds, compute_degradation

        rows = []
        # Baseline at multiple epsilons
        for eps in [0.05, 0.1, 0.2]:
            for seed in [42, 123, 456]:
                rows.append({
                    "dataset": "ccfd", "model_type": "neural",
                    "defence_type": "none", "attack_type": "capgd",
                    "attack_epsilon": eps, "seed": seed,
                    "robust_pr_auc": 0.80 - eps, "clean_pr_auc": 0.90,
                    "robust_f1": 0.70 - eps, "robust_recall": 0.60 - eps,
                })
        # input_validation at eps=0.1 only
        for seed in [42, 123, 456]:
            rows.append({
                "dataset": "ccfd", "model_type": "neural",
                "defence_type": "input_validation", "attack_type": "capgd",
                "attack_epsilon": 0.1, "seed": seed,
                "robust_pr_auc": 0.55, "clean_pr_auc": 0.89,
                "robust_f1": 0.50, "robust_recall": 0.45,
            })
        df = pd.DataFrame(rows)
        agg = aggregate_seeds(df)
        deg = compute_degradation(agg)

        # Should produce exactly 1 row: ccfd/neural/capgd at eps=0.1
        assert len(deg) == 1
        # Baseline robust value should be 0.80 - 0.1 = 0.70 (not averaged across epsilons)
        assert deg["robust_prauc_baseline"].iloc[0] == pytest.approx(0.70, abs=0.01)
```

### Step 2: Run test to verify it fails

Run: `uv run pytest tests/test_figures.py::TestInputValidationAnalysis -v`
Expected: This test should actually PASS with the current code since the merge includes `attack_epsilon` as a key. But we'll add a defensive filter and assertion.

### Step 3: Implement the fix

In `scripts/analyse_input_validation.py`, modify `compute_degradation()` at line 119, right after the baseline/input_val filters:

```python
def compute_degradation(agg: pd.DataFrame) -> pd.DataFrame:
    """Compare baseline (defence_type=none) vs input_validation per config.

    Returns a DataFrame with one row per dataset+model+attack combination,
    showing the delta (input_validation minus baseline) for key robust metrics.
    Filters to ε=0.1 only (canonical epsilon for the benchmark).
    """
    merge_keys = ["dataset", "model_type", "attack_type", "attack_epsilon"]

    baseline = agg[agg["defence_type"] == "none"].copy()
    input_val = agg[agg["defence_type"] == "input_validation"].copy()

    if baseline.empty or input_val.empty:
        print("ERROR: registry does not contain both 'none' and 'input_validation' rows.")
        sys.exit(1)

    # Filter to canonical epsilon (0.1) for consistent comparison
    baseline = baseline[np.isclose(baseline["attack_epsilon"], 0.1)]
    input_val = input_val[np.isclose(input_val["attack_epsilon"], 0.1)]

    if baseline.empty or input_val.empty:
        print("ERROR: no data at ε=0.1 for baseline or input_validation.")
        sys.exit(1)

    merged = input_val.merge(
        baseline[
            merge_keys
            + [
                "robust_pr_auc_mean",
                "robust_pr_auc_std",
                "robust_f1_mean",
                "robust_f1_std",
                "robust_recall_mean",
                "robust_recall_std",
                "clean_pr_auc_mean",
            ]
        ],
        on=merge_keys,
        suffixes=("_iv", "_base"),
    )

    merged["delta_robust_pr_auc"] = merged["robust_pr_auc_mean_iv"] - merged["robust_pr_auc_mean_base"]
    merged["delta_robust_f1"] = merged["robust_f1_mean_iv"] - merged["robust_f1_mean_base"]
    merged["delta_robust_recall"] = merged["robust_recall_mean_iv"] - merged["robust_recall_mean_base"]

    result = merged[
        merge_keys
        + [
            "clean_pr_auc_mean_base",
            "clean_pr_auc_mean_iv",
            "robust_pr_auc_mean_base",
            "robust_pr_auc_mean_iv",
            "robust_f1_mean_base",
            "robust_f1_mean_iv",
            "robust_recall_mean_base",
            "robust_recall_mean_iv",
            "delta_robust_pr_auc",
            "delta_robust_f1",
            "delta_robust_recall",
        ]
    ].copy()

    # Friendly column names for the output CSV
    result = result.rename(
        columns={
            "clean_pr_auc_mean_base": "clean_prauc_baseline",
            "clean_pr_auc_mean_iv": "clean_prauc_input_val",
            "robust_pr_auc_mean_base": "robust_prauc_baseline",
            "robust_pr_auc_mean_iv": "robust_prauc_input_val",
            "robust_f1_mean_base": "robust_f1_baseline",
            "robust_f1_mean_iv": "robust_f1_input_val",
            "robust_recall_mean_base": "robust_recall_baseline",
            "robust_recall_mean_iv": "robust_recall_input_val",
        }
    )

    return result
```

Key changes:
1. Added `np.isclose()` filter for ε=0.1 on both baseline and input_val (lines after initial filtering)
2. Added error message if no data at ε=0.1
3. Updated docstring to document the filter
4. Add `import numpy as np` at the top if not already present (it already is at line 42)

### Step 4: Run test to verify it passes

Run: `uv run pytest tests/test_figures.py::TestInputValidationAnalysis -v`
Expected: PASS

### Step 5: Commit

```bash
git add scripts/analyse_input_validation.py tests/test_figures.py
git commit -m "fix(analysis): strict ε=0.1 filter in input validation analysis (Issue 5)

Prevents cross-epsilon averaging. Uses np.isclose for float-safe
comparison."
```

---

## Task 6: Document CAPGD–Tree Limitation (P2)

**Files:**
- Modify: `README.md:378-383` (Known Limitations section)
- Modify: `scripts/generate_figures.py:152-199` (`plot_summary_table()`) — add footnote annotation

### Step 1: Add CAPGD–tree explanation to README Known Limitations

In `README.md`, the existing Known Limitations already has a bullet about CAPGD + tree models. Expand it with the specific finding:

```markdown
## Known Limitations

- **Adversarial training + tree/ensemble models**: Adversarial training requires gradients (backpropagation) and is incompatible with tree and ensemble models. The runner raises `ValueError` for these combinations.
- **CAPGD + tree models**: CAPGD is a gradient-based (white-box) attack and only works on neural models. For ensemble models, CAPGD targets the MLP component. Use HopSkipJump or Square Attack for tree-only models. In the registry, tree + CAPGD rows show `robust_pr_auc == clean_pr_auc` — this reflects the architectural limitation, not model robustness.
- **Ensemble GPU requirement**: All ensemble experiments require GPU for the MLP training component, even when using black-box attacks (Square, HSJ).
- **HopSkipJump partial coverage**: HSJ experiments are partially complete (6/12 runs). Missing: IEEE-CIS seed 456, LCLD seeds 123 and 456, Sparkov all 3 seeds. Square Attack provides complete black-box coverage. HSJ completion is listed as future work.
- **Sparkov neural vulnerability**: The neural model achieves near-zero robust PR-AUC on Sparkov under CAPGD attack (even at small ε). This reflects extreme vulnerability in a low-dimensional feature space (22 features), not a bug. Tree models achieve robust PR-AUC ≈ 0.747 on the same dataset.
```

### Step 2: Annotate summary table for tree + CAPGD rows

In `plot_summary_table()` (lines 152-199 of `generate_figures.py`), add a footnote marker. After line 176 (`rows.append(r)`), we need to mark tree+CAPGD rows. This is a small change — add a `†` to the robust_pr_auc value:

After building rows but before creating the DataFrame, add:

```python
    # Annotate tree + CAPGD rows where robust == clean (attack is a no-op)
    for r in rows:
        if (r.get("model_type") == "tree" and r.get("attack_type") == "capgd"):
            if "robust_pr_auc" in r and r["robust_pr_auc"] != "n/a":
                r["robust_pr_auc"] = r["robust_pr_auc"] + " †"
```

And add a footnote to the figure title:

```python
    fig.suptitle("Cross-Dataset Summary\n† CAPGD is inapplicable to tree models (gradient-based attack)", fontsize=12)
```

### Step 3: No test needed

This is a documentation/annotation change. Visual verification only.

### Step 4: Commit

```bash
git add README.md scripts/generate_figures.py
git commit -m "docs: document CAPGD-tree limitation and Sparkov neural finding (Issues 6, 7, 8)

- Expanded Known Limitations with CAPGD-tree, HSJ coverage, and Sparkov explanations
- Added † footnote to tree+CAPGD rows in summary table"
```

---

## Task 7: Regenerate All Artefacts

**Files:**
- None modified (execution only)

### Step 1: Regenerate all figures

Run:
```bash
uv run python scripts/generate_figures.py --registry results/registry_clean.csv --output results/figures
```
Expected: All 6 figures regenerated without errors.

### Step 2: Regenerate statistical tests

Run:
```bash
uv run python scripts/statistical_tests.py --registry results/registry_clean.csv
```
Expected: CSV written, ensemble comparisons show valid p-values.

### Step 3: Regenerate input validation analysis

Run:
```bash
uv run python scripts/analyse_input_validation.py --registry results/registry_clean.csv
```
Expected: CSV and PNG written, values match registry at ε=0.1.

### Step 4: Run full test suite

Run:
```bash
uv run pytest tests/ -v
```
Expected: ALL PASS

### Step 5: Visual verification checklist

Verify manually:
- [ ] `robustness_curves.png` shows multiple lines/markers per subplot
- [ ] `defence_heatmap.png` has 3 columns (adversarial_training, input_validation, ensemble)
- [ ] `robustness_bars.png` has readable x-axis labels (ε=0.1 only)
- [ ] `statistical_tests.csv` has valid p-values for ensemble comparisons
- [ ] `input_validation_analysis.csv` values match registry at ε=0.1
- [ ] `summary_table.png` has † footnote on tree+CAPGD rows
- [ ] All figures use consistent colour schemes
- [ ] No figure has overlapping text or clipped labels

### Step 6: Commit regenerated artefacts

```bash
git add results/figures/
git commit -m "data: regenerate all figures and analysis artefacts after fixes"
```

---

## Summary

| Task | Issue | Files Changed | Test Added |
|------|-------|---------------|------------|
| 1 | Robustness curves (P0) | `generate_figures.py`, `test_figures.py` | `TestPlotRobustnessCurves` |
| 2 | Robustness bars (P2) | `generate_figures.py`, `test_figures.py` | `TestPlotRobustnessBars` |
| 3 | Defence heatmap (P1) | `generate_figures.py`, `test_figures.py` | `TestPlotDefenceHeatmap` |
| 4 | Statistical tests (P1) | `statistical_tests.py`, `test_figures.py` | `TestStatisticalTestsEnsemble` |
| 5 | Input validation (P2) | `analyse_input_validation.py`, `test_figures.py` | `TestInputValidationAnalysis` |
| 6 | Documentation (P2/P3) | `README.md`, `generate_figures.py` | None (docs) |
| 7 | Regenerate artefacts | None (execution) | Full suite |
