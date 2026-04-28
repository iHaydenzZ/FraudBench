"""Phase A diagnostic: quantify all-zero OHE rows in LCLD + IEEE-CIS splits per seed.

Implements steps A1.1, A1.2, A1.3, A2.1 from
docs/plans/2026-04-28-lcld-seed42-ohe-fix.md

Run on Colab (or any environment with dataset access):
    python scratch/diagnose_seed42.py

Expected runtime: ~30 seconds (CPU is fine, no GPU needed).

Paste the full output back to Claude to make the Phase B path decision.
"""

from datasets.loader import load_dataset
from datasets.splitter import split_dataset
from preprocessing.processor import DataPreprocessor


def diagnose_dataset(name: str, sample_frac: float, seeds: list[int], focus_cols: list[str] | None = None) -> dict:
    """For each (seed, categorical column), count all-zero OHE rows in the test split."""
    print(f"\n{'=' * 70}")
    print(f"=== {name.upper()} — all-zero OHE row counts in test split ===")
    print(f"{'=' * 70}")
    print(f"{'seed':<6}{'col':<25}{'all_zero':<12}{'pct':<10}{'test_size':<10}")

    findings = {}
    for seed in seeds:
        ds = load_dataset(name, {"sample_frac": sample_frac})
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(ds, random_state=seed)
        pp = DataPreprocessor(ds.feature_types).fit(X_train)
        X_test_p = pp.transform(X_test)

        cols_to_check = focus_cols or [c for c, t in ds.feature_types.items() if t in ("categorical", "binary")]
        for cat_col in cols_to_check:
            ohe_cols = [c for c in X_test_p.columns if c.startswith(cat_col + "_")]
            if not ohe_cols:
                continue
            block = X_test_p[ohe_cols]
            all_zero_count = int((block.sum(axis=1) == 0).sum())
            all_zero_pct = all_zero_count / len(block) * 100
            findings[(seed, cat_col)] = (all_zero_count, all_zero_pct, len(X_test))
            if all_zero_count > 0:
                print(f"{seed:<6}{cat_col:<25}{all_zero_count:<12}{all_zero_pct:<10.3f}{len(X_test):<10}")
    return findings


def diagnose_test_only_categories(name: str, sample_frac: float, seeds: list[int], cols: list[str]):
    """For each offending column on each seed, list test-only categories with row counts."""
    print(f"\n--- {name.upper()} test-only categories per (seed, col) ---")
    for seed in seeds:
        ds = load_dataset(name, {"sample_frac": sample_frac})
        X_train, X_val, X_test, _, _, _ = split_dataset(ds, random_state=seed)
        for cat_col in cols:
            if cat_col not in X_train.columns:
                continue
            train_cats = set(X_train[cat_col].dropna().unique())
            test_cats = set(X_test[cat_col].dropna().unique())
            test_only = test_cats - train_cats
            if test_only:
                print(
                    f"seed={seed}, col={cat_col}: test-only={sorted(test_only)[:10]}{'...' if len(test_only) > 10 else ''}"
                )
                for cat in sorted(test_only):
                    n_test = int((X_test[cat_col] == cat).sum())
                    print(f"    '{cat}': {n_test} rows in test, 0 in train")


def diagnose_train_frequency(name: str, sample_frac: float, seed: int, cols: list[str]):
    """Show bottom-of-distribution train counts to guide min_frequency choice."""
    print(f"\n--- {name.upper()} train-frequency distribution (seed={seed}, bottom 10 per col) ---")
    ds = load_dataset(name, {"sample_frac": sample_frac})
    X_train, _, _, _, _, _ = split_dataset(ds, random_state=seed)
    for cat_col in cols:
        if cat_col not in X_train.columns:
            continue
        counts = X_train[cat_col].value_counts()
        print(
            f"\n{cat_col}: total unique={len(counts)}, min_count={counts.min()}, "
            f"<10={int((counts < 10).sum())}, <5={int((counts < 5).sum())}, <3={int((counts < 3).sum())}"
        )
        print("  bottom 10:")
        for cat, n in counts.tail(10).items():
            print(f"    {cat}: {n}")


def main():
    SEEDS = [42, 123, 456]

    # Step A1.1: LCLD all-column scan
    lcld_findings = diagnose_dataset("lcld", sample_frac=0.1, seeds=SEEDS)

    # Identify offending columns (≥1% all-zero on any seed) for further inspection
    offenders = sorted({col for (seed, col), (cnt, pct, _) in lcld_findings.items() if pct >= 0.5})
    if offenders:
        print(f"\n*** LCLD offending columns (≥0.5% all-zero on any seed): {offenders} ***")
        # Step A1.2: which categories are test-only
        diagnose_test_only_categories("lcld", sample_frac=0.1, seeds=SEEDS, cols=offenders)
        # Step A1.3: how rare are categories in train (drives min_frequency choice)
        diagnose_train_frequency("lcld", sample_frac=0.1, seed=42, cols=offenders)
    else:
        print(
            "\n*** No LCLD offenders found — Phase A hypothesis is wrong. Check OHE-validity check semantics instead (Path B-γ). ***"
        )

    # Step A2.1: IEEE-CIS regression check on the 3 binding OHE blocks
    diagnose_dataset(
        "ieee_cis",
        sample_frac=0.1,
        seeds=SEEDS,
        focus_cols=["ProductCD", "card4", "card6"],
    )

    print("\n" + "=" * 70)
    print("Phase A diagnostic complete.")
    print("=" * 70)
    print("\nNext: paste this entire output back to Claude to make the Phase B path decision.")


if __name__ == "__main__":
    main()
