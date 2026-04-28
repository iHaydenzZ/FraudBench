"""Phase A.2 confirmatory diagnostic: measure raw clean g2/g3 pass rate per seed on LCLD.

Replaces the original Phase A hypothesis (OHE all-zero rows) which was falsified
by scratch/diagnose_seed42.py. New hypothesis (per per-seed table in
docs/g1_projection_findings.md and a code comment in notebooks/g1_projection_attack.ipynb):
seed-42 LCLD test set has ~11% of clean rows where pub_rec_bankruptcies > pub_rec
(g3 violations), independent of any preprocessing or attack.

Run on Colab:
    python scratch/diagnose_g3_baseline.py

Expected output:
    seed=42  g2 pass = ~0.997  g3 pass = ~0.890  (g3 is the bottleneck)
    seed=123 g2 pass = ~0.993  g3 pass = ~1.000
    seed=456 g2 pass = ~0.997  g3 pass = ~0.995
"""

from datasets.loader import load_dataset
from datasets.splitter import split_dataset


def check_g2(df):
    """g2: open_acc <= total_acc (inequality on raw integer counts)."""
    return df["open_acc"].astype(float) <= df["total_acc"].astype(float)


def check_g3(df):
    """g3: pub_rec_bankruptcies <= pub_rec (inequality on raw integer counts)."""
    return df["pub_rec_bankruptcies"].astype(float) <= df["pub_rec"].astype(float)


def diagnose_constraint_violations(name: str, sample_frac: float, seeds: list):
    print(f"\n{'=' * 78}")
    print(f"=== {name.upper()} — clean baseline g2/g3 pass rate per seed (raw test set) ===")
    print(f"{'=' * 78}")
    print(
        f"{'seed':<6}{'n_test':<10}{'g2_pass':<12}{'g3_pass':<12}"
        f"{'g3_viol_n':<12}{'g3_viol_pct':<14}{'nan_pubrec':<12}{'nan_bankrupt':<14}"
    )
    for seed in seeds:
        ds = load_dataset(name, {"sample_frac": sample_frac})
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(ds, random_state=seed)
        g2_mask = check_g2(X_test)
        g3_mask = check_g3(X_test)
        g3_viol_n = int((~g3_mask).sum())
        nan_pubrec = int(X_test["pub_rec"].isna().sum())
        nan_bankrupt = int(X_test["pub_rec_bankruptcies"].isna().sum())
        print(
            f"{seed:<6}{len(X_test):<10}"
            f"{g2_mask.mean():<12.4f}{g3_mask.mean():<12.4f}"
            f"{g3_viol_n:<12}{g3_viol_n / len(X_test) * 100:<14.2f}"
            f"{nan_pubrec:<12}{nan_bankrupt:<14}"
        )


def diagnose_violation_examples(name: str, sample_frac: float, seed: int, n_examples: int = 8):
    """Show actual violating rows so we can see what kind of data they are."""
    print(f"\n--- Sample of g3 violations in {name.upper()} test set (seed={seed}) ---")
    ds = load_dataset(name, {"sample_frac": sample_frac})
    _, _, X_test, _, _, y_test = split_dataset(ds, random_state=seed)
    g3_mask = check_g3(X_test)
    violators = X_test[~g3_mask]
    if len(violators) == 0:
        print("  (no violations)")
        return
    cols = ["pub_rec", "pub_rec_bankruptcies", "open_acc", "total_acc"]
    print(f"  total violators: {len(violators)} ({len(violators) / len(X_test) * 100:.2f}% of test)")
    print(f"  showing first {min(n_examples, len(violators))}:")
    print(violators[cols].head(n_examples).to_string())

    # Distribution of violation magnitude
    diff = X_test["pub_rec_bankruptcies"].astype(float) - X_test["pub_rec"].astype(float)
    diff_violators = diff[~g3_mask]
    print("\n  violation magnitude (pub_rec_bankruptcies - pub_rec) stats on violators:")
    print(
        f"    min={diff_violators.min()}, median={diff_violators.median()}, "
        f"max={diff_violators.max()}, mean={diff_violators.mean():.2f}"
    )


def diagnose_split_overlap(name: str, sample_frac: float, seeds: list):
    """Are violator rows shared across seeds? (i.e. bad data) or split-specific?"""
    print(f"\n--- {name.upper()} g3 violator row-index overlap across seeds ---")
    violator_idx = {}
    for seed in seeds:
        ds = load_dataset(name, {"sample_frac": sample_frac})
        _, _, X_test, _, _, _ = split_dataset(ds, random_state=seed)
        g3_mask = check_g3(X_test)
        violator_idx[seed] = set(X_test.index[~g3_mask].tolist())
        print(f"  seed={seed}: {len(violator_idx[seed])} violators in test set")

    print(
        f"\n  All-seeds intersection (rows that are violators AND landed in test on all 3 seeds): "
        f"{len(set.intersection(*violator_idx.values()))}"
    )
    print(
        f"  Any-seed union (rows that are violators in test on at least one seed): "
        f"{len(set.union(*violator_idx.values()))}"
    )
    print("\n  Per-pair overlap (rows that are violators in test on both seeds in pair):")
    seeds_l = list(seeds)
    for i, sa in enumerate(seeds_l):
        for sb in seeds_l[i + 1 :]:
            inter = violator_idx[sa] & violator_idx[sb]
            print(f"    {sa} ∩ {sb}: {len(inter)} rows")


def main():
    SEEDS = [42, 123, 456]
    diagnose_constraint_violations("lcld", sample_frac=0.1, seeds=SEEDS)
    diagnose_violation_examples("lcld", sample_frac=0.1, seed=42)
    diagnose_split_overlap("lcld", sample_frac=0.1, seeds=SEEDS)

    print("\n" + "=" * 78)
    print("Phase A.2 diagnostic complete.")
    print("=" * 78)
    print("\nKey question to answer from output:")
    print("  - Is seed-42 g3_pass ~0.89? (confirms hypothesis)")
    print("  - Are violators in test set drawn from a global pool of 'bad' rows in")
    print("    the LCLD source data? (split-overlap section)")
    print("\nPaste full output back to Claude.")


if __name__ == "__main__":
    main()
