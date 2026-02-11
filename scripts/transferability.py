"""Cross-model adversarial transferability experiments.

Tests whether adversarial examples generated against one model type
(e.g., Neural MLP) can fool another model type (e.g., XGBoost) on
the same dataset. Key research question about attack portability.

Usage:
    uv run python -m scripts.transferability --dataset ccfd --seed 42 --epsilon 0.1
    uv run python -m scripts.transferability --dataset ieee_cis --attack-type hopskipjump
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

# Attack-model compatibility: CAPGD requires a differentiable (neural) source.
# HopSkipJump and Square are black-box and work on any model.
_NEURAL_ONLY_ATTACKS = {"capgd"}


def _run_attack(attack_type, model, X, y, schema, feature_types, params):
    """Dispatch to the appropriate attack implementation."""
    if attack_type == "capgd":
        from attacks.capgd import capgd_attack
        return capgd_attack(model, X, y, schema, feature_types, params)
    elif attack_type == "hopskipjump":
        from attacks.hopskipjump import hopskipjump_attack
        return hopskipjump_attack(model, X, y, schema, feature_types, params)
    elif attack_type == "square":
        from attacks.square import square_attack
        return square_attack(model, X, y, schema, feature_types, params)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


def run_transferability(dataset_name, seed, epsilon, attack_type, sample_frac):
    """Run transferability experiment: generate advex on source, evaluate on target."""

    from datasets.loader import load_dataset
    from datasets.splitter import split_dataset
    from preprocessing.processor import DataPreprocessor, get_preprocessor_path
    from constraints.schema import ConstraintSchema
    from evaluation.metrics import compute_metrics
    from models.neural import NeuralModel
    from models.tree import TreeModel

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"\n[1] Loading dataset: {dataset_name} (sample_frac={sample_frac})...")
    dataset_config = {"sample_frac": sample_frac}
    dataset = load_dataset(dataset_name, config=dataset_config)
    print(f"    Loaded {len(dataset.X)} samples, {len(dataset.feature_names)} features.")
    if "fraud_rate" in dataset.meta:
        print(f"    Fraud rate: {dataset.meta['fraud_rate']:.4f}")

    # ------------------------------------------------------------------
    # 2. Split
    # ------------------------------------------------------------------
    print(f"\n[2] Splitting dataset (seed={seed})...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        dataset, test_size=0.2, val_size=0.2, random_state=seed,
    )
    print(f"    Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # 3. Preprocess
    # ------------------------------------------------------------------
    print("\n[3] Preprocessing features...")
    ds_name = dataset.meta.get("name", "unknown")
    n_samples = len(dataset.X)
    preprocessor_path = get_preprocessor_path(ds_name, seed, n_samples)

    if os.path.exists(preprocessor_path):
        print("    Loading existing preprocessor...")
        preprocessor = DataPreprocessor.load(preprocessor_path)
        X_train_proc = preprocessor.transform(X_train)
    else:
        preprocessor = DataPreprocessor(dataset.feature_types)
        print("    Fitting preprocessor on Train set...")
        X_train_proc = preprocessor.fit_transform(X_train)
        preprocessor.save(preprocessor_path)

    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    print(f"    Processed feature count: {X_train_proc.shape[1]}")

    # ------------------------------------------------------------------
    # 4. Build constraint schema (processed space)
    # ------------------------------------------------------------------
    processed_feature_types = {c: "numeric" for c in X_train_proc.columns}
    schema = ConstraintSchema.from_data(X_train_proc, processed_feature_types)

    # ------------------------------------------------------------------
    # 5. Train both models
    # ------------------------------------------------------------------
    print("\n[4] Training Neural model...")
    neural_params = {"epochs": 15, "hidden_dim": 128, "batch_size": 256}
    neural = NeuralModel(neural_params)
    t0 = time.time()
    neural.fit(X_train_proc, y_train)
    neural_train_time = time.time() - t0
    print(f"    Neural training complete ({neural_train_time:.1f}s)")

    print("\n[5] Training Tree model...")
    tree_params = {"max_depth": 6, "n_estimators": 200}
    tree = TreeModel(tree_params)
    t0 = time.time()
    tree.fit(X_train_proc, y_train)
    tree_train_time = time.time() - t0
    print(f"    Tree training complete ({tree_train_time:.1f}s)")

    # ------------------------------------------------------------------
    # 6. Evaluate both models on clean data
    # ------------------------------------------------------------------
    print("\n[6] Evaluating models on clean test data...")
    neural_clean = compute_metrics(y_test, neural.predict_proba(X_test_proc))
    tree_clean = compute_metrics(y_test, tree.predict_proba(X_test_proc))

    print("    Neural (clean):", {k: f"{v:.4f}" for k, v in neural_clean.items()})
    print("    Tree   (clean):", {k: f"{v:.4f}" for k, v in tree_clean.items()})

    # ------------------------------------------------------------------
    # 7. Determine source/target pairs based on attack compatibility
    # ------------------------------------------------------------------
    models = {"neural": neural, "tree": tree}
    clean_metrics = {"neural": neural_clean, "tree": tree_clean}

    if attack_type in _NEURAL_ONLY_ATTACKS:
        # CAPGD only works on differentiable models
        source_names = ["neural"]
        print(f"\n    Note: {attack_type} requires a differentiable source -> source=neural only")
    else:
        source_names = ["neural", "tree"]

    attack_params = {"epsilon": epsilon, "steps": 10, "type": attack_type}
    results = []

    # ------------------------------------------------------------------
    # 8. For each source model, generate advex and evaluate on both models
    # ------------------------------------------------------------------
    for src_name in source_names:
        src_model = models[src_name]
        print(f"\n{'='*60}")
        print(f"  Source model: {src_name} | Attack: {attack_type} | eps={epsilon}")
        print(f"{'='*60}")

        print(f"    Generating adversarial examples on {src_name}...")
        t0 = time.time()
        X_adv = _run_attack(
            attack_type, src_model, X_test_proc, y_test,
            schema, processed_feature_types, attack_params,
        )
        attack_time = time.time() - t0
        print(f"    Attack complete ({attack_time:.1f}s)")

        # Evaluate each model on the adversarial examples
        for tgt_name in ["neural", "tree"]:
            tgt_model = models[tgt_name]
            is_transfer = tgt_name != src_name

            y_probs_adv = tgt_model.predict_proba(X_adv)
            adv_metrics = compute_metrics(y_test, y_probs_adv)

            label = "TRANSFER" if is_transfer else "DIRECT"
            print(f"\n    [{label}] {src_name} -> {tgt_name}:")
            print(f"      Clean PR-AUC:  {clean_metrics[tgt_name]['pr_auc']:.4f}")
            print(f"      Adv   PR-AUC:  {adv_metrics['pr_auc']:.4f}")
            print(f"      Clean F1:      {clean_metrics[tgt_name]['f1']:.4f}")
            print(f"      Adv   F1:      {adv_metrics['f1']:.4f}")

            # Compute transfer rate:
            # How much of the source's degradation transfers to the target?
            src_clean_pr = clean_metrics[src_name]["pr_auc"]
            src_adv_pr = compute_metrics(y_test, src_model.predict_proba(X_adv))["pr_auc"]
            src_drop = src_clean_pr - src_adv_pr

            tgt_clean_pr = clean_metrics[tgt_name]["pr_auc"]
            tgt_drop = tgt_clean_pr - adv_metrics["pr_auc"]

            transfer_rate = tgt_drop / src_drop if abs(src_drop) > 1e-9 else float("nan")

            results.append({
                "dataset": dataset_name,
                "seed": seed,
                "attack_type": attack_type,
                "epsilon": epsilon,
                "source_model": src_name,
                "target_model": tgt_name,
                "is_transfer": is_transfer,
                "source_clean_pr_auc": src_clean_pr,
                "source_adv_pr_auc": src_adv_pr,
                "source_pr_auc_drop": src_drop,
                "target_clean_pr_auc": tgt_clean_pr,
                "target_adv_pr_auc": adv_metrics["pr_auc"],
                "target_pr_auc_drop": tgt_drop,
                "transfer_rate": transfer_rate,
                "target_clean_f1": clean_metrics[tgt_name]["f1"],
                "target_adv_f1": adv_metrics["f1"],
                "target_clean_recall": clean_metrics[tgt_name]["recall"],
                "target_adv_recall": adv_metrics["recall"],
                "attack_time_sec": attack_time,
            })

    # ------------------------------------------------------------------
    # 9. Summary table
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("TRANSFERABILITY SUMMARY")
    print("=" * 70)

    summary_cols = [
        "source_model", "target_model", "is_transfer",
        "target_clean_pr_auc", "target_adv_pr_auc",
        "target_pr_auc_drop", "transfer_rate",
    ]
    print(df[summary_cols].to_string(index=False, float_format="%.4f"))

    # ------------------------------------------------------------------
    # 10. Save results
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    out_path = "results/transferability_results.csv"

    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(f"Total rows in file: {len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model adversarial transferability experiment"
    )
    parser.add_argument(
        "--dataset", default="ccfd",
        choices=["ccfd", "ieee_cis", "lcld", "sparkov"],
        help="Dataset name (default: ccfd)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Attack epsilon (default: 0.1)"
    )
    parser.add_argument(
        "--attack-type", default="capgd",
        choices=["capgd", "hopskipjump", "square"],
        help="Attack type (default: capgd). "
             "Note: capgd requires neural source model.",
    )
    parser.add_argument(
        "--sample-frac", type=float, default=0.1,
        help="Fraction of dataset to use (default: 0.1)",
    )
    args = parser.parse_args()

    # Validate
    if args.attack_type in _NEURAL_ONLY_ATTACKS:
        print(f"Note: {args.attack_type} is gradient-based; "
              f"only neural model will be used as source.")

    run_transferability(
        dataset_name=args.dataset,
        seed=args.seed,
        epsilon=args.epsilon,
        attack_type=args.attack_type,
        sample_frac=args.sample_frac,
    )


if __name__ == "__main__":
    main()
