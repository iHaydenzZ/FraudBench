import argparse
import yaml
import sys
import time
from pathlib import Path


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_parser():
    parser = argparse.ArgumentParser(description="FRBS MVP Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    print(f"Loading config from {config_path}...")
    config = load_config(config_path)

    # Apply seed override from CLI
    if args.seed is not None:
        config["seed"] = args.seed

    print("Config loaded successfully.")

    # Early validation: incompatible model+defence combos (before dataset loading)
    model_type = config["model"]["type"]
    defence_type = config.get("defence", {}).get("type", "none")
    if defence_type == "adversarial_training" and model_type in ("tree", "ensemble"):
        raise ValueError(
            f"Adversarial training is not supported for {model_type} models. "
            "Use defence.type: 'input_validation' or 'none' instead."
        )

    # 1. Load Dataset
    print(f"\n[1] Loading dataset: {config['dataset']['name']}...")
    from datasets.loader import load_dataset
    from datasets.splitter import split_dataset

    dataset_config = config["dataset"].copy()
    dataset_name = dataset_config.pop("name")
    dataset = load_dataset(dataset_name, config=dataset_config)
    print(f"    Loaded {len(dataset.X)} samples with {len(dataset.feature_names)} features.")
    if "fraud_rate" in dataset.meta:
        print(f"    Fraud rate: {dataset.meta['fraud_rate']:.4f} ({dataset.meta['fraud_rate'] * 100:.2f}%)")

    # 2. Split Dataset
    seed = config.get("seed", 42)
    print(f"[2] Splitting dataset (test_size={config['dataset'].get('test_size', 0.2)}, seed={seed})...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        dataset,
        test_size=config["dataset"].get("test_size", 0.2),
        val_size=config["dataset"].get("val_size", 0.2),
        random_state=seed,
    )
    print(f"    Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # 3. Preprocessing
    print("\n[3] Preprocessing features...")
    from preprocessing.processor import DataPreprocessor, get_preprocessor_path
    import os

    dataset_name = dataset.meta.get("name", "unknown")
    n_samples = len(dataset.X)
    preprocessor_path = get_preprocessor_path(dataset_name, seed, n_samples)

    if os.path.exists(preprocessor_path):
        print("    Loading existing preprocessor...")
        preprocessor = DataPreprocessor.load(preprocessor_path)
        X_train_processed = preprocessor.transform(X_train)
    else:
        preprocessor = DataPreprocessor(dataset.feature_types)
        print("    Fitting preprocessor on Train set...")
        X_train_processed = preprocessor.fit_transform(X_train)
        preprocessor.save(preprocessor_path)

    print("    Transforming Val and Test sets...")
    _X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print(f"    Processed feature count: {X_train_processed.shape[1]}")

    # Build processed-space schema (used by both attacks and defences)
    from constraints.schema import ConstraintSchema

    processed_feature_types = {c: "numeric" for c in X_train_processed.columns}
    processed_schema = ConstraintSchema.from_data(X_train_processed, processed_feature_types)
    processed_feature_names = X_train_processed.columns.tolist()

    print("\n[4] Training Model...")
    model_params = config["model"].get("params", {})

    defence_config = config.get("defence", {})
    if defence_config.get("type") == "adversarial_training":
        print("    Configuring Adversarial Training...")
        model_params["adv_training"] = True
        model_params["adv_epsilon"] = defence_config.get("params", {}).get("epsilon", 0.1)
        model_params["adv_schema"] = processed_schema
        model_params["adv_feature_names"] = processed_feature_names
        model_params["adv_feature_types"] = processed_feature_types

    if model_type == "tree":
        from models.tree import TreeModel

        model = TreeModel(model_params)
    elif model_type == "neural":
        from models.neural import NeuralModel

        model = NeuralModel(model_params)
    elif model_type == "ensemble":
        from defences.ensemble import EnsembleModel

        model = EnsembleModel(model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"    Initializing {model_type} model with params: {model_params}")
    train_start = time.time()
    model.fit(X_train_processed, y_train)
    train_time = time.time() - train_start
    print(f"    Training complete. (Time: {train_time:.2f}s)")

    # Save model checkpoint
    model_dir = os.path.join("results", "models")
    os.makedirs(model_dir, exist_ok=True)
    ext = ".pt" if model_type == "neural" else ".joblib"
    model_path = os.path.join(model_dir, f"{config['experiment_name']}_seed{seed}{ext}")
    model.save(model_path)
    print(f"    Model saved to {model_path}")

    print("\n[5] Evaluating Model (Clean)...")
    from evaluation.metrics import compute_metrics

    # Setup Input Validation Defence early so it applies to BOTH clean and robust eval
    input_validator = None
    if defence_config.get("type") == "input_validation":
        print("    Configuring Input Validation Defence...")
        from defences.input_validation import InputValidator

        iv_params = defence_config.get("params", {})
        input_validator = InputValidator(
            processed_schema,
            mode=iv_params.get("mode", "sanitise"),
            z_threshold=iv_params.get("z_threshold", 3.0),
        )
        input_validator.fit(X_train_processed)

    # Apply input validation to clean test data too (fair comparison)
    X_test_eval = input_validator.transform(X_test_processed) if input_validator else X_test_processed
    y_probs_clean = model.predict_proba(X_test_eval)
    metrics = compute_metrics(y_test, y_probs_clean)

    print("    Test Metrics:")
    for k, v in metrics.items():
        print(f"      {k}: {v:.4f}")

    print("\n[6] Constraints setup...")
    from constraints.validator import ConstraintValidator

    print("    Inferring constraints from raw Train set...")
    schema = ConstraintSchema.from_data(X_train, dataset.feature_types)
    validator = ConstraintValidator(schema)

    print("    Validating raw Test set (should be 1.0)...")
    validity_rate = validator.validate(X_test)
    print(f"    Validity Rate: {validity_rate:.4f}")

    print("\n[7] Running Attack...")

    attack_config = config.get("attack", {})
    attack_type = attack_config.get("type", "none")

    # Determine epsilon values to sweep
    epsilon_values = attack_config.get("epsilon_values", None)
    if epsilon_values is None:
        # Backward compat: single epsilon
        epsilon_values = [attack_config.get("epsilon", 0.1)]

    from evaluation.registry import ExperimentRegistry

    registry = ExperimentRegistry()

    def _run_attack(atype, mdl, X, y, schema, ftypes, params):
        """Dispatch to the appropriate attack implementation."""
        if atype == "capgd":
            from attacks.capgd import capgd_attack

            return capgd_attack(mdl, X, y, schema, ftypes, params)
        elif atype == "hopskipjump":
            from attacks.hopskipjump import hopskipjump_attack

            return hopskipjump_attack(mdl, X, y, schema, ftypes, params)
        elif atype == "square":
            from attacks.square import square_attack

            return square_attack(mdl, X, y, schema, ftypes, params)
        else:
            raise ValueError(f"Unknown attack type: {atype}")

    if attack_type in ("capgd", "hopskipjump", "square"):
        print(f"    Attack type: {attack_type}")
        for eps in epsilon_values:
            attack_time = None
            adv_validity_rate = None

            iter_attack_config = {**attack_config, "epsilon": eps}
            print(f"\n    --- Epsilon = {eps} ---")
            print(f"    Generating adversarial examples (eps={eps})...")

            attack_start = time.time()
            X_test_adv = _run_attack(
                attack_type,
                model,
                X_test_processed,
                y_test,
                processed_schema,
                processed_feature_types,
                params=iter_attack_config,
            )
            attack_time = time.time() - attack_start
            print(f"    Attack complete. (Time: {attack_time:.2f}s)")

            # Validate adversarial samples against processed schema
            print("    Validating adversarial samples...")
            adv_validator = ConstraintValidator(processed_schema)
            adv_validity_rate = adv_validator.validate(X_test_adv)
            print(f"    Adversarial Validity Rate: {adv_validity_rate:.4f}")

            print("\n[8] Evaluating Model (Robust)...")

            # Apply Input Validation if active
            if input_validator:
                print("    Applying Input Validation Defence to Adversarial Samples...")
                X_test_adv = input_validator.transform(X_test_adv)

            y_probs_adv = model.predict_proba(X_test_adv)
            metrics_adv = compute_metrics(y_test, y_probs_adv)

            print("    Robust Test Metrics:")
            for k, v in metrics_adv.items():
                print(f"      {k}: {v:.4f}")

            print("\n    Comparison (Clean vs Robust):")
            print(f"    PR-AUC: {metrics['pr_auc']:.4f} -> {metrics_adv['pr_auc']:.4f}")

            # Log this epsilon's results
            print(f"\n[9] Logging Results (eps={eps})...")
            iter_config = {**config, "attack": iter_attack_config}
            registry.log_experiment(
                iter_config,
                metrics,
                metrics_adv,
                validity_rate,
                adv_validity_rate=adv_validity_rate,
                train_time_sec=train_time,
                attack_time_sec=attack_time,
            )

    else:
        print("    No attack configured.")
        print("\n[9] Logging Results...")
        registry.log_experiment(
            config,
            metrics,
            None,
            validity_rate,
            train_time_sec=train_time,
        )

    print("\nMVP Run Complete.")


if __name__ == "__main__":
    main()
