import argparse
import yaml
import sys
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="FRBS MVP Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    print(f"Loading config from {config_path}...")
    config = load_config(config_path)
    
    print("Config loaded successfully.")
    
    # 1. Load Dataset
    print(f"\n[1] Loading dataset: {config['dataset']['name']}...")
    from datasets.loader import load_dataset
    from datasets.splitter import split_dataset
    
    dataset = load_dataset(config['dataset']['name'])
    print(f"    Loaded {len(dataset.X)} samples with {len(dataset.feature_names)} features.")
    
    # 2. Split Dataset
    print(f"[2] Splitting dataset (test_size={config['dataset'].get('test_size', 0.2)})...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        dataset, 
        test_size=config['dataset'].get('test_size', 0.2),
        val_size=config['dataset'].get('val_size', 0.2)
    )
    print(f"    Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # 3. Preprocessing
    print("\n[3] Preprocessing features...")
    from preprocessing.processor import DataPreprocessor
    
    preprocessor = DataPreprocessor(dataset.feature_types)
    print("    Fitting preprocessor on Train set...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    print("    Transforming Val and Test sets...")
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"    Processed feature count: {X_train_processed.shape[1]}")
    # print(f"    Sample features: {X_train_processed.columns.tolist()[:5]}")

    print("\n[4] Training Model...")
    model_type = config['model']['type']
    model_params = config['model'].get('params', {})
    
    # Check for Adversarial Training Defence
    defence_config = config.get('defence', {})
    if defence_config.get('type') == 'adversarial_training':
        print("    Configuring Adversarial Training...")
        model_params['adv_training'] = True
        model_params['adv_epsilon'] = defence_config.get('params', {}).get('epsilon', 0.1)
    
    if model_type == "tree":
        from models.tree import TreeModel
        model = TreeModel(model_params)
    elif model_type == "neural":
        from models.neural import NeuralModel
        model = NeuralModel(model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    print(f"    Initializing {model_type} model with params: {model_params}")
    model.fit(X_train_processed, y_train)
    print("    Training complete.")
    
    print("\n[5] Evaluating Model (Clean)...")
    from evaluation.metrics import compute_metrics
    y_probs_clean = model.predict_proba(X_test_processed)
    metrics = compute_metrics(y_test, y_probs_clean)
    
    print("    Test Metrics:")
    for k, v in metrics.items():
        print(f"      {k}: {v:.4f}")

    print("\n[6] Constraints setup...")
    from constraints.schema import ConstraintSchema
    from constraints.validator import ConstraintValidator
    
    # Infer schema from TRAIN (before preprocessing? Or after? 
    # Usually constraints are on the INPUT features *before* preprocessing if attacks generate raw inputs.
    # CAPGD usually generates inputs in the model space. 
    # IF preprocessing is part of the model (e.g. in pipeline), then attack generates RAW inputs.
    # IF model assumes preprocessed inputs, then attack generates preprocessed inputs.
    
    # MVP Plan: "Fit preprocessing on train only; apply to val/test." 
    # Ideally attack happens on RAW features x -> preprocessor -> model.
    # So constraints should be on RAW features.
    
    print("    Inferring constraints from raw Train set...")
    schema = ConstraintSchema.from_data(X_train, dataset.feature_types)
    validator = ConstraintValidator(schema)
    
    print("    Validating raw Test set (should be 1.0)...")
    validity_rate = validator.validate(X_test)
    print(f"    Validity Rate: {validity_rate:.4f}")


    print("\n[7] Running Attack (CAPGD)...")
    from attacks.capgd import capgd_attack
    
    # Setup Input Validation Defence if enabled
    input_validator = None
    if defence_config.get('type') == 'input_validation':
        print("    Configuring Input Validation Defence...")
        from defences.input_validation import InputValidator
        # Use schema from processed train (since attack is in processed space)
        # Note: If we use processed schema, we clip to processed bounds (approx -3 to 3).
        # This makes sense.
        fake_types = {c: 'numeric' for c in X_train_processed.columns}
        schema_processed = ConstraintSchema.from_data(X_train_processed, fake_types)
        input_validator = InputValidator(schema_processed)

    attack_config = config.get('attack', {})
    if attack_config.get('type') == 'capgd':
        print(f"    Generating adversarial examples (eps={attack_config.get('epsilon')})...")
        # Generate attacks on Test set
        # Note: CAPGD needs raw inputs?
        # If model expects preprocessed, we need to handle that.
        # Our neural model expects preprocessed inputs.
        # So we attack in PREPROCESSED space?
        # Plan says: "Constraints Layer ... standard interface ... generate(model, X, y, constraints)".
        # Ideally, we attack raw X, then preprocess, then model. (Differentiable preprocessing).
        # BUT our preprocessor uses sklearn (non-differentiable).
        # So for MVP, we assume we attack the *features* fed to the model (Post-Preprocessing).
        # This implies constraints should be on PROCESSED features.
        # BUT we inferred constraints on RAW features.
        
        # Conflict: 
        # 1. Attack Model Space (Processed) -> Schema must be for Processed.
        # 2. Attack Raw Space -> Preprocessor must be differentiable.
        
        # Resolution for MVP:
        # Attack Processed Space.
        # Re-infer schema for Processed Data?
        # Yes.
        
        print("    Inferring constraints from PROCESSED Train set for attack...")
        schema_processed = ConstraintSchema.from_data(X_train_processed, dataset.feature_types) # Feature types keys might differ?
        # Processed data usually loses 'feature_types' mapping unless we track it.
        # For dummy (all numeric) -> all numeric. 
        # Processor output is all numeric (StandardScaler + OneHot).
        # So we can assume all numeric.
        fake_types = {c: 'numeric' for c in X_train_processed.columns}
        schema_processed = ConstraintSchema.from_data(X_train_processed, fake_types)
        
        X_test_adv = capgd_attack(
            model, 
            X_test_processed, 
            y_test, 
            schema_processed, 
            fake_types,
            params=attack_config
        )
        
        print("\n[8] Evaluating Model (Robust)...")
        
        # Apply Input Validation if active
        if input_validator:
            print("    Applying Input Validation Defence to Adversarial Samples...")
            X_test_adv = input_validator.transform(X_test_adv)
            
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
    
    else:
        print("    No attack configured.")
        metrics_adv = None

    print("\n[9] Logging Results...")
    from evaluation.registry import ExperimentRegistry
    registry = ExperimentRegistry()
    registry.log_experiment(config, metrics, metrics_adv, validity_rate)

    print("\nMVP Run Complete.")

if __name__ == "__main__":
    main()
