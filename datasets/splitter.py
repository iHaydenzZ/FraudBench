from typing import Tuple, Optional
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from .loader import DatasetObj


def get_split_path(dataset_name: str, seed: int, n_samples: int, output_dir: str = "results") -> str:
    """Returns the path for split indices file, including dataset size for cache invalidation."""
    return os.path.join(output_dir, f"split_indices_{dataset_name}_n{n_samples}_seed{seed}.json")


def save_split_indices(
    dataset_name: str,
    train_idx: list,
    val_idx: list,
    test_idx: list,
    seed: int,
    n_samples: int,
    output_dir: str = "results"
):
    """Saves split indices to JSON for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    path = get_split_path(dataset_name, seed, n_samples, output_dir)

    data = {
        "dataset": dataset_name,
        "seed": seed,
        "n_samples": n_samples,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx
    }

    with open(path, 'w') as f:
        json.dump(data, f)

    print(f"    Split indices saved to {path}")


def load_split_indices(
    dataset_name: str,
    seed: int,
    n_samples: int,
    output_dir: str = "results"
) -> Optional[dict]:
    """Loads split indices from JSON if exists and matches current dataset size."""
    path = get_split_path(dataset_name, seed, n_samples, output_dir)

    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)

        # Validate that cached indices match current dataset size
        cached_n = data.get('n_samples', 0)
        if cached_n != n_samples:
            print(f"    Warning: Cached split has {cached_n} samples but dataset has {n_samples}. Regenerating.")
            return None

        # Validate indices are within bounds
        all_indices = data['train_indices'] + data['val_indices'] + data['test_indices']
        if max(all_indices) >= n_samples:
            print(f"    Warning: Cached indices out of bounds for dataset size {n_samples}. Regenerating.")
            return None

        return data

    return None


def split_dataset(
    dataset: DatasetObj,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    save_indices: bool = True,
    output_dir: str = "results"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits dataset into Train, Validation, and Test.

    If split indices exist for this dataset+seed+size, reuses them.
    Otherwise, creates new splits and optionally saves indices.

    Cache invalidation: Indices are keyed by (dataset_name, seed, n_samples).
    Changing any of these will trigger regeneration.

    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = dataset.X
    y = dataset.y
    dataset_name = dataset.meta.get('name', 'unknown')
    n_samples = len(X)

    # Try to load existing split indices (with size validation)
    existing = load_split_indices(dataset_name, random_state, n_samples, output_dir)

    if existing is not None:
        print(f"    Reusing existing split indices for {dataset_name} (n={n_samples}, seed={random_state})")
        train_idx = existing['train_indices']
        val_idx = existing['val_indices']
        test_idx = existing['test_indices']

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        y_test = y.iloc[test_idx]

        return X_train, X_val, X_test, y_train, y_val, y_test

    # Create new split
    # First split: Train+Val vs Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Adjust val_size to be relative to temp size
    relative_val_size = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, stratify=y_temp, random_state=random_state
    )

    # Save indices if requested
    if save_indices:
        save_split_indices(
            dataset_name,
            X_train.index.tolist(),
            X_val.index.tolist(),
            X_test.index.tolist(),
            random_state,
            n_samples,
            output_dir
        )

    return X_train, X_val, X_test, y_train, y_val, y_test
