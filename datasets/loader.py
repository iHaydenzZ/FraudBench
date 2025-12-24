from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import os
import pandas as pd
import numpy as np

# Default path to external datasets
DEFAULT_DATA_ROOT = "/Users/xitong/Local_Document/AAA_CAPSTONE/datasets"


@dataclass
class DatasetObj:
    X: pd.DataFrame
    y: pd.Series
    feature_types: Dict[str, str] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


def load_ccfd(data_root: str = DEFAULT_DATA_ROOT, sample_frac: float = None) -> DatasetObj:
    """
    Loads the Credit Card Fraud Detection (CCFD) dataset.

    Dataset: Kaggle Credit Card Fraud Detection
    - 284,807 transactions, 31 features
    - V1-V28: PCA components (anonymized)
    - Time: seconds elapsed from first transaction
    - Amount: transaction amount
    - Class: 0=legitimate, 1=fraud (0.17% fraud rate)
    """
    path = os.path.join(data_root, "CCFD", "creditcard.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"CCFD dataset not found at {path}")

    df = pd.read_csv(path)

    # Optional sampling for faster iteration
    if sample_frac is not None and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # Separate features and target
    target_col = "Class"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    # All features are numeric in CCFD
    feature_types = {col: "numeric" for col in feature_cols}

    return DatasetObj(
        X=X,
        y=y,
        feature_types=feature_types,
        feature_names=feature_cols,
        meta={
            "name": "ccfd",
            "target_col": target_col,
            "fraud_rate": y.mean(),
            "source": "Kaggle Credit Card Fraud Detection"
        }
    )


def load_ieee_cis(
    data_root: str = DEFAULT_DATA_ROOT,
    sample_frac: float = None,
    use_identity: bool = False
) -> DatasetObj:
    """
    Loads the IEEE-CIS Fraud Detection dataset.

    Dataset: Kaggle IEEE-CIS Fraud Detection
    - 590,540 transactions, 394 features (transaction only)
    - Mix of numeric and categorical features
    - isFraud: 0=legitimate, 1=fraud (3.5% fraud rate)
    - High missing value rate in many columns

    Args:
        data_root: Root path to datasets
        sample_frac: Optional sampling fraction for faster iteration
        use_identity: Whether to merge identity data (adds more features but more missing)
    """
    txn_path = os.path.join(data_root, "IEEE-CIS", "ieee-fraud-detection", "train_transaction.csv")

    if not os.path.exists(txn_path):
        raise FileNotFoundError(f"IEEE-CIS dataset not found at {txn_path}")

    df = pd.read_csv(txn_path)

    # Optionally merge identity data
    if use_identity:
        id_path = os.path.join(data_root, "IEEE-CIS", "ieee-fraud-detection", "train_identity.csv")
        if os.path.exists(id_path):
            df_id = pd.read_csv(id_path)
            df = df.merge(df_id, on="TransactionID", how="left")

    # Optional sampling
    if sample_frac is not None and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # Separate features and target
    target_col = "isFraud"
    id_col = "TransactionID"
    exclude_cols = [target_col, id_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df[target_col]

    # Determine feature types
    feature_types = {}
    for col in feature_cols:
        if X[col].dtype == "object":
            feature_types[col] = "categorical"
        else:
            feature_types[col] = "numeric"

    return DatasetObj(
        X=X,
        y=y,
        feature_types=feature_types,
        feature_names=feature_cols,
        meta={
            "name": "ieee_cis",
            "target_col": target_col,
            "fraud_rate": y.mean(),
            "source": "Kaggle IEEE-CIS Fraud Detection",
            "use_identity": use_identity
        }
    )


def load_dataset(dataset_name: str, config: Optional[Dict] = None) -> DatasetObj:
    """
    Load a dataset by name.

    Supported datasets:
        - ccfd: Credit Card Fraud Detection (Kaggle)
        - ieee_cis: IEEE-CIS Fraud Detection (Kaggle)

    Args:
        dataset_name: Name of the dataset to load
        config: Optional config dict with dataset-specific options
            - sample_frac: Fraction of data to sample (for faster iteration)
            - data_root: Override default data root path
            - use_identity: For IEEE-CIS, whether to include identity data
    """
    config = config or {}
    data_root = config.get("data_root", DEFAULT_DATA_ROOT)
    sample_frac = config.get("sample_frac", None)

    if dataset_name == "ccfd":
        return load_ccfd(data_root=data_root, sample_frac=sample_frac)

    elif dataset_name == "ieee_cis":
        use_identity = config.get("use_identity", False)
        return load_ieee_cis(
            data_root=data_root,
            sample_frac=sample_frac,
            use_identity=use_identity
        )

    else:
        raise ValueError(
            f"Dataset '{dataset_name}' not implemented. "
            f"Available: ccfd, ieee_cis"
        )
