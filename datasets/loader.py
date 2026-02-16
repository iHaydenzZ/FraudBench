from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os
import pandas as pd

# Default path to external datasets (relative to project root)
DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")


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
            "source": "Kaggle Credit Card Fraud Detection",
        },
    )


def load_ieee_cis(
    data_root: str = DEFAULT_DATA_ROOT, sample_frac: float = None, use_identity: bool = False
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
    txn_path = os.path.join(data_root, "ieee-fraud-detection", "train_transaction.csv")

    if not os.path.exists(txn_path):
        raise FileNotFoundError(f"IEEE-CIS dataset not found at {txn_path}")

    df = pd.read_csv(txn_path)

    # Optionally merge identity data
    if use_identity:
        id_path = os.path.join(data_root, "ieee-fraud-detection", "train_identity.csv")
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
            "use_identity": use_identity,
        },
    )


def load_lcld(data_root: str = DEFAULT_DATA_ROOT, sample_frac: float = None) -> DatasetObj:
    """
    Loads the Lending Club Loan Default (LCLD) dataset.

    Dataset: Lending Club open data
    - ~2.26M loans, 145 raw columns
    - Target: loan_status — 'Charged Off' or 'Default' = 1 (default), else 0
    - Loans with status 'Current' are dropped (outcome unknown)
    - Post-origination/leakage columns dropped
    - High-missing (>50%) and constant columns dropped
    - ~19.6% default rate after filtering
    """
    path = os.path.join(data_root, "LCLD", "loan.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"LCLD dataset not found at {path}")

    df = pd.read_csv(path, low_memory=False)

    # --- Target engineering ---
    # Drop loans with unknown outcome
    df = df[df["loan_status"] != "Current"].copy()

    # Binary target: default = 1
    default_statuses = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
    df["target"] = df["loan_status"].isin(default_statuses).astype(int)

    # --- Column removal ---
    # IDs and non-predictive
    id_cols = ["id", "member_id", "url", "desc", "policy_code"]

    # Post-origination leakage (known only after loan is issued)
    leakage_cols = [
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "last_credit_pull_d",
        "hardship_flag",
        "hardship_type",
        "hardship_reason",
        "hardship_status",
        "deferral_term",
        "hardship_amount",
        "hardship_start_date",
        "hardship_end_date",
        "payment_plan_start_date",
        "hardship_length",
        "hardship_dpd",
        "hardship_loan_status",
        "orig_projected_additional_accrued_interest",
        "hardship_payoff_balance_amount",
        "hardship_last_payment_amount",
        "debt_settlement_flag",
        "debt_settlement_flag_date",
        "settlement_status",
        "settlement_date",
        "settlement_amount",
        "settlement_percentage",
        "settlement_term",
    ]

    # Dates (not directly usable without feature engineering)
    date_cols = ["issue_d", "earliest_cr_line"]

    # High-cardinality text
    high_card_cols = ["emp_title", "title", "zip_code"]

    # Near-constant columns
    constant_cols = ["pymnt_plan", "acc_now_delinq", "tax_liens"]

    drop_cols = set(id_cols + leakage_cols + date_cols + high_card_cols + constant_cols + ["loan_status", "target"])

    # Also drop columns with >50% missing
    miss_rate = df.drop(columns=["target"], errors="ignore").isnull().mean()
    high_miss_cols = miss_rate[miss_rate > 0.5].index.tolist()
    drop_cols.update(high_miss_cols)

    # Keep only columns that exist and aren't dropped
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].reset_index(drop=True)
    y = df["target"].reset_index(drop=True)

    # Sampling after column filtering so the >50% missing threshold is computed on full data
    if sample_frac is not None and sample_frac < 1.0:
        n = int(len(X) * sample_frac)
        idx = X.sample(n=n, random_state=42).index
        X = X.loc[idx].reset_index(drop=True)
        y = y.loc[idx].reset_index(drop=True)

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
        meta={"name": "lcld", "target_col": "loan_status", "fraud_rate": y.mean(), "source": "Lending Club open data"},
    )


def load_sparkov(data_root: str = DEFAULT_DATA_ROOT, sample_frac: float = None) -> DatasetObj:
    """
    Loads the Sparkov synthetic fraud dataset.

    Dataset: Sparkov Data Generation (Kaggle)
    - ~1.85M transactions (train + test concatenated)
    - Target: is_fraud — 0=legitimate, 1=fraud (~0.52% fraud rate)
    - PII and high-cardinality columns dropped
    - Concat train+test, then let the splitter handle the split
    """
    # Files may be nested in directories with the same name
    train_path = os.path.join(data_root, "Sparkov", "fraudTrain.csv")
    test_path = os.path.join(data_root, "Sparkov", "fraudTest.csv")

    # Handle nested directory structure (fraudTrain.csv/fraudTrain.csv)
    if os.path.isdir(train_path):
        train_path = os.path.join(train_path, "fraudTrain.csv")
    if os.path.isdir(test_path):
        test_path = os.path.join(test_path, "fraudTest.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Sparkov dataset not found at {train_path}")

    df_train = pd.read_csv(train_path)
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        df = pd.concat([df_train, df_test], ignore_index=True)
    else:
        df = df_train

    # Optional sampling (before feature selection for speed)
    if sample_frac is not None and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # --- Column removal ---
    target_col = "is_fraud"

    # Drop PII, high-cardinality, and non-predictive columns
    drop_cols = [
        "Unnamed: 0",  # index column
        "trans_date_trans_time",  # raw datetime string
        "cc_num",  # credit card number (PII)
        "first",
        "last",  # names (PII)
        "street",  # address (PII)
        "city",  # high-cardinality (~900 unique)
        "dob",  # date of birth (PII)
        "trans_num",  # transaction ID
        "merchant",  # high-cardinality (~700 unique)
        "job",  # high-cardinality (~500 unique)
        target_col,
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]

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
            "name": "sparkov",
            "target_col": target_col,
            "fraud_rate": y.mean(),
            "source": "Sparkov Data Generation (Kaggle)",
        },
    )


def load_dataset(dataset_name: str, config: Optional[Dict] = None) -> DatasetObj:
    """
    Load a dataset by name.

    Supported datasets:
        - ccfd: Credit Card Fraud Detection (Kaggle)
        - ieee_cis: IEEE-CIS Fraud Detection (Kaggle)
        - lcld: Lending Club Loan Default
        - sparkov: Sparkov Synthetic Fraud (Kaggle)

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
        return load_ieee_cis(data_root=data_root, sample_frac=sample_frac, use_identity=use_identity)

    elif dataset_name == "lcld":
        return load_lcld(data_root=data_root, sample_frac=sample_frac)

    elif dataset_name == "sparkov":
        return load_sparkov(data_root=data_root, sample_frac=sample_frac)

    else:
        raise ValueError(f"Dataset '{dataset_name}' not implemented. Available: ccfd, ieee_cis, lcld, sparkov")
