from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

@dataclass
class DatasetObj:
    X: pd.DataFrame
    y: pd.Series
    feature_types: Dict[str, str] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

def load_dummy_dataset() -> DatasetObj:
    """Creates a synthetic dataset for testing."""
    X_np, y_np = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_redundant=5, 
        random_state=42
    )
    feature_names = [f"feat_{i}" for i in range(20)]
    X = pd.DataFrame(X_np, columns=feature_names)
    y = pd.Series(y_np, name="target")
    
    # Mock feature types (all numeric for this dummy)
    feature_types = {f: "numeric" for f in feature_names}
    
    return DatasetObj(X=X, y=y, feature_types=feature_types, feature_names=feature_names, meta={"name": "dummy"})

def load_dataset(dataset_name: str, config: Optional[Dict] = None) -> DatasetObj:
    if dataset_name == "dummy_dataset":
        return load_dummy_dataset()
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented yet.")
