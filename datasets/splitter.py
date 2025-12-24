from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from .loader import DatasetObj

def split_dataset(dataset: DatasetObj, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits dataset into Train, Validation, and Test.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = dataset.X
    y = dataset.y
    
    # First split: Train+Val vs Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Adjust val_size to be relative to the original size or temp size? 
    # Usually val_size is fraction of total or fraction of train.
    # Let's assume input val_size is fraction of TOTAL.
    # So if test=0.2, val=0.2, then train=0.6.
    # The size of temp is 0.8. We need 0.2/0.8 = 0.25 of temp for val.
    
    relative_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, stratify=y_temp, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
