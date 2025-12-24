import pandas as pd
import numpy as np
from constraints.schema import ConstraintSchema

class InputValidator:
    def __init__(self, schema: ConstraintSchema):
        self.schema = schema

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Projects samples to feasible region (clipping).
        """
        X_clean = X.copy()
        for col, constraint in self.schema.features.items():
            if col not in X_clean.columns:
                continue
                
            if constraint.type == 'numeric':
                min_v = constraint.min_val if constraint.min_val is not None else -float('inf')
                max_v = constraint.max_val if constraint.max_val is not None else float('inf')
                X_clean[col] = X_clean[col].clip(lower=min_v, upper=max_v)
                
            elif constraint.type in ['categorical', 'binary']:
                # Input validation implies REJECTING or FIXING.
                # Fixing is hard without probability.
                # Simple defense: if invalid, map to mode or generic 'other'?
                # For MVP: simple clipping is main thing.
                # If we encounter invalid categorical, we leave it or replace with mode.
                pass
                
        return X_clean
