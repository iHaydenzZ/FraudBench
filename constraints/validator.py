from .schema import ConstraintSchema
import pandas as pd
import numpy as np

class ConstraintValidator:
    def __init__(self, schema: ConstraintSchema):
        self.schema = schema

    def validate_sample(self, x: pd.Series) -> bool:
        """Validates a single sample."""
        for col, constraint in self.schema.features.items():
            if col not in x:
                continue

            val = x[col]

            # Handle NaN/missing values upfront
            val_is_nan = pd.isna(val)
            if val_is_nan:
                if constraint.has_missing:
                    continue  # NaN is expected for this feature
                else:
                    return False  # NaN not expected

            if constraint.type == 'numeric':
                if constraint.min_val is not None and val < constraint.min_val:
                    return False
                if constraint.max_val is not None and val > constraint.max_val:
                    return False
                # Non-negative check is covered by min_val >= 0 effectively

            elif constraint.type in ['categorical', 'binary']:
                if val not in constraint.allowed_values:
                    return False
        return True

    def validate(self, X: pd.DataFrame) -> float:
        """Returns validity rate (0.0 to 1.0)."""
        valid_count = 0
        total = len(X)
        if total == 0:
             return 1.0
             
        # iterate for now, verify vectorization later if slow
        for i in range(total):
            if self.validate_sample(X.iloc[i]):
                valid_count += 1
                
        return valid_count / total
