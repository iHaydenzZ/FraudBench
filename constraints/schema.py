from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


@dataclass
class FeatureConstraint:
    name: str
    type: str  # 'numeric', 'categorical', 'binary'
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    is_non_negative: bool = False
    has_missing: bool = False  # Track if feature has missing values


@dataclass
class ConstraintSchema:
    features: Dict[str, FeatureConstraint] = field(default_factory=dict)

    @classmethod
    def from_data(cls, X: pd.DataFrame, feature_types: Dict[str, str]):
        """
        Infer constraint schema from data.

        Handles:
        - NaN/missing values in numeric columns
        - Mixed types in categorical columns
        - Empty columns
        """
        schema = cls()
        for col in X.columns:
            ftype = feature_types.get(col, 'numeric')
            has_missing = X[col].isna().any()

            if ftype == 'numeric':
                # Handle NaN by using nanmin/nanmax
                col_data = X[col].dropna()

                if len(col_data) == 0:
                    # All NaN column - use default bounds
                    min_v = 0.0
                    max_v = 1.0
                else:
                    min_v = float(col_data.min())
                    max_v = float(col_data.max())

                    # Handle edge case where min == max
                    if min_v == max_v:
                        max_v = min_v + 1.0

                    # Handle inf values
                    if np.isinf(min_v):
                        min_v = -1e10
                    if np.isinf(max_v):
                        max_v = 1e10

                schema.features[col] = FeatureConstraint(
                    name=col,
                    type='numeric',
                    min_val=min_v,
                    max_val=max_v,
                    is_non_negative=(min_v >= 0),
                    has_missing=has_missing
                )

            elif ftype in ['categorical', 'binary']:
                # Get unique values, handling mixed types safely
                unique_vals = X[col].unique()

                # Separate NaN from other values
                allowed = []
                for val in unique_vals:
                    if pd.isna(val):
                        continue  # Track via has_missing flag instead
                    allowed.append(val)

                # Sort with string conversion for mixed types
                try:
                    allowed = sorted(allowed)
                except TypeError:
                    # Mixed types that can't be compared - sort as strings
                    allowed = sorted(allowed, key=lambda x: str(x))

                schema.features[col] = FeatureConstraint(
                    name=col,
                    type=ftype,
                    allowed_values=allowed,
                    has_missing=has_missing
                )

        return schema
