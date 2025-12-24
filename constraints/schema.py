from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

@dataclass
class FeatureConstraint:
    name: str
    type: str # 'numeric', 'categorical', 'binary'
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    is_non_negative: bool = False

@dataclass
class ConstraintSchema:
    features: Dict[str, FeatureConstraint] = field(default_factory=dict)
    
    @classmethod
    def from_data(cls, X: pd.DataFrame, feature_types: Dict[str, str]):
        schema = cls()
        for col in X.columns:
            ftype = feature_types.get(col, 'numeric')
            
            if ftype == 'numeric':
                min_v = float(X[col].min())
                max_v = float(X[col].max())
                # Add some buffer or just adhere to observed bounds?
                # For CAPGD, we usually want "realistic" bounds. Observed min/max is a safe start.
                schema.features[col] = FeatureConstraint(
                    name=col,
                    type='numeric',
                    min_val=min_v,
                    max_val=max_v,
                    is_non_negative=(min_v >= 0)
                )
            elif ftype in ['categorical', 'binary']:
                # For categorical, we track allowed values
                allowed = sorted(list(X[col].unique()))
                schema.features[col] = FeatureConstraint(
                    name=col,
                    type=ftype,
                    allowed_values=allowed
                )
        return schema
