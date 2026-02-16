from .base import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import os

class TreeModel(BaseModel):
    def __init__(self, params=None):
        super().__init__(params)
        # Default params + overrides
        self.xgb_params = {
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "n_jobs": -1,
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": "aucpr"
        }
        if params:
            self.xgb_params.update(params)

        self.model = xgb.XGBClassifier(**self.xgb_params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Return probability of class 1
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        """Save the XGBClassifier and original params via joblib."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"model": self.model, "params": self.params}, path)

    @classmethod
    def load(cls, path: str) -> 'TreeModel':
        """Load a TreeModel from a joblib checkpoint."""
        data = joblib.load(path)
        instance = cls(data.get("params", {}))
        instance.model = data["model"]
        return instance
