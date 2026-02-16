import pandas as pd
from constraints.schema import ConstraintSchema


class InputValidator:
    def __init__(self, schema: ConstraintSchema, mode: str = "sanitise", z_threshold: float = 3.0):
        """
        Args:
            schema: Feature constraint schema for bound clipping.
            mode: 'sanitise' (clip outliers) or 'reject' (mark outliers as NaN).
            z_threshold: Z-score threshold for outlier detection. Only active after fit().
        """
        self.schema = schema
        self.mode = mode
        self.z_threshold = z_threshold
        self._fitted = False
        self._means = {}
        self._stds = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Compute per-feature mean/std for outlier detection."""
        for col, constraint in self.schema.features.items():
            if col not in X.columns or constraint.type != "numeric":
                continue
            self._means[col] = X[col].mean()
            self._stds[col] = X[col].std()
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame, return_metadata: bool = False):
        """
        Projects samples to feasible region.

        If fitted, also applies outlier detection (clipping or rejection).
        """
        X_clean = X.copy()
        n_rejected = 0
        rejected_mask = pd.Series(False, index=X_clean.index)

        for col, constraint in self.schema.features.items():
            if col not in X_clean.columns:
                continue

            if constraint.type == "numeric":
                # 1. Bound clipping (always)
                min_v = constraint.min_val if constraint.min_val is not None else -float("inf")
                max_v = constraint.max_val if constraint.max_val is not None else float("inf")
                X_clean[col] = X_clean[col].clip(lower=min_v, upper=max_v)

                # 2. Outlier detection (only if fitted)
                if self._fitted and col in self._means and self._stds.get(col, 0) > 0:
                    mean = self._means[col]
                    std = self._stds[col]
                    lower_bound = mean - self.z_threshold * std
                    upper_bound = mean + self.z_threshold * std

                    if self.mode == "sanitise":
                        X_clean[col] = X_clean[col].clip(lower=lower_bound, upper=upper_bound)
                    elif self.mode == "reject":
                        outlier_mask = (X_clean[col] < lower_bound) | (X_clean[col] > upper_bound)
                        rejected_mask = rejected_mask | outlier_mask

        if self.mode == "reject" and rejected_mask.any():
            n_rejected = rejected_mask.sum()

        if return_metadata:
            metadata = {
                "n_rejected": int(n_rejected),
                "total": len(X),
                "rejection_rate": n_rejected / len(X) if len(X) > 0 else 0.0,
            }
            return X_clean, metadata

        return X_clean
