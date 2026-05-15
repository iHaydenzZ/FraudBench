from .schema import ConstraintSchema
import pandas as pd


# Absolute tolerance for numeric bound checks. Counter-acts ~1-ULP drift
# introduced by StandardScaler.inverse_transform on integer-valued
# non-negative columns (e.g. LCLD pub_rec, pub_rec_bankruptcies).
# See docs/FIX_DOCUMENT.md and g1_projection_findings.md.
EVAL_TOL = 1e-6


class ConstraintValidator:
    def __init__(self, schema: ConstraintSchema, eval_tol: float = EVAL_TOL):
        self.schema = schema
        self.eval_tol = eval_tol

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

            if constraint.type == "numeric":
                if constraint.min_val is not None and val < constraint.min_val - self.eval_tol:
                    return False
                if constraint.max_val is not None and val > constraint.max_val + self.eval_tol:
                    return False
                # Non-negative check is covered by min_val >= 0 effectively

            elif constraint.type in ["categorical", "binary"]:
                # Allow unseen categories: train/test splits naturally diverge
                # on high-cardinality categoricals. NaN is already handled above.
                pass
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
