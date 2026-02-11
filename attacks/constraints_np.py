"""NumPy-based constraint projection for black-box attacks (ART compatibility)."""
import numpy as np
from typing import Dict, Tuple
from constraints.schema import ConstraintSchema


def project_constraints_np(
    x_adv: np.ndarray,
    x_orig: np.ndarray,
    schema: ConstraintSchema,
    feature_names: list,
    feature_types: Dict[str, str],
) -> np.ndarray:
    """
    Projects adversarial examples back into the feasible domain.

    NumPy mirror of attacks.capgd.project_constraints (PyTorch version).
    Clips numeric features to schema bounds and reverts non-numeric features
    to their original values.
    """
    x_proj = x_adv.copy()

    for i, fname in enumerate(feature_names):
        constraint = schema.features.get(fname)
        if constraint and constraint.type == "numeric":
            min_v = constraint.min_val if constraint.min_val is not None else -np.inf
            max_v = constraint.max_val if constraint.max_val is not None else np.inf
            x_proj[:, i] = np.clip(x_proj[:, i], min_v, max_v)
        else:
            # Revert non-numeric features to original
            x_proj[:, i] = x_orig[:, i]

    return x_proj


def compute_clip_values(
    schema: ConstraintSchema, feature_names: list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute global (min, max) clip arrays for ART classifiers.

    Returns:
        (clip_min, clip_max) each of shape (n_features,)
    """
    mins = []
    maxs = []
    for fname in feature_names:
        constraint = schema.features.get(fname)
        if constraint and constraint.type == "numeric":
            mins.append(constraint.min_val if constraint.min_val is not None else -1e10)
            maxs.append(constraint.max_val if constraint.max_val is not None else 1e10)
        else:
            mins.append(-1e10)
            maxs.append(1e10)

    return np.array(mins, dtype=np.float32), np.array(maxs, dtype=np.float32)


def parse_norm(norm_value) -> float:
    """Convert YAML norm value (e.g. 'inf' string) to float."""
    if isinstance(norm_value, str) and norm_value.lower() == "inf":
        return np.inf
    return float(norm_value)
