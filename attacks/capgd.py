import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, Any
from constraints.schema import ConstraintSchema


def project_constraints(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    schema: ConstraintSchema,
    feature_names: list,
    feature_types: Dict[str, str],
):
    """
    Projects adversarial examples back into the feasible domain defined by the schema.
    """
    # x_adv is (batch, features)
    # We iterate over features and clip.
    # Note: Vectorization is faster but requires aligning schema with tensor indices.

    # Pre-compute bounds vectors if possible.
    # For now, simplistic iteration or creating a mask.

    # 1. Clip to min/max bounds
    # We construct min/max tensors

    # Optimization: cache these tensors? For MVP, just rebuild.
    min_vals = []
    max_vals = []

    # We need to map feature index to schema
    # Use feature_names list from dataset

    for i, fname in enumerate(feature_names):
        constraint = schema.features.get(fname)
        if constraint and constraint.type == "numeric":
            min_v = constraint.min_val if constraint.min_val is not None else -float("inf")
            max_v = constraint.max_val if constraint.max_val is not None else float("inf")
            min_vals.append(min_v)
            max_vals.append(max_v)
        else:
            # Categorical/Binary or missing constraint: keep original value?
            # Or allow unbounded?
            # For categorical one-hot, projection is harder (softmax/argmax).
            # For MVP, assume we only perturb NUMERIC features.
            # So for non-numeric, we reset to x_orig.
            min_vals.append(float("nan"))
            max_vals.append(float("nan"))

    # Convert to tensor
    # Handling NaNs is tricky.
    # Approach:
    #   create numeric mask
    #   clip numeric columns
    #   reset non-numeric columns to x_orig

    # Slow Loop implementation for correctness first
    x_adv_clone = x_adv.clone()

    for i, fname in enumerate(feature_names):
        constraint = schema.features.get(fname)
        if constraint and constraint.type == "numeric":
            min_v = constraint.min_val if constraint.min_val is not None else -float("inf")
            max_v = constraint.max_val if constraint.max_val is not None else float("inf")
            x_adv_clone[:, i] = torch.clamp(x_adv_clone[:, i], min=min_v, max=max_v)
        else:
            # Revert to original for non-numeric
            x_adv_clone[:, i] = x_orig[:, i]

    return x_adv_clone


def capgd_attack(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    schema: ConstraintSchema,
    feature_types: Dict[str, str],
    params: Dict[str, Any] = None,
) -> pd.DataFrame:
    """
    Constrained APGD attack.

    Args:
        model: Must expose .model (PyTorch module)
        X: Input DataFrame (Clean)
        y: Target label Series
        schema: Constraints
    """
    if params is None:
        params = {}

    epsilon = params.get("epsilon", 0.1)
    steps = params.get("steps", 10)
    step_size = params.get("step_size", epsilon / 4)

    # Check if model supports torch
    if not hasattr(model, "model") or not isinstance(model.model, nn.Module):
        print("Warning: Model does not appear to be a PyTorch model. CAPGD requires gradients. Skipping.")
        return X

    torch_model = model.model
    device = model.device
    torch_model.eval()

    # Prepare Data
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)
    feature_names = X.columns.tolist()

    # Initialize x_adv with random start
    noise = torch.zeros_like(X_tensor).uniform_(-epsilon, epsilon)
    x_adv = X_tensor + noise
    x_adv = project_constraints(x_adv, X_tensor, schema, feature_names, feature_types)
    x_adv = x_adv.detach()
    x_adv.requires_grad = True

    # Check if model outputs logits (uses BCEWithLogitsLoss during training)
    use_logits = hasattr(model, "_use_logits") and model._use_logits
    if use_logits:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCELoss()

    for step in range(steps):
        outputs = torch_model(x_adv)

        # Loss: We want to MAXIMIZE loss (Gradient Acent) to cause misclassification.
        # But wait, untargeted attack: Maximize Loss(pred, y).
        # Sample y=1, pred=0.9. Loss low. Maximize loss -> pred ~ 0.
        # Sample y=0, pred=0.1. Loss low. Maximize loss -> pred ~ 1.
        loss = criterion(outputs, y_tensor)

        torch_model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = x_adv.grad
            # PGD update: x + step * sign(grad)
            x_adv = x_adv + step_size * grad.sign()

            # Projection 1: Epsilon ball (L-inf) around X_orig (optional, usually CAPGD has budget)
            if epsilon > 0:
                delta = x_adv - X_tensor
                delta = torch.clamp(delta, -epsilon, epsilon)
                x_adv = X_tensor + delta

            # Projection 2: Feasibility Constraints
            x_adv = project_constraints(x_adv, X_tensor, schema, feature_names, feature_types)

            x_adv.requires_grad = True

    # Wrap result
    return pd.DataFrame(x_adv.detach().cpu().numpy(), columns=feature_names, index=X.index)
