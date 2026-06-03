"""Constraint-aware CAPGD with per-step projection and an optional mask.

This consolidates the projection/mask CAPGD variants that previously lived
inside the projection notebooks (`capgd_attack_g1_projected`,
`capgd_attack_ohe_projected`, `capgd_attack_m1_g1_projected`) into one reusable
function driving every Protocol C run in the ICDM grid:

    * Protocol C1 (`C1_projection`)       -> projections, no mask
    * Protocol C2 (`C2_mask_projection`)  -> projections + attacker-capability mask

The attack inner loop is byte-for-byte the stock `attacks.capgd.capgd_attack`
loop with two hooks inserted after the schema projection: (1) freeze immutable
cells to their clean value when a mask is supplied, and (2) apply each dataset
projection (OHE-block argmax snap; LCLD term snap + g1 instalment derivation).

A `Projection` carries an in-loop tensor op and an optional float64 post-loop
correction. The post hook matters only for the masked LCLD case: the attack runs
in float32, so after immutable columns are restored from the clean float64 frame
the instalment must be re-derived at full precision, otherwise ~1e-7 drift flips
the strict `<=` g3 check (the documented seed-42/456 artefact).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from attacks.capgd import project_constraints
from constraints.feasibility import get_scaler_and_num_names, normalize_dataset, reconstruct_term_from_ohe

# IEEE-CIS / Sparkov OHE blocks projected by C1 (spec §1.6 "Projected by C1").
_IEEE_OHE_PREFIXES = ["ProductCD", "card4", "card6"]
_SPARKOV_OHE_PREFIXES = ["state", "category", "gender"]


@dataclass
class Projection:
    """One per-step projection plus an optional float64 post-loop correction.

    tensor_fn: maps x_adv (B, F) -> x_adv in place; called inside torch.no_grad.
    post_fn:   maps (x_adv_df, X_clean_df) -> x_adv_df; applied once after the
               loop and after immutable columns are restored. None for OHE-only
               projections whose snapped 0/1 values survive the float cast.
    """

    tensor_fn: Callable[[torch.Tensor], torch.Tensor]
    post_fn: Optional[Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]] = None


# --------------------------------------------------------------------------- #
# Tensor projection operators (lifted verbatim from the projection notebooks)
# --------------------------------------------------------------------------- #
def project_ohe_block_tensor(x_adv: torch.Tensor, indices: List[int]) -> torch.Tensor:
    """Argmax-snap one OHE block (the winning column -> 1, the rest -> 0)."""
    block = x_adv[:, indices]
    argmax = block.argmax(dim=1)
    snapped = torch.zeros_like(block)
    snapped[torch.arange(block.size(0), device=x_adv.device), argmax] = 1.0
    x_adv[:, indices] = snapped
    return x_adv


def project_term_ohe_tensor(x_adv: torch.Tensor, term_info: dict) -> torch.Tensor:
    return project_ohe_block_tensor(x_adv, term_info["indices"])


def project_g1_tensor(x_adv: torch.Tensor, g1_info: dict, term_info: dict) -> torch.Tensor:
    """Overwrite the instalment column with its amortisation-formula value.

    Works in processed (scaled) space: un-scale loan_amnt/int_rate, read term
    from the (already-snapped) term OHE, compute the instalment in raw space,
    re-scale. Call after project_term_ohe_tensor so the term argmax is valid.
    """
    loan_raw = x_adv[:, g1_info["idx_loan"]] * g1_info["scale_loan"] + g1_info["mean_loan"]
    rate_raw = x_adv[:, g1_info["idx_rate"]] * g1_info["scale_rate"] + g1_info["mean_rate"]
    term_values = torch.as_tensor(term_info["values"], device=x_adv.device, dtype=x_adv.dtype)
    term_raw = term_values[x_adv[:, term_info["indices"]].argmax(dim=1)]
    r = rate_raw / 12.0 / 100.0
    factor = (1.0 + r) ** term_raw
    expected_raw = loan_raw * r * factor / (factor - 1.0 + 1e-12)
    x_adv[:, g1_info["idx_inst"]] = (expected_raw - g1_info["mean_inst"]) / g1_info["scale_inst"]
    return x_adv


# --------------------------------------------------------------------------- #
# Projection-info builders
# --------------------------------------------------------------------------- #
def build_ohe_block(processed_feature_names: List[str], raw_prefix: str) -> Optional[dict]:
    indices = [i for i, c in enumerate(processed_feature_names) if c.startswith(raw_prefix + "_")]
    if not indices:
        return None
    return {"prefix": raw_prefix, "indices": indices}


def build_term_proj_info(processed_feature_names: List[str]) -> dict:
    indices, values = [], []
    for i, col in enumerate(processed_feature_names):
        if not col.startswith("term_"):
            continue
        v = pd.to_numeric(col.replace("term_", "").replace("months", "").strip(), errors="coerce")
        if np.isnan(v):
            continue
        indices.append(i)
        values.append(float(v))
    if not indices:
        raise RuntimeError("No term_ OHE columns found in processed feature space")
    return {"indices": indices, "values": values}


def build_g1_proj_info(processed_feature_names: List[str], scaler, num_feature_names: List[str]) -> dict:
    def _locate(raw_name: str):
        proc_idx = processed_feature_names.index(raw_name)
        s_idx = num_feature_names.index(raw_name)
        return proc_idx, float(scaler.mean_[s_idx]), float(scaler.scale_[s_idx])

    idx_loan, m_loan, s_loan = _locate("loan_amnt")
    idx_rate, m_rate, s_rate = _locate("int_rate")
    idx_inst, m_inst, s_inst = _locate("installment")
    return {
        "idx_loan": idx_loan,
        "mean_loan": m_loan,
        "scale_loan": s_loan,
        "idx_rate": idx_rate,
        "mean_rate": m_rate,
        "scale_rate": s_rate,
        "idx_inst": idx_inst,
        "mean_inst": m_inst,
        "scale_inst": s_inst,
    }


def _lcld_g1_post(g1_info: dict) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Re-derive instalment at float64 precision after immutable restoration."""

    def post(x_adv_df: pd.DataFrame, X_clean: pd.DataFrame) -> pd.DataFrame:
        names = x_adv_df.columns.tolist()
        loan_col, rate_col, inst_col = (
            names[g1_info["idx_loan"]],
            names[g1_info["idx_rate"]],
            names[g1_info["idx_inst"]],
        )
        loan_raw = x_adv_df[loan_col].astype(float).values * g1_info["scale_loan"] + g1_info["mean_loan"]
        rate_raw = x_adv_df[rate_col].astype(float).values * g1_info["scale_rate"] + g1_info["mean_rate"]
        term_raw = reconstruct_term_from_ohe(x_adv_df).astype(float).values
        r = rate_raw / 12.0 / 100.0
        factor = (1.0 + r) ** term_raw
        expected_raw = loan_raw * r * factor / (factor - 1.0 + 1e-12)
        x_adv_df[inst_col] = (expected_raw - g1_info["mean_inst"]) / g1_info["scale_inst"]
        return x_adv_df

    return post


def build_projections(dataset_name: str, processed_feature_names: List[str], preprocessor) -> List[Projection]:
    """Per-dataset C1 projection operators (spec §1.6). Empty for CCFD."""
    key = normalize_dataset(dataset_name)

    if key == "lcld":
        scaler, num_names = get_scaler_and_num_names(preprocessor)
        term_info = build_term_proj_info(processed_feature_names)
        g1_info = build_g1_proj_info(processed_feature_names, scaler, num_names)

        def tensor_fn(x):
            x = project_term_ohe_tensor(x, term_info)
            return project_g1_tensor(x, g1_info, term_info)

        return [Projection(tensor_fn=tensor_fn, post_fn=_lcld_g1_post(g1_info))]

    if key in ("ieee_cis", "sparkov"):
        prefixes = _IEEE_OHE_PREFIXES if key == "ieee_cis" else _SPARKOV_OHE_PREFIXES
        blocks = [b for b in (build_ohe_block(processed_feature_names, p) for p in prefixes) if b is not None]
        index_lists = [b["indices"] for b in blocks]

        def tensor_fn(x):
            for idxs in index_lists:
                x = project_ohe_block_tensor(x, idxs)
            return x

        return [Projection(tensor_fn=tensor_fn)]

    return []  # ccfd: no constraints, no projection


# --------------------------------------------------------------------------- #
# The attack
# --------------------------------------------------------------------------- #
def capgd_attack_constrained(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    schema,
    feature_types: Dict[str, str],
    projections: Optional[List[Projection]] = None,
    mutable_mask: Optional[np.ndarray] = None,
    params: Dict[str, Any] = None,
) -> pd.DataFrame:
    """CAPGD with per-step constraint projection (C1) and optional mask (C2).

    With ``projections=[]`` and ``mutable_mask=None`` this reduces to the stock
    CAPGD loop. The same fitted ``model`` is used for every protocol — the mask
    enters only the attack, never training (spec §1.9.2), so weight hashes are
    preserved across A/C1/C2.
    """
    params = params or {}
    epsilon = params.get("epsilon", 0.1)
    steps = params.get("steps", 10)
    step_size = params.get("step_size", epsilon / 4)
    projections = projections or []

    if not hasattr(model, "model") or not isinstance(model.model, nn.Module):
        print("Warning: non-PyTorch model; CAPGD requires gradients. Returning clean data.")
        return X

    torch_model = model.model
    device = model.device
    torch_model.eval()

    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)
    feature_names = X.columns.tolist()

    use_mask = mutable_mask is not None
    if use_mask:
        mask_tensor = torch.as_tensor(mutable_mask, device=device, dtype=torch.float32)

    def _apply_projections(x):
        for proj in projections:
            x = proj.tensor_fn(x)
        return x

    # Initialisation: random start (masked if C2), schema clip, freeze, project.
    noise = torch.zeros_like(X_tensor).uniform_(-epsilon, epsilon)
    if use_mask:
        noise = noise * mask_tensor
    x_adv = X_tensor + noise
    x_adv = project_constraints(x_adv, X_tensor, schema, feature_names, feature_types)
    with torch.no_grad():
        if use_mask:
            x_adv = torch.where(mask_tensor.bool(), x_adv, X_tensor)
        x_adv = _apply_projections(x_adv)
    x_adv = x_adv.detach()
    x_adv.requires_grad = True

    use_logits = hasattr(model, "_use_logits") and model._use_logits
    criterion = nn.BCEWithLogitsLoss() if use_logits else nn.BCELoss()

    for _ in range(steps):
        outputs = torch_model(x_adv)
        loss = criterion(outputs, y_tensor)
        torch_model.zero_grad()
        loss.backward()
        with torch.no_grad():
            grad = x_adv.grad * mask_tensor if use_mask else x_adv.grad
            x_adv = x_adv + step_size * grad.sign()
            if epsilon > 0:
                x_adv = X_tensor + torch.clamp(x_adv - X_tensor, -epsilon, epsilon)
            x_adv = project_constraints(x_adv, X_tensor, schema, feature_names, feature_types)
            if use_mask:
                x_adv = torch.where(mask_tensor.bool(), x_adv, X_tensor)
            x_adv = _apply_projections(x_adv)
            x_adv.requires_grad = True

    x_adv_df = pd.DataFrame(x_adv.detach().cpu().numpy(), columns=feature_names, index=X.index)

    if use_mask:
        # Restore immutable columns from the clean float64 frame (kills float32
        # round-trip drift on integer-valued fields), then re-run float64 post
        # corrections (LCLD instalment) so feasibility checks match exactly.
        immutable_cols = [feature_names[i] for i in range(len(feature_names)) if not mutable_mask[i]]
        x_adv_df[immutable_cols] = X[immutable_cols].values
        for proj in projections:
            if proj.post_fn is not None:
                x_adv_df = proj.post_fn(x_adv_df, X)

    return x_adv_df


def run_capgd_protocol(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    schema,
    feature_types: Dict[str, str],
    dataset_name: str,
    protocol: str,
    preprocessor=None,
    params: Dict[str, Any] = None,
) -> pd.DataFrame:
    """Dispatch a CAPGD attack for one protocol against an already-fitted model.

    A_unconstrained -> stock CAPGD; C1_projection -> projections only;
    C2_mask_projection -> projections + the dataset's attacker-capability mask.
    `preprocessor` is required for C1/C2 (to build the projection metadata).
    """
    from attacks.capgd import capgd_attack
    from constraints.masks import get_mutable_mask

    if protocol == "A_unconstrained":
        return capgd_attack(model, X, y, schema, feature_types, params=params)

    if preprocessor is None:
        raise ValueError(f"protocol {protocol!r} needs the fitted preprocessor to build projections")

    projections = build_projections(dataset_name, X.columns.tolist(), preprocessor)
    if protocol == "C1_projection":
        return capgd_attack_constrained(model, X, y, schema, feature_types, projections=projections, params=params)
    if protocol == "C2_mask_projection":
        mask = get_mutable_mask(dataset_name, X.columns.tolist())
        return capgd_attack_constrained(
            model, X, y, schema, feature_types, projections=projections, mutable_mask=mask, params=params
        )

    raise ValueError(f"Unknown CAPGD protocol {protocol!r}")
