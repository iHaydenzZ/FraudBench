"""Dataset-aware feasibility evaluation for the ICDM protocol grid (spec §1.6).

This generalises the per-dataset feasibility checkers that previously lived
inside `notebooks/g1_projection_attack.ipynb` (`lcld_feasibility`) and
`notebooks/ieee_cis_ohe_projection_attack.ipynb` (`ieee_feasibility`) into a
single dataset-keyed evaluator. The shape is unchanged: each constraint is a
predicate returning a per-row boolean Series, the *aggregate feasibility* is the
fraction of rows passing the **full conjunction** of every constraint, and the
*binding* constraint is the one with the lowest pass rate.

Two things this module adds over the notebook code, both required by the spec:

1. **OHE blocks are folded into the aggregate conjunction for every dataset.**
   The old Sparkov path never folded its `state`/`category`/`gender` one-hot
   blocks (thesis Appendix G L9); doing it here, uniformly, removes the need for
   a separate "Sparkov fix" notebook.
2. **A per-row feasibility mask** is returned so callers can derive Protocol B
   (`B_posthoc_filter`): infeasible adversarial rows are reverted to clean.

Core logic (the predicates and the fold) is pure pandas/numpy and is unit-tested
on synthetic fixtures. The only I/O dependency — recovering raw-space numeric
columns from the fitted preprocessor — is isolated in `build_raw_frame`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from constraints.validator import EVAL_TOL

# OHE-block validity tolerance: a block is a valid one-hot when |sum - 1| < tol
# and |max - 1| < tol. 0.01 matches the existing projection notebooks.
TOL_OHE = 0.01

# g1 (LCLD instalment) tolerance. The amortisation formula round-trips through
# float64 scaling, so a 0.10 absolute tolerance on the dollar value is used —
# the same value the g1-projection notebook validated against.
G1_TOL = 0.10

# Sparkov merchant-location bounding box. The Sparkov generator places merchants
# at continental-US-plus-territories coordinates; merch_lat/merch_long outside
# this box are geographically impossible. These bounds are a documented, non-
# binding sanity check (the binding Sparkov constraint is `s_state_ohe`); adjust
# if a tighter geographic prior is wanted.
SPARKOV_MERCH_LAT_BOUNDS = (18.0, 72.0)
SPARKOV_MERCH_LONG_BOUNDS = (-170.0, -65.0)

# A constraint predicate maps (X_raw, X_proc) -> per-row bool Series, or None
# when the constraint's columns are absent (treated as vacuously passing).
ConstraintFn = Callable[[pd.DataFrame, pd.DataFrame], Optional["pd.Series"]]

# Canonical dataset keys. Accepts the registry casing (`IEEE-CIS`) and the
# loader casing (`ieee_cis`) alike.
_DATASET_ALIASES = {
    "ccfd": "ccfd",
    "ieee-cis": "ieee_cis",
    "ieee_cis": "ieee_cis",
    "lcld": "lcld",
    "sparkov": "sparkov",
}


def normalize_dataset(name: str) -> str:
    """Map any spelling of a dataset name onto its canonical loader key."""
    key = _DATASET_ALIASES.get(name.strip().lower())
    if key is None:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {sorted(set(_DATASET_ALIASES.values()))}")
    return key


@dataclass
class FeasibilityResult:
    """Outcome of evaluating one frame against a dataset's constraint set.

    Attributes:
        per_constraint: constraint_name -> pass rate in [0, 1].
        aggregate_feasibility: fraction of rows passing the full conjunction.
        main_failed_constraint: constraint with the lowest pass rate (the
            "binding" one when this frame is adversarial); "none" when the
            dataset has no binding constraints (CCFD) or all constraints pass.
        feasible_row_mask: per-row boolean (index-aligned to the input) — True
            where the row passes every constraint. Used to derive Protocol B.
    """

    per_constraint: Dict[str, float]
    aggregate_feasibility: float
    main_failed_constraint: str
    feasible_row_mask: pd.Series = field(repr=False)


# --------------------------------------------------------------------------- #
# Raw-space helpers (lifted verbatim from the projection notebooks)
# --------------------------------------------------------------------------- #
def _to_float(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.\-]", "", regex=True),
        errors="coerce",
    )


def get_scaler_and_num_names(preprocessor):
    """Return (fitted StandardScaler, raw numeric column names) from a preprocessor."""
    for name, transformer, columns in preprocessor.pipeline.transformers_:
        if name == "num":
            return transformer.named_steps["scaler"], list(columns)
    raise RuntimeError("No 'num' branch on the preprocessor")


def inverse_transform_numeric(X_proc: pd.DataFrame, num_feature_names: List[str], scaler) -> pd.DataFrame:
    """Recover raw-space numeric columns from processed (scaled) columns.

    Processed column names are sanitised (``[]<>`` -> ``_``) for XGBoost, so we
    re-derive the mapping before un-scaling. Columns the scaler did not see are
    silently skipped, so this is safe to call on any processed frame.
    """
    sanitize = lambda c: re.sub(r"[\[\]<>]", "_", c)  # noqa: E731
    sanitized_num = [sanitize(c) for c in num_feature_names]
    proc_cols = X_proc.columns.tolist()
    matched = [(raw, san) for raw, san in zip(num_feature_names, sanitized_num) if san in proc_cols]
    if not matched:
        return pd.DataFrame(index=X_proc.index)
    raw_names = [m[0] for m in matched]
    san_names = [m[1] for m in matched]
    idx_in_scaler = [num_feature_names.index(r) for r in raw_names]
    X_scaled = X_proc[san_names].values
    means = scaler.mean_[idx_in_scaler]
    scales = scaler.scale_[idx_in_scaler]
    return pd.DataFrame(X_scaled * scales + means, columns=raw_names, index=X_proc.index)


def reconstruct_term_from_ohe(X_proc: pd.DataFrame) -> Optional[pd.Series]:
    """LCLD: rebuild the numeric `term` (months) from its one-hot columns."""
    term_cols = [c for c in X_proc.columns if c.startswith("term_")]
    if not term_cols:
        return None
    term_vals = {}
    for col in term_cols:
        v = pd.to_numeric(col.replace("term_", "").replace("months", "").strip(), errors="coerce")
        if not np.isnan(v):
            term_vals[col] = v
    if not term_vals:
        return None
    term_df = X_proc[list(term_vals.keys())]
    return term_df.idxmax(axis=1).map(term_vals)


def build_raw_frame(dataset_name: str, X_proc: pd.DataFrame, preprocessor) -> pd.DataFrame:
    """Recover the raw-space frame the numeric predicates need.

    Inverse-transforms the scaled numeric columns and, for LCLD, restores the
    numeric `term` from its OHE block (needed by the g1 instalment formula).
    """
    scaler, num_names = get_scaler_and_num_names(preprocessor)
    X_raw = inverse_transform_numeric(X_proc, num_names, scaler)
    if normalize_dataset(dataset_name) == "lcld":
        term = reconstruct_term_from_ohe(X_proc)
        if term is not None:
            X_raw = X_raw.copy()
            X_raw["term"] = term.values
    return X_raw


# --------------------------------------------------------------------------- #
# Generic constraint predicates
# --------------------------------------------------------------------------- #
def _ohe_block_valid(X_proc: pd.DataFrame, raw_prefix: str, tol: float = TOL_OHE) -> Optional[pd.Series]:
    """True per row when the `<prefix>_*` columns form a valid one-hot vector."""
    cols = [c for c in X_proc.columns if c.startswith(raw_prefix + "_")]
    if not cols:
        return None
    ohe = X_proc[cols].values
    valid = (np.abs(ohe.sum(axis=1) - 1.0) < tol) & (np.abs(ohe.max(axis=1) - 1.0) < tol)
    return pd.Series(valid, index=X_proc.index)


def _nonneg(X_raw: pd.DataFrame, pattern: re.Pattern, floor: float = -0.5) -> Optional[pd.Series]:
    """True per row when every column matching `pattern` is >= floor."""
    cols = [c for c in X_raw.columns if pattern.match(c)]
    if not cols:
        return None
    vals = X_raw[cols].apply(_to_float).fillna(0.0)
    return (vals >= floor).all(axis=1)


def _positive(X_raw: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in X_raw.columns:
        return None
    return _to_float(X_raw[col]) > 0


def _within(X_raw: pd.DataFrame, col: str, lo: float, hi: float) -> Optional[pd.Series]:
    if col not in X_raw.columns:
        return None
    v = _to_float(X_raw[col])
    return (v >= lo - EVAL_TOL) & (v <= hi + EVAL_TOL)


_C_PATTERN = re.compile(r"^C\d+$")
_D_PATTERN = re.compile(r"^D\d+$")


# --------------------------------------------------------------------------- #
# LCLD-specific predicates (instalment formula, relational bureau bounds)
# --------------------------------------------------------------------------- #
def _lcld_g1_installment(X_raw: pd.DataFrame, X_proc: pd.DataFrame) -> Optional[pd.Series]:
    needed = {"loan_amnt", "int_rate", "term", "installment"}
    if not needed.issubset(X_raw.columns):
        return None
    loan = _to_float(X_raw["loan_amnt"])
    rate = _to_float(X_raw["int_rate"])
    term = _to_float(X_raw["term"])
    inst = _to_float(X_raw["installment"])
    r = rate / 12.0 / 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = loan * r * (1 + r) ** term / ((1 + r) ** term - 1)
    return (inst - expected).abs() < G1_TOL


def _lcld_g2_open_total(X_raw: pd.DataFrame, X_proc: pd.DataFrame) -> Optional[pd.Series]:
    if not {"open_acc", "total_acc"}.issubset(X_raw.columns):
        return None
    return _to_float(X_raw["open_acc"]) <= _to_float(X_raw["total_acc"]) + EVAL_TOL


def _lcld_g3_bankruptcies(X_raw: pd.DataFrame, X_proc: pd.DataFrame) -> Optional[pd.Series]:
    if not {"pub_rec_bankruptcies", "pub_rec"}.issubset(X_raw.columns):
        return None
    return _to_float(X_raw["pub_rec_bankruptcies"]) <= _to_float(X_raw["pub_rec"]) + EVAL_TOL


# --------------------------------------------------------------------------- #
# Per-dataset constraint registries (ordered: spec §1.6 tables)
# --------------------------------------------------------------------------- #
def _lcld_constraints() -> Dict[str, ConstraintFn]:
    return {
        "g1": _lcld_g1_installment,
        "g2": _lcld_g2_open_total,
        "g3": _lcld_g3_bankruptcies,
        "g4_term_ohe": lambda raw, proc: _ohe_block_valid(proc, "term"),
    }


def _ieee_constraints() -> Dict[str, ConstraintFn]:
    return {
        "i_product_ohe": lambda raw, proc: _ohe_block_valid(proc, "ProductCD"),
        "i_card4_ohe": lambda raw, proc: _ohe_block_valid(proc, "card4"),
        "i_card6_ohe": lambda raw, proc: _ohe_block_valid(proc, "card6"),
        "i_d_nonneg": lambda raw, proc: _nonneg(raw, _D_PATTERN),
        "i_c_nonneg": lambda raw, proc: _nonneg(raw, _C_PATTERN),
        "i_amt_positive": lambda raw, proc: _positive(raw, "TransactionAmt"),
    }


def _sparkov_constraints() -> Dict[str, ConstraintFn]:
    return {
        "s_state_ohe": lambda raw, proc: _ohe_block_valid(proc, "state"),
        "s_category_ohe": lambda raw, proc: _ohe_block_valid(proc, "category"),
        "s_gender_ohe": lambda raw, proc: _ohe_block_valid(proc, "gender"),
        "s_merch_bbox": lambda raw, proc: _sparkov_merch_bbox(raw),
        "s_city_pop_pos": lambda raw, proc: _positive(raw, "city_pop"),
        "s_amt_positive": lambda raw, proc: _positive(raw, "amt"),
    }


def _sparkov_merch_bbox(X_raw: pd.DataFrame) -> Optional[pd.Series]:
    lat = _within(X_raw, "merch_lat", *SPARKOV_MERCH_LAT_BOUNDS)
    lon = _within(X_raw, "merch_long", *SPARKOV_MERCH_LONG_BOUNDS)
    if lat is None and lon is None:
        return None
    if lat is None:
        return lon
    if lon is None:
        return lat
    return lat & lon


_REGISTRY: Dict[str, Callable[[], Dict[str, ConstraintFn]]] = {
    "ccfd": dict,  # negative control: no binding constraints (PCA features)
    "ieee_cis": _ieee_constraints,
    "lcld": _lcld_constraints,
    "sparkov": _sparkov_constraints,
}


def get_constraint_names(dataset_name: str) -> List[str]:
    """Ordered constraint names for a dataset ([] for CCFD)."""
    return list(_REGISTRY[normalize_dataset(dataset_name)]().keys())


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
def evaluate_feasibility(
    dataset_name: str,
    X_proc: pd.DataFrame,
    preprocessor=None,
    X_raw: Optional[pd.DataFrame] = None,
) -> FeasibilityResult:
    """Evaluate one frame against a dataset's full constraint conjunction.

    Args:
        dataset_name: any spelling (`LCLD`, `lcld`, `IEEE-CIS`, ...).
        X_proc: processed-space frame (post-OHE) — used by OHE-validity checks.
        preprocessor: fitted DataPreprocessor; required when `X_raw` is not given
            so raw numeric columns can be recovered for the relational checks.
        X_raw: pre-built raw-space frame (skips the preprocessor round-trip;
            used by unit tests). Must include the numeric columns and, for LCLD,
            a numeric `term` column.

    Returns:
        FeasibilityResult with per-constraint pass rates, the aggregate
        (full-conjunction) feasibility, the binding constraint, and the per-row
        feasibility mask used to derive Protocol B.
    """
    key = normalize_dataset(dataset_name)
    constraints = _REGISTRY[key]()
    idx = X_proc.index

    if X_raw is None:
        if preprocessor is None:
            X_raw = pd.DataFrame(index=idx)
        else:
            X_raw = build_raw_frame(dataset_name, X_proc, preprocessor)

    # CCFD (negative control): no constraints -> everything feasible.
    if not constraints:
        return FeasibilityResult(
            per_constraint={},
            aggregate_feasibility=1.0,
            main_failed_constraint="none",
            feasible_row_mask=pd.Series(True, index=idx),
        )

    per_constraint: Dict[str, float] = {}
    conjunction = pd.Series(True, index=idx)
    for name, fn in constraints.items():
        result = fn(X_raw, X_proc)
        # An absent constraint (None) is vacuously satisfied — it neither lowers
        # the pass rate nor narrows the conjunction.
        passes = pd.Series(True, index=idx) if result is None else result.reindex(idx).fillna(True).astype(bool)
        per_constraint[name] = float(passes.mean())
        conjunction &= passes

    # Binding = lowest pass rate. Ties resolve to the first in registry order,
    # which follows the spec's per-dataset ordering.
    min_rate = min(per_constraint.values())
    if min_rate >= 1.0:
        main_failed = "none"
    else:
        main_failed = min(per_constraint, key=lambda k: per_constraint[k])

    return FeasibilityResult(
        per_constraint=per_constraint,
        aggregate_feasibility=float(conjunction.mean()),
        main_failed_constraint=main_failed,
        feasible_row_mask=conjunction,
    )
