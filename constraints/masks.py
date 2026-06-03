"""Attacker-capability mutability masks for Protocol C2 (spec §1.6, §4).

A *mutable mask* is a boolean vector over processed (post-OHE) features: True
means the attacker may perturb the cell, False means it is frozen at its clean
value. Freezing immutable cells is what separates C2 (attacker-capability-aware)
from C1 (feasibility-only projection).

Both raw sets and both builders are lifted verbatim from the repo's existing
notebooks (spec §0: "reuse, do not reimplement"):

* **LCLD** uses an *immutable* set (`LCLD_IMMUTABLE_RAW`, the M1 mask from
  `notebooks/mask_ablation.ipynb`) with `build_processed_mutable_mask`.
* **IEEE-CIS** uses a *mutable* allow-list (`IEEE_CIS_MUTABLE_RAW` from
  `notebooks/ieee_cis_ohe_projection_attack.ipynb` Cell 14, per
  `docs/constraint_evaluation_guidance.md` §3.3) with the inverted builder —
  IEEE has ~380 opaque immutable columns (V*/C*/D*/card*/…), so only the six
  attacker-controllable transaction-facing fields are enumerated. This is the
  set that produced the thesis Table 12 / spec §4.4 anchor (robust PR-AUC
  ≈0.40, robust accuracy ≈0.88, ~7.7 feasible-flipped). NOTE: spec §1.6's
  "(IEEE-CIS: D-fields)" is an inaccurate shorthand — the D-fields are merely
  the block whose non-negativity C1 leaves broken; the C2 mask freezes far more.
* **Sparkov** is not specified by the spec (§1.6 names only LCLD and IEEE, yet
  §4.2 runs C2 on every constrained dataset). The default below freezes victim
  geography/identity (state, coordinates, city population, gender, merchant
  location) on the premise that a fraudster controls the *transaction* (amount,
  category, time) but not the victim's location or demographics.
  TODO(debt): confirm this Sparkov immutable set with the experiment owner —
  it has no golden anchor in §4.4 and is a domain judgement.
"""

from __future__ import annotations

from typing import List, Set

import numpy as np

from constraints.feasibility import normalize_dataset

# --- LCLD M1 mask (verbatim from mask_ablation.ipynb / tabularbench_comparison) ---
LCLD_IMMUTABLE_RAW: Set[str] = {
    # LC internal pricing / grading
    "grade",
    "sub_grade",
    "int_rate",
    "installment",
    "funded_amnt",
    "funded_amnt_inv",
    "initial_list_status",
    # LC verification outcomes
    "verification_status",
    "verification_status_joint",
    # Credit bureau data
    "delinq_2yrs",
    "inq_last_6mths",
    "inq_last_12m",
    "inq_fi",
    "open_acc",
    "open_acc_6m",
    "open_act_il",
    "open_il_12m",
    "open_il_24m",
    "open_rv_12m",
    "open_rv_24m",
    "pub_rec",
    "pub_rec_bankruptcies",
    "total_acc",
    "revol_bal",
    "revol_util",
    "il_util",
    "all_util",
    "tot_cur_bal",
    "tot_hi_cred_lim",
    "total_bal_il",
    "total_rev_hi_lim",
    "max_bal_bc",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "collections_12_mths_ex_med",
    "mths_since_last_delinq",
    "mths_since_last_il_delinq",
    "mths_since_last_major_delinq",
    "mths_since_last_record",
    "mths_since_rcnt_il",
    "payment_inc_ratio",
}

# --- Sparkov immutable set (documented default; see module docstring) ---
SPARKOV_IMMUTABLE_RAW: Set[str] = {
    "state",
    "lat",
    "long",
    "city_pop",
    "gender",
    "merch_lat",
    "merch_long",
    "zip",
}

# --- IEEE-CIS mutable allow-list (verbatim, ieee_cis_ohe_projection_attack.ipynb Cell 14) ---
# Defined as the (small) mutable set because IEEE-CIS has hundreds of opaque
# immutable features; the six below are the attacker-controllable, transaction-
# facing fields. ProductCD is OHE-expanded and matched by prefix.
IEEE_CIS_MUTABLE_RAW: Set[str] = {
    "TransactionAmt",
    "ProductCD",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
}


def build_processed_mutable_mask(processed_feature_names: List[str], immutable_raw: Set[str]) -> np.ndarray:
    """Boolean mask over processed features. True = mutable, False = immutable.

    A processed column is frozen when it matches an immutable raw name directly
    (numeric features keep their name) or via the OHE prefix that produced it
    (e.g. ``verification_status_Verified`` -> raw ``verification_status``). The
    longest matching prefix wins, mirroring the one-hot expansion.
    """
    mask = np.ones(len(processed_feature_names), dtype=bool)  # default: mutable
    for i, col in enumerate(processed_feature_names):
        if col in immutable_raw:
            mask[i] = False
            continue
        parts = col.split("_")
        for k in range(1, len(parts)):
            if "_".join(parts[:k]) in immutable_raw:
                mask[i] = False
                break
    return mask


def build_processed_mutable_mask_inverted(processed_feature_names: List[str], mutable_raw: Set[str]) -> np.ndarray:
    """Boolean mask built from the (smaller) *mutable* set; True = mutable.

    The inverse convention of `build_processed_mutable_mask`, for datasets where
    enumerating the immutable features is impractical (IEEE-CIS: ~380 columns).
    A processed column is mutable only when it matches a mutable raw name
    directly or via its OHE prefix; everything else is frozen.
    """
    mask = np.zeros(len(processed_feature_names), dtype=bool)  # default: immutable
    for i, col in enumerate(processed_feature_names):
        if col in mutable_raw:
            mask[i] = True
            continue
        parts = col.split("_")
        for k in range(1, len(parts)):
            if "_".join(parts[:k]) in mutable_raw:
                mask[i] = True
                break
    return mask


def get_mutable_mask(dataset_name: str, processed_feature_names: List[str]) -> np.ndarray:
    """Dispatch the C2 mutable mask for a dataset.

    CCFD has no C2 (PCA features, no constraints) and returns an all-mutable
    mask for completeness. IEEE-CIS uses the mutable allow-list (inverted
    builder); LCLD and Sparkov use named immutable sets.
    """
    key = normalize_dataset(dataset_name)
    if key == "lcld":
        return build_processed_mutable_mask(processed_feature_names, LCLD_IMMUTABLE_RAW)
    if key == "sparkov":
        return build_processed_mutable_mask(processed_feature_names, SPARKOV_IMMUTABLE_RAW)
    if key == "ieee_cis":
        return build_processed_mutable_mask_inverted(processed_feature_names, IEEE_CIS_MUTABLE_RAW)
    # ccfd or anything else: all mutable.
    return np.ones(len(processed_feature_names), dtype=bool)
