"""HopSkipJump black-box attack wrapper via ART."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from art.attacks.evasion import HopSkipJump
from constraints.schema import ConstraintSchema
from attacks.art_wrapper import ARTModelWrapper
from attacks.constraints_np import project_constraints_np, compute_clip_values, parse_norm


def hopskipjump_attack(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    schema: ConstraintSchema,
    feature_types: Dict[str, str],
    params: Dict[str, Any] = None,
) -> pd.DataFrame:
    """
    Run HopSkipJump (decision-based) attack. Works on any model type.

    Same signature as capgd_attack for drop-in use in the runner.
    """
    if params is None:
        params = {}

    feature_names = X.columns.tolist()
    clip_min, clip_max = compute_clip_values(schema, feature_names)
    clip_values = (clip_min, clip_max)

    art_model = ARTModelWrapper(
        model,
        input_shape=(X.shape[1],),
        feature_names=feature_names,
        clip_values=clip_values,
    )

    norm = parse_norm(params.get("norm", np.inf))
    attack = HopSkipJump(
        classifier=art_model,
        targeted=False,
        norm=norm,
        max_iter=params.get("max_iter", 20),
        max_eval=params.get("max_eval", 1000),
        init_eval=params.get("init_eval", 100),
        verbose=False,
    )

    x_np = X.values.astype(np.float32)
    x_adv = attack.generate(x=x_np)

    # Post-attack constraint projection
    x_adv = project_constraints_np(x_adv, x_np, schema, feature_names, feature_types)

    return pd.DataFrame(x_adv, columns=feature_names, index=X.index)
