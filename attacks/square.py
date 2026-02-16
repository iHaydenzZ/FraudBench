"""Square Attack (score-based) black-box attack wrapper via ART."""

import math
import pandas as pd
import numpy as np
from typing import Dict, Any
from art.attacks.evasion import SquareAttack
from art.estimators.classification import BlackBoxClassifierNeuralNetwork
from constraints.schema import ConstraintSchema
from attacks.constraints_np import project_constraints_np, compute_clip_values, parse_norm


def _pad_dims(n_features):
    """Compute (h, w) for reshaping tabular features into a pseudo-image.

    SquareAttack computes tile sizes as max(round(sqrt(p * h * w)), 1) with
    p up to 0.8.  To avoid tile == h or tile == w, we need h, w >= 5.
    """
    side = max(math.ceil(math.sqrt(n_features)), 5)
    return side, side, side * side


class _SquareARTWrapper(BlackBoxClassifierNeuralNetwork):
    """
    ART wrapper that accepts 4D input (N,1,H,W) as required by SquareAttack,
    reshapes to 2D internally (stripping padding), and delegates to BaseModel.
    """

    def __init__(self, model, n_features, h, w, feature_names, clip_values=None):
        self._frbs_model = model
        self._feature_names = feature_names
        self._n_features = n_features

        clip_4d = None
        if clip_values is not None:
            c_min, c_max = clip_values
            pad_len = h * w - n_features
            c_min_pad = np.pad(c_min, (0, pad_len), constant_values=-1e10)
            c_max_pad = np.pad(c_max, (0, pad_len), constant_values=1e10)
            clip_4d = (
                c_min_pad.reshape(1, 1, h, w),
                c_max_pad.reshape(1, 1, h, w),
            )

        super().__init__(
            predict_fn=self._predict_fn,
            input_shape=(1, h, w),
            nb_classes=2,
            clip_values=clip_4d,
        )

    def _predict_fn(self, x: np.ndarray) -> np.ndarray:
        x_flat = x.reshape(x.shape[0], -1)[:, : self._n_features]
        df = pd.DataFrame(x_flat, columns=self._feature_names)
        probs = self._frbs_model.predict_proba(df)
        return np.column_stack([1 - probs, probs]).astype(np.float32)


def square_attack(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    schema: ConstraintSchema,
    feature_types: Dict[str, str],
    params: Dict[str, Any] = None,
) -> pd.DataFrame:
    """
    Run Square Attack (score-based) black-box attack. Works on any model type.

    Same signature as capgd_attack for drop-in use in the runner.
    """
    if params is None:
        params = {}

    feature_names = X.columns.tolist()
    n_features = X.shape[1]
    h, w, padded_total = _pad_dims(n_features)

    clip_min, clip_max = compute_clip_values(schema, feature_names)

    art_model = _SquareARTWrapper(
        model,
        n_features=n_features,
        h=h,
        w=w,
        feature_names=feature_names,
        clip_values=(clip_min, clip_max),
    )

    norm = parse_norm(params.get("norm", np.inf))
    epsilon = params.get("epsilon", 0.1)
    attack = SquareAttack(
        estimator=art_model,
        norm=norm,
        eps=epsilon,
        max_iter=params.get("max_iter", 100),
        verbose=False,
    )

    # Reshape to pseudo-image: (N, D) -> pad to (N, H*W) -> (N, 1, H, W)
    x_np = X.values.astype(np.float32)
    pad_len = padded_total - n_features
    if pad_len > 0:
        x_padded = np.pad(x_np, ((0, 0), (0, pad_len)), constant_values=0.0)
    else:
        x_padded = x_np
    x_4d = x_padded.reshape(-1, 1, h, w)

    y_np = y.values.astype(np.int64)
    x_adv_4d = attack.generate(x=x_4d, y=y_np)

    # Reshape back: (N, 1, H, W) -> (N, H*W) -> (N, D)
    x_adv = x_adv_4d.reshape(-1, padded_total)[:, :n_features]

    # Post-attack constraint projection
    x_adv = project_constraints_np(x_adv, x_np, schema, feature_names, feature_types)

    return pd.DataFrame(x_adv, columns=feature_names, index=X.index)
