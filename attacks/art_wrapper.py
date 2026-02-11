"""ART-compatible wrapper for BaseModel instances."""
import numpy as np
import pandas as pd
from art.estimators.classification import BlackBoxClassifierNeuralNetwork
from models.base import BaseModel


class ARTModelWrapper(BlackBoxClassifierNeuralNetwork):
    """
    Wraps any BaseModel for use with ART black-box attacks.

    Converts BaseModel.predict_proba (returns shape (N,)) to the
    (N, 2) format ART expects, and handles numpy-to-DataFrame conversion.
    """

    def __init__(self, model: BaseModel, input_shape: tuple,
                 feature_names: list, clip_values: tuple = None):
        self._frbs_model = model
        self._feature_names = feature_names

        super().__init__(
            predict_fn=self._predict_fn,
            input_shape=input_shape,
            nb_classes=2,
            clip_values=clip_values,
        )

    def _predict_fn(self, x: np.ndarray) -> np.ndarray:
        """Convert model output from (N,) fraud probability to (N,2)."""
        df = pd.DataFrame(x, columns=self._feature_names)
        probs = self._frbs_model.predict_proba(df)  # shape (N,)
        return np.column_stack([1 - probs, probs]).astype(np.float32)
