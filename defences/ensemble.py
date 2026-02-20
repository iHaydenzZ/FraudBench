"""Ensemble defence: heterogeneous model ensemble with soft voting.

The ensemble combines LogisticRegression, XGBoost, and a 3-layer MLP.
Diversity across model families is the defensive mechanism — attacking
one component's decision boundary does not necessarily fool the others.

CAPGD compatibility: The MLP component is exposed via .model so CAPGD
can compute gradients. The full ensemble evaluates robustness via predict_proba.
"""

import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from models.base import BaseModel
from models.neural import SimpleMLP


class EnsembleModel(BaseModel):
    """Heterogeneous ensemble: LogisticRegression + XGBoost + MLP with soft voting.

    The MLP component is exposed via .model for CAPGD gradient-based attacks.
    predict_proba() averages P(fraud) from all three sub-models.
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # MLP hyperparams
        self.mlp_epochs = self.params.get("epochs", 15)
        self.mlp_hidden_dim = self.params.get("hidden_dim", 128)
        self.mlp_batch_size = self.params.get("batch_size", 256)
        self.mlp_lr = self.params.get("lr", 0.001)

        # Sub-models (initialized during fit)
        self.lr_model = None
        self.xgb_model = None
        self.mlp = None
        self.model = None  # alias for self.mlp — CAPGD compatibility
        self._use_logits = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        print("  Training Ensemble (LR + XGBoost + MLP)...")

        neg_count = int((y == 0).sum())
        pos_count = max(int((y == 1).sum()), 1)

        # 1. Logistic Regression
        print("    [1/3] LogisticRegression...")
        self.lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        self.lr_model.fit(X, y)

        # 2. XGBoost
        print("    [2/3] XGBClassifier...")
        self.xgb_model = XGBClassifier(
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1,
            scale_pos_weight=neg_count / pos_count,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="aucpr",
            n_jobs=int(os.environ.get("XGBOOST_NTHREADS", 0)) or -1,
        )
        self.xgb_model.fit(X, y)

        # 3. MLP (class-weighted loss, same approach as NeuralModel)
        print("    [3/3] SimpleMLP...")
        input_dim = X.shape[1]
        pos_weight_tensor = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self._use_logits = True

        self.mlp = SimpleMLP(input_dim, self.mlp_hidden_dim, use_sigmoid=False).to(self.device)
        self.model = self.mlp  # CAPGD reads model.model
        self.mlp.train()
        optimizer = optim.Adam(self.mlp.parameters(), lr=self.mlp_lr)

        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.mlp_batch_size, shuffle=True)

        for epoch in range(self.mlp_epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.mlp(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"      MLP Epoch {epoch + 1}/{self.mlp_epochs}, Loss: {total_loss / len(loader):.4f}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Soft voting: average P(fraud) from all three sub-models."""
        # LR
        lr_probs = self.lr_model.predict_proba(X)[:, 1]

        # XGBoost
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]

        # MLP
        self.mlp.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.mlp(X_tensor)
            if self._use_logits:
                outputs = torch.sigmoid(outputs)
        mlp_probs = outputs.cpu().numpy().flatten()

        return (lr_probs + xgb_probs + mlp_probs) / 3.0

    def save(self, path: str) -> None:
        """Save all three sub-models to a single joblib file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "lr_model": self.lr_model,
            "xgb_model": self.xgb_model,
            "mlp_state_dict": {k: v.cpu() for k, v in self.mlp.state_dict().items()},
            "mlp_config": {
                "input_dim": self.mlp.fc1.in_features,
                "hidden_dim": self.mlp_hidden_dim,
                "use_sigmoid": self.mlp.use_sigmoid,
            },
            "params": self.params,
            "_use_logits": self._use_logits,
        }
        joblib.dump(checkpoint, path)

    @classmethod
    def load(cls, path: str) -> "EnsembleModel":
        """Load an EnsembleModel from a joblib checkpoint."""
        data = joblib.load(path)
        instance = cls(data.get("params", {}))
        instance.lr_model = data["lr_model"]
        instance.xgb_model = data["xgb_model"]

        cfg = data["mlp_config"]
        instance.mlp = SimpleMLP(cfg["input_dim"], cfg["hidden_dim"], cfg["use_sigmoid"]).to(instance.device)
        state_dict = {k: v.to(instance.device) for k, v in data["mlp_state_dict"].items()}
        instance.mlp.load_state_dict(state_dict)
        instance.mlp.eval()
        instance.model = instance.mlp
        instance._use_logits = data.get("_use_logits", False)

        return instance
