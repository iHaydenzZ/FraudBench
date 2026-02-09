from .base import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, use_sigmoid=True):
        super(SimpleMLP, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_logits=False):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        if return_logits or not self.use_sigmoid:
            return logits
        return self.sigmoid(logits)

class NeuralModel(BaseModel):
    def __init__(self, params=None):
        super().__init__(params)
        self.hidden_dim = self.params.get("hidden_dim", 64)
        self.lr = self.params.get("lr", 0.001)
        self.epochs = self.params.get("epochs", 10)
        self.batch_size = self.params.get("batch_size", 32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        input_dim = X.shape[1]

        # Handle class imbalance with weighted loss
        use_class_weight = self.params.get("class_weight", True)
        if use_class_weight:
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            pos_weight = neg_count / max(pos_count, 1)
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            print(f"  Using class weights (pos_weight={pos_weight:.2f})")
            self._use_logits = True
            self.model = SimpleMLP(input_dim, self.hidden_dim, use_sigmoid=False).to(self.device)
        else:
            criterion = nn.BCELoss()
            self._use_logits = False
            self.model = SimpleMLP(input_dim, self.hidden_dim, use_sigmoid=True).to(self.device)

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Convert to tensor
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"Training Neural Model on {self.device}...")
        
        adv_training = self.params.get("adv_training", False)
        adv_epsilon = self.params.get("adv_epsilon", 0.1)
        adv_schema = self.params.get("adv_schema", None)
        adv_feature_names = self.params.get("adv_feature_names", None)
        adv_feature_types = self.params.get("adv_feature_types", None)

        if adv_training:
            from defences.adversarial_training import adversarial_train_step
            print(f"  Enabled Adversarial Training (eps={adv_epsilon})")

        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                if adv_training:
                    loss = adversarial_train_step(
                        self.model, X_batch, y_batch, criterion, optimizer, self.device,
                        epsilon=adv_epsilon, schema=adv_schema,
                        feature_names=adv_feature_names, feature_types=adv_feature_types
                    )
                    total_loss += loss
                else:
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            # optional: print epoch loss
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            # Apply sigmoid if model outputs logits
            if hasattr(self, '_use_logits') and self._use_logits:
                outputs = torch.sigmoid(outputs)
        return outputs.cpu().numpy().flatten()
