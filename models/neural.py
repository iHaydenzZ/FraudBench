from .base import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

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
        self.model = SimpleMLP(input_dim, self.hidden_dim).to(self.device)
        self.model.train()
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Convert to tensor
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"Training Neural Model on {self.device}...")
        
        adv_training = self.params.get("adv_training", False)
        adv_epsilon = self.params.get("adv_epsilon", 0.1)
        
        if adv_training:
            from defences.adversarial_training import adversarial_train_step
            print(f"  Enabled Adversarial Training (eps={adv_epsilon})")

        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                if adv_training:
                    loss = adversarial_train_step(self.model, X_batch, y_batch, criterion, optimizer, self.device, epsilon=adv_epsilon)
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
        return outputs.cpu().numpy().flatten()
