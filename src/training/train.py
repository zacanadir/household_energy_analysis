#!/usr/bin/env python3
"""
train.py — Train a neural network regressor on power consumption data
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ========================
# Dataset class
# ========================

class PowerDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, scaler=None, fit_scaler=True):
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.float32).reshape(-1, 1)

        if scaler is None:
            scaler = StandardScaler()
            if fit_scaler:
                X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.scaler = scaler

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ========================
# Model class
# ========================

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


# ========================
# Training / Eval
# ========================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    preds_all, y_all = [], []
    total_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item() * X.size(0)
            preds_all.append(preds.cpu().numpy())
            y_all.append(y.cpu().numpy())

    preds_all = np.vstack(preds_all)
    y_all = np.vstack(y_all)

    mse = mean_squared_error(y_all, preds_all)
    mae = mean_absolute_error(y_all, preds_all)
    r2 = r2_score(y_all, preds_all)
    return total_loss / len(dataloader.dataset), mse, mae, r2


# ========================
# Main
# ========================

def main(args):
    # 1. Load data
    data = pd.read_csv(args.data_path)
    # For relatively quicker demo and proof of concept
    df = data.sample(10000, random_state=42)

    target_col = args.target_col
    exclude_cols = [target_col, 'datetime', 'Voltage', 'Global_active_power_lag1',
       'Global_reactive_power_lag1', 'Global_intensity_lag1',
       'Global_active_power_lag5', 'Global_reactive_power_lag5',
       'Global_intensity_lag5', 'Global_active_power_lag10',
       'Global_reactive_power_lag10', 'Global_intensity_lag10',
       'Global_active_power_lag15', 'Global_reactive_power_lag15',
       'Global_intensity_lag15']
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 2. Prepare dataset
    full_dataset = PowerDataset(df, feature_cols, target_col)
    scaler = full_dataset.scaler

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. Model / optimizer / loss
    model = MLPRegressor(input_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 5. Training loop
    best_rmse = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, mse, mae, r2 = eval_epoch(model, test_loader, criterion, device)

        rmse = np.sqrt(mse)
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val RMSE: {rmse:.4f} | R²: {r2:.4f}")

        # Save best model
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print("✅ Saved new best model.")

    # Save final scaler
    import joblib
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.pkl"))
    print(f"Training complete. Best RMSE: {best_rmse:.4f}")


# ========================
# CLI Entry Point
# ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network regressor.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to preprocessed CSV")
    parser.add_argument("--target-col", type=str, default="Global_reactive_power")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--output-dir", type=str, default="artifacts")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
