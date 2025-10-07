#!/usr/bin/env python3
"""
train_lightning.py — Train a neural network regressor with PyTorch Lightning
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import joblib


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
# LightningModule
# ========================

class MLPRegressor(pl.LightningModule):
    def __init__(self, input_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        loss = self.loss_fn(preds, y)
        mse = mean_squared_error(y.cpu(), preds.cpu())
        mae = mean_absolute_error(y.cpu(), preds.cpu())
        r2 = r2_score(y.cpu(), preds.cpu())
        self.log_dict({"val_loss": loss, "val_mse": mse, "val_mae": mae, "val_r2": r2}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ========================
# DataModule
# ========================

class PowerDataModule(pl.LightningDataModule):
    def __init__(self, df, target_col, batch_size=256):
        super().__init__()
        self.df = df
        self.target_col = target_col
        self.batch_size = batch_size

    def setup(self, stage=None):
        exclude_cols = [self.target_col, 'datetime', 'Voltage', 'Global_active_power_lag1',
       'Global_reactive_power_lag1', 'Global_intensity_lag1',
       'Global_active_power_lag5', 'Global_reactive_power_lag5',
       'Global_intensity_lag5', 'Global_active_power_lag10',
       'Global_reactive_power_lag10', 'Global_intensity_lag10',
       'Global_active_power_lag15', 'Global_reactive_power_lag15',
       'Global_intensity_lag15']
        feature_cols = [c for c in self.df.columns if c not in exclude_cols]

        full_dataset = PowerDataset(self.df, feature_cols, self.target_col)
        self.scaler = full_dataset.scaler
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, test_size])

        self.input_dim = len(feature_cols)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


# ========================
# Main
# ========================

def main(args):
    df = pd.read_csv(args.data_path)

    dm = PowerDataModule(df, target_col=args.target_col, batch_size=args.batch_size)
    dm.setup()

    model = MLPRegressor(input_dim=dm.input_dim, lr=args.lr)

    logger = CSVLogger(save_dir=args.output_dir, name="training_logs")
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",  # GPU/CPU automatically
        devices="auto",
        logger=logger,
        callbacks=[checkpoint],
        log_every_n_steps=10
    )

    trainer.fit(model, dm)

    # Save scaler
    joblib.dump(dm.scaler, os.path.join(args.output_dir, "scaler.pkl"))
    print(f"✅ Training complete. Best model saved at: {checkpoint.best_model_path}")


# ========================
# CLI entry
# ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network with PyTorch Lightning")
    parser.add_argument("--data-path", type=str, required=True, help="Path to preprocessed CSV")
    parser.add_argument("--target-col", type=str, default="Global_reactive_power")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--output-dir", type=str, default="artifacts_lightning")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
