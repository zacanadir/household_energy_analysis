from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Optional
from datetime import datetime
from pathlib import Path

# =======================
# 1- Define request schema
# =======================
class PredictRequest(BaseModel):
    features: List[float]  # Flattened feature vector in same order as training
    timestamp: Optional[str] = None  # optional request timestamp

# =======================
# 2- Define response schema
# =======================
class PredictResponse(BaseModel):
    prediction: float
    timestamp: str
    model_version: str

# =======================
# 3- FastAPI app
# =======================
app = FastAPI(title="Household Power Prediction API")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = os.path.join(PROJECT_ROOT, "src/training/artifacts/best_model.pt")  # Lightning checkpoint or .pt for raw PyTorch
SCALER_PATH = os.path.join(PROJECT_ROOT, "src/training/artifacts/scaler.pkl")
MODEL_VERSION = "v1"

# =======================
# 4- Load model & scaler
# =======================
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self, x):
        return self.model(x)

# Load scaler
scaler = joblib.load(SCALER_PATH)

# Define input_dim (must match feature length)
INPUT_DIM = scaler.mean_.shape[0]
model = MLPRegressor(INPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# =======================
# 5- Logging setup
# =======================
LOG_FILE = "logs/pred_requests.csv"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "features", "prediction", "model_version"]).to_csv(LOG_FILE, index=False)

# =======================
# 6- Prediction endpoint
# =======================
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        # Convert features to numpy array and scale
        X = np.array(req.features, dtype=np.float32).reshape(1, -1)
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            pred = model(X_tensor).item()

        # Timestamp
        ts = req.timestamp if req.timestamp else datetime.now().isoformat()

        # Log request
        log_df = pd.DataFrame([{
            "timestamp": ts,
            "features": X.tolist(),
            "prediction": pred,
            "model_version": MODEL_VERSION
        }])
        log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)

        return PredictResponse(prediction=pred, timestamp=ts, model_version=MODEL_VERSION)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =======================
# 7- SHAP setup
# =======================
import shap

# Prepare a small background dataset for SHAP (sample from training)
# Note: For large datasets, pick 100-500 random rows

# At startup
background = pd.read_csv(os.path.join(PROJECT_ROOT, "data/background.csv"))
background_scaled = scaler.transform(background)
background_tensor = torch.tensor(background_scaled, dtype=torch.float32)
explainer = shap.DeepExplainer(model, background_tensor)

# =======================
# 8️⃣ Explain endpoint
# =======================
class ExplainResponse(BaseModel):
    feature_contributions: List[float]
    prediction: float
    timestamp: str
    model_version: str

@app.post("/explain", response_model=ExplainResponse)
def explain(req: PredictRequest):
    try:
        # Convert features to numpy array and scale
        X = np.array(req.features, dtype=np.float32).reshape(1, -1)
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            pred = model(X_tensor).item()

        # Compute SHAP values
        shap_values = explainer.shap_values(X_tensor)[0]  # returns array of shape (1, n_features)
        shap_values_list = shap_values.tolist()

        # Timestamp
        ts = req.timestamp if req.timestamp else datetime.utcnow().isoformat()

        # Optionally log explanation requests
        log_df = pd.DataFrame([{
            "timestamp": ts,
            "features": X.tolist(),
            "prediction": pred,
            "model_version": MODEL_VERSION,
            "shap_values": shap_values_list
        }])
        log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)

        flat_shap = np.array(shap_values).flatten().tolist()
        return ExplainResponse(
            feature_contributions=flat_shap,
            prediction=float(pred),
            timestamp=ts,
            model_version=MODEL_VERSION
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
