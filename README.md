Household Power Prediction — End-to-End ML Pipeline

Overview

This project predicts household global reactive power based on smart meter data. The system is adaptive, explainable, and production-ready, including:
Feature engineering & time-lag analysis
Neural network modeling with PyTorch Lightning
FastAPI prediction + explainability endpoints
Drift detection & automated retraining via Airflow

Architecture
Raw Household Data
       │
       ▼
Notebook FE → power_features.csv
       │
       ▼
Initial Training → best_model.pt + scaler.pkl
       │
       ▼
FastAPI API ──> Receives requests/features
       │
       ▼
Predictions + Logs → pred_requests.csv
       │
       ▼
Airflow DAG
   ├─ Drift Detection
   │     └─ if drift → retrain_model()
   └─ Retraining → Updated model/scaler
       │
       ▼
FastAPI reloads or uses new model
       │
       ▼
New Predictions (cycle continues)

Highlights:
✅ Adaptive retraining: Model updates automatically when feature drift is detected.
✅ Explainable predictions: SHAP or integrated gradients via /explain endpoint.
✅ Monitoring: Logs every prediction request for auditing and drift detection.
✅ Scalable: Supports multi-GPU training and growing datasets.

Features & Workflow
Data Processing
Parse date/time, add temporal features (hour, day-of-week, weekend).
Compute lag features for power, submetering, intensity.
Scale features and save power_features.csv.
Modeling
Train a neural network (MLP) using PyTorch Lightning.
Save best model + scaler to artifacts_lightning/.
API Deployment
/predict: returns predicted reactive power.
/explain: returns feature contributions via SHAP.
Logs every request to pred_requests.csv.
Drift Detection & Retraining
Airflow DAG runs periodically:
Checks recent predictions for distribution drift.
If drift detected, combines new data + original training set.
Retrains the model automatically and updates the API.

Quick Start
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train initial model
python train_lightning.py --data-path data/processed/power_features.csv

# 3. Launch API
uvicorn src.api.app:app --reload --port 8000

# 4. Schedule Airflow DAG for drift detection & retraining
airflow dags trigger power_model_drift_retrain

