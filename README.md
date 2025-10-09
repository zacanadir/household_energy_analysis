# Household Power Prediction — End-to-End ML Pipeline

Predict household global reactive power using smart meter data with a production-ready, adaptive ML system.

---

## Tech Stack & Tools
- Python | PyTorch Lightning | FastAPI + Uvicorn  
- Pandas / NumPy for data processing  
- SHAP / Integrated Gradients for explainability  
- Airflow for drift detection & automated retraining  
- Joblib for scaler persistence  

---

## Problem & Motivation
Short-term prediction of reactive power helps households and utilities optimize energy consumption and detect anomalies. This project demonstrates full ML engineering skills: feature engineering, model training, explainability, deployment, monitoring, and adaptive retraining.

---

## Solution Overview
1. **Data Processing & Feature Engineering**  
   - Parse timestamps, add temporal features (hour, day-of-week, weekend).  
   - Compute lag features for power, sub-metering, and intensity.  
   - Scale features and save to `power_features.csv`.

2. **Model Training**  
   - Neural network (MLP) trained with PyTorch Lightning.  
   - Best model and scaler persisted to `artifacts_lightning/`.

3. **API Deployment with FastAPI**  
   - `/predict`: returns predicted reactive power.  
   - `/explain`: returns feature contributions using SHAP.  
   - Logs every request for auditing and drift monitoring.

4. **Drift Detection & Automated Retraining**  
   - Airflow DAG checks for distribution drift in incoming prediction data.  
   - If drift detected, retrains model using new + original data.  
   - Updates API automatically with new model/scaler for continuous deployment.

---

## Project Structure
- artifacts_lightning/
       - training_logs/version_0/
       - best_model.ckpt
       - scaler.pkl
- data/
       - background.csv
       - raw_data.csv
- notebooks/
       - 01_eda_visualization.ipynb
       - 02_model_training.ipynb
- src/
       - api/
              - logs/
              - app.py
              - test_api.py
       - features/
              - feature_engineering.py
       - monitoring_retraining_orchestration/
              - airflow_pipeline.py
       - training/
              - artifacts/
                     - best_model.pt
                     - scaler.pkl
              - train.py
              - train_lightning.py
- README.md

---

## Quick Start
1. Install dependencies**
pip install -r requirements.txt

2. Train initial model
python src/train_lightning.py --data-path data/power_features.csv

3. Launch API
uvicorn src.api.app:app --reload --port 8000

4. Schedule Airflow DAG for drift detection & retraining
airflow dags trigger power_model_drift_retrain

## Highlights & Results
- Adaptive retraining: model updates automatically on feature drift
- Explainable predictions via SHAP / integrated gradients
- Logging and monitoring of every prediction request
- Scalable training and inference with PyTorch Lightning

