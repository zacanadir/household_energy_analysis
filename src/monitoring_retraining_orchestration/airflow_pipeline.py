
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import subprocess
from scipy.stats import wasserstein_distance
import os
from pathlib import Path

# -------------------------
# Config
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_FEATURES_FILE = os.path.join(PROJECT_ROOT, "data/processed_data.csv") # original processed features
PRED_LOG_FILE = os.path.join(PROJECT_ROOT, "logs/pred_requests.csv")
PROCESSED_FILE = os.path.join(PROJECT_ROOT, "data/processed/power_features.csv")  # combined file for retraining
DRIFT_THRESHOLD = 0.05  # adjust based on experimentation
feature_cols = ['Global_active_power', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
 'Sub_total', 'Apparent_power', 'Power_factor', 'month', 'hour', 'day_of_week']

# -------------------------
# Drift Detection
# -------------------------
def detect_drift():
    # Load recent prediction logs
    if not os.path.exists(PRED_LOG_FILE):
        print("No prediction logs yet. Skipping drift check.")
        return False

    df_pred = pd.read_csv(PRED_LOG_FILE)
    if df_pred.empty:
        print("Prediction log empty. No drift detected.")
        return False
    

    # Convert string list to numeric array
    recent_features = np.vstack(df_pred['features'].apply(eval))
    
    # Load original training features
    df_train = pd.read_csv(TRAIN_FEATURES_FILE)
    train_features = df_train.drop(columns=["Global_reactive_power"]).values
    train_features = train_features[feature_cols]

    # Compute Wasserstein distance per feature
    drift_scores = [wasserstein_distance(train_features[:, i], recent_features[:, i])
                    for i in range(train_features.shape[1])]
    avg_drift = np.mean(drift_scores)
    print(f"Average drift: {avg_drift:.4f}")

    return avg_drift > DRIFT_THRESHOLD

# -------------------------
# Retraining Task
# -------------------------
def retrain_model():
    # Load original training data
    df_train = pd.read_csv(TRAIN_FEATURES_FILE)
    df_train = df_train.sample(1000)

    # Load new prediction logs
    if os.path.exists(PRED_LOG_FILE):
        df_pred = pd.read_csv(PRED_LOG_FILE)
        if not df_pred.empty:
            pred_features = pd.DataFrame(df_pred['features'].apply(eval).tolist(),
                                         columns=df_train.columns[:-1])
            # Pseudo-label: use previous predictions
            pred_features['Global_reactive_power'] = df_pred['prediction']
            df_combined = pd.concat([df_train, pred_features], ignore_index=True)
        else:
            df_combined = df_train
    else:
        df_combined = df_train

    # Save combined processed features for training
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    df_combined.to_csv(PROCESSED_FILE, index=False)
    print(f"Combined training data saved to {PROCESSED_FILE}")

    # Call training script
    subprocess.run([
        "python", "train_lightning.py",
        "--data-path", PROCESSED_FILE,
        "--epochs", "25",
        "--output-dir", "artifacts_lightning"
    ], check=True)

# -------------------------
# DAG Definition
# -------------------------
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="power_model_drift_retrain",
    default_args=default_args,
    start_date=datetime(2025, 10, 7),
    schedule_interval="*/30 * * * *",  # every 30 minutes
    catchup=False,
) as dag:

    check_drift = PythonOperator(
        task_id="check_drift",
        python_callable=detect_drift
    )

    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model
    )

    # Only retrain if drift is detected
    check_drift >> retrain
