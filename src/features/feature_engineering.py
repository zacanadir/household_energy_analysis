import pandas as pd
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load household energy data CSV.
    """
    df = pd.read_csv(file_path, sep=';', parse_dates=[[0, 1]], 
                     infer_datetime_format=True, na_values=['?'])
    df.rename(columns={'Date_Time': 'datetime'}, inplace=True)
    df.set_index('datetime', inplace=True)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset: convert types, handle missing values.
    """
    # Convert all numeric columns to float
    numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values (forward fill)
    df.fillna(method='ffill', inplace=True)

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features:
    - Total sub-metering
    - Power factor
    - Hour of day, day of week
    - Lag features
    """
    # Total sub-metering energy (Wh)
    df['Sub_total'] = df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].sum(axis=1)

    # Power Factor
    df['Apparent_power'] = (df['Voltage'] * df['Global_intensity']) / 1000  # kVA
    df['Power_factor'] = df['Global_active_power'] / df['Apparent_power']
    df['Power_factor'] = df['Power_factor'].clip(0, 1)  # PF should be <= 1

    # Time-based features
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6

    # Lag features for temporal modeling
    lags = [1, 5, 10, 15]  # in minutes
    for lag in lags:
        df[f'Global_active_power_lag{lag}'] = df['Global_active_power'].shift(lag)
        df[f'Global_reactive_power_lag{lag}'] = df['Global_reactive_power'].shift(lag)
        df[f'Global_intensity_lag{lag}'] = df['Global_intensity'].shift(lag)
    
    # Fill any remaining NaNs after lagging
    df.fillna(method='bfill', inplace=True)
    
    return df

def preprocess(file_path: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    """
    df = load_data(file_path)
    df = clean_data(df)
    df = feature_engineering(df)
    return df

if __name__ == "__main__":
    import os
    from argparse import ArgumentParser
    from pathlib import Path
    parser = ArgumentParser()
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed_data.csv")
    parser.add_argument("--data", type=str)
    args = parser.parse_args()
    df = preprocess(args.data)
    df.to_csv(DATA_PATH)
    print(df.head())
