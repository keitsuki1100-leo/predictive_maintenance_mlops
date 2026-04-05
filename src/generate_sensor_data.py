# src/generate_sensor_data.py
# Experiment 6 — Simulate IoT sensor stream and save to SQLite

import pandas as pd
import numpy as np
import sqlite3
import random

def generate_sensor_batch(machine_id, n=500):
    """Simulate one batch of high-frequency sensor readings."""
    np.random.seed(42)
    is_pre_failure = random.random() < 0.15  # 15% failure rate

    base_temp = 95 if is_pre_failure else 72
    base_vibr = 12 if is_pre_failure else 3.5

    df = pd.DataFrame({
        'machine_id'        : machine_id,
        'timestamp'         : pd.date_range(
                                'now', periods=n, freq='1min'),
        'temperature_c'     : np.random.normal(base_temp, 5,   n),
        'vibration_hz'      : np.random.normal(base_vibr, 1.2, n),
        'pressure_bar'      : np.random.normal(4.5,       0.8, n),
        'rpm'               : np.random.normal(2200,      150, n),
        'failure_within_24h': int(is_pre_failure),
    })
    return df

def ingest_to_sqlite(df, db_path='mlflow_local.db'):
    """Write sensor batch to local SQLite — air-gap safe."""
    conn = sqlite3.connect(db_path)
    df.to_sql('sensor_readings', conn,
              if_exists='append', index=False)
    conn.close()
    print(f"Logged {len(df)} rows for {df['machine_id'].iloc[0]}")

if __name__ == '__main__':
    print("Generating sensor data for 3 machines...")
    for mid in ['M001', 'M002', 'M003']:
        df = generate_sensor_batch(mid, n=500)
        ingest_to_sqlite(df)
    print("Done! Data saved to mlflow_local.db")