# src/train_model.py
# Experiment 6 — Train model and log experiments to local MLflow

import mlflow
import mlflow.sklearn
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             roc_auc_score,
                             classification_report)

# ── Local MLflow tracking (no internet needed) ─────────────
mlflow.set_tracking_uri('sqlite:///mlflow_local.db')
mlflow.set_experiment('predictive_maintenance_sensors')

def load_data():
    """Load sensor data from local SQLite database."""
    conn = sqlite3.connect('mlflow_local.db')
    df = pd.read_sql('SELECT * FROM sensor_readings', conn)
    conn.close()

    features = ['temperature_c', 'vibration_hz',
                'pressure_bar', 'rpm']
    X = df[features]
    y = df['failure_within_24h']
    return X, y

def run_experiment(n_estimators=100, max_depth=5):
    """Train one model run and log everything to MLflow."""
    print(f"\nRunning experiment: n_estimators={n_estimators}, max_depth={max_depth}")

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(
            run_name=f'rf_n{n_estimators}_d{max_depth}'):

        # ── Train model ────────────────────────────────────
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=42)
        model.fit(X_train, y_train)

        # ── Predictions ────────────────────────────────────
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        # ── Calculate metrics ──────────────────────────────
        acc     = accuracy_score(y_test, preds)
        f1      = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, proba)

        # ── Log parameters ─────────────────────────────────
        mlflow.log_params({
            'n_estimators': n_estimators,
            'max_depth'   : max_depth,
            'class_weight': 'balanced'
        })

        # ── Log metrics ────────────────────────────────────
        mlflow.log_metrics({
            'accuracy': acc,
            'f1_score': f1,
            'roc_auc' : roc_auc
        })

        # ── Save model artifact ────────────────────────────
        mlflow.sklearn.log_model(model, 'failure_predictor')

        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  ROC AUC  : {roc_auc:.4f}")
        print(f"  Run logged to MLflow locally!")
        print(classification_report(y_test, preds))

if __name__ == '__main__':
    # Run 3 experiments with different hyperparameters
    run_experiment(n_estimators=100, max_depth=5)
    run_experiment(n_estimators=200, max_depth=8)
    run_experiment(n_estimators=300, max_depth=10)
    print("\nAll experiments logged! Run 'mlflow ui' to view them.")