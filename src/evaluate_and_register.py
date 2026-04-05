# src/evaluate_and_register.py
# Experiment 8 — Evaluate model and register to MLflow Registry

import mlflow
import mlflow.sklearn
import sqlite3
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient

# ── Local MLflow tracking ───────────────────────────────────
mlflow.set_tracking_uri('sqlite:///mlflow_local.db')
client = MlflowClient()

# ── Production thresholds ───────────────────────────────────
THRESHOLDS = {
    'accuracy_score': 0.92,
    'f1_score'      : 0.88,
    'roc_auc'       : 0.95,
}

def load_test_data():
    """Load test data from local SQLite database."""
    conn = sqlite3.connect('mlflow_local.db')
    df = pd.read_sql('SELECT * FROM sensor_readings', conn)
    conn.close()

    features = ['temperature_c', 'vibration_hz',
                'pressure_bar', 'rpm']
    X = df[features]
    y = df['failure_within_24h']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    test_df = X_test.copy()
    test_df['failure_within_24h'] = y_test
    return test_df

def get_best_run():
    """Find the best run by ROC AUC score."""
    print("Looking for experiments in MLflow...")
    exp = client.get_experiment_by_name(
              'predictive_maintenance_sensors')

    if exp is None:
        print("No experiment found! Running train_model.py...")
        subprocess.run(['py', 'src/train_model.py'])
        exp = client.get_experiment_by_name(
                  'predictive_maintenance_sensors')

    if exp is None:
        print("Still no experiment found. Exiting.")
        return None, None

    runs = client.search_runs(
               exp.experiment_id,
               order_by=['metrics.roc_auc DESC'])

    if not runs:
        print("No runs found! Exiting.")
        return None, None

    best   = runs[0]
    run_id = best.info.run_id
    print(f"Best run ID : {run_id}")
    print(f"Best ROC AUC: {best.data.metrics['roc_auc']:.4f}")
    return run_id, best.data.metrics

def evaluate_model(run_id):
    """Evaluate model using mlflow.evaluate()."""
    print("\nEvaluating model with mlflow.evaluate()...")
    model_uri = f'runs:/{run_id}/failure_predictor'
    test_data = load_test_data()

    result = mlflow.evaluate(
        model      = model_uri,
        data       = test_data,
        targets    = 'failure_within_24h',
        model_type = 'classifier',
    )

    print("\nEvaluation Results:")
    for key, val in result.metrics.items():
        print(f"  {key}: {val:.4f}")
    return result.metrics

def register_model(run_id, metrics):
    """Register model if it meets all thresholds."""
    print("\nChecking production thresholds...")

    passes = all(
        metrics.get(k, 0) >= v
        for k, v in THRESHOLDS.items()
    )

    if not passes:
        print("Model did NOT meet thresholds.")
        print("Staying in Staging — not promoted.")
        for k, v in THRESHOLDS.items():
            actual = metrics.get(k, 0)
            status = "PASS" if actual >= v else "FAIL"
            print(f"  {k}: {actual:.4f} (need {v}) [{status}]")
        return

    print("All thresholds passed! Registering model...")

    mv = mlflow.register_model(
        model_uri = f'runs:/{run_id}/failure_predictor',
        name      = 'PredictiveMaintenance_RF'
    )

    client.transition_model_version_stage(
        name    = 'PredictiveMaintenance_RF',
        version = mv.version,
        stage   = 'Production',
    )
    print(f"Model v{mv.version} promoted to Production!")
    print("Ready to sync to Azure ML.")

if __name__ == '__main__':
    print("Starting evaluation pipeline...")
    print("=" * 45)
    run_id, metrics = get_best_run()
    if run_id:
        eval_metrics = evaluate_model(run_id)
        register_model(run_id, eval_metrics)
    print("=" * 45)
    print("Done!")