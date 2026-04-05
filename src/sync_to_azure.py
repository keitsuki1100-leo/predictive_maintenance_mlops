# src/sync_to_azure.py
# Experiment 8 — Sync Production model to Azure ML
# Run this only when internet is available

import mlflow
import os
from mlflow.tracking import MlflowClient

# ── Local MLflow tracking ───────────────────────────────────
mlflow.set_tracking_uri('sqlite:///mlflow_local.db')
client = MlflowClient()

def sync_to_azure():
    """Sync Production model from local MLflow to Azure ML."""

    # ── Step 1: Check if Azure credentials are set ──────────
    required_vars = [
        'AZURE_SUBSCRIPTION_ID',
        'AZURE_RESOURCE_GROUP',
        'AZURE_WORKSPACE',
    ]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print("Missing Azure credentials:")
        for v in missing:
            print(f"  {v} not set in .env")
        print("\nSet them in your .env file first!")
        return

    # ── Step 2: Get Production model from local registry ────
    print("Looking for Production model locally...")
    try:
        versions = client.get_latest_versions(
            name   = 'PredictiveMaintenance_RF',
            stages = ['Production']
        )
    except Exception as e:
        print(f"Error: {e}")
        print("Run evaluate_and_register.py first!")
        return

    if not versions:
        print("No Production model found locally.")
        print("Run evaluate_and_register.py first!")
        return

    mv         = versions[0]
    run_id     = mv.run_id
    version    = mv.version
    print(f"Found Production model v{version}")
    print(f"Run ID: {run_id}")

    # ── Step 3: Download model artifact locally ─────────────
    print("\nDownloading model artifact...")
    model_uri  = f'runs:/{run_id}/failure_predictor'
    local_path = mlflow.artifacts.download_artifacts(model_uri)
    print(f"Model saved to: {local_path}")

    # ── Step 4: Register in Azure ML ────────────────────────
    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Model
        from azure.identity import DefaultAzureCredential

        print("\nConnecting to Azure ML...")
        az_client = MLClient(
            credential      = DefaultAzureCredential(),
            subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID'),
            resource_group  = os.getenv('AZURE_RESOURCE_GROUP'),
            workspace_name  = os.getenv('AZURE_WORKSPACE'),
        )

        az_model = Model(
            path        = local_path,
            name        = 'predictive-maintenance-rf',
            description = 'Factory failure predictor synced from edge',
            type        = 'mlflow_model',
        )
        registered = az_client.models.create_or_update(az_model)
        print(f"Synced to Azure ML: v{registered.version}")

    except ImportError:
        print("Azure SDK not installed.")
        print("Run: pip install azure-ai-ml azure-identity")
        return

    # ── Step 5: Push data to Azure Blob via DVC ─────────────
    print("\nPushing data to Azure Blob via DVC...")
    os.system('dvc push')
    print("DVC data pushed to Azure Blob!")
    print("\nSync complete!")

if __name__ == '__main__':
    sync_to_azure()