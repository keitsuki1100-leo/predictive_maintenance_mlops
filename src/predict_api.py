# src/predict_api.py
# FastAPI — Serve predictions from local MLflow model

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ── Load model from local MLflow Registry ──────────────────
mlflow.set_tracking_uri('sqlite:///mlflow_local.db')

app = FastAPI(
    title       = "Predictive Maintenance API",
    description = "Predicts machine failure from IoT sensor data",
    version     = "1.0.0"
)

# ── Load Production model ───────────────────────────────────
print("Loading Production model from MLflow...")
model = mlflow.sklearn.load_model(
    "models:/PredictiveMaintenance_RF/Production"
)
print("Model loaded successfully!")

# ── Input schema ────────────────────────────────────────────
class SensorData(BaseModel):
    temperature_c : float
    vibration_hz  : float
    pressure_bar  : float
    rpm           : float

# ── Health check endpoint ────────────────────────────────────
@app.get("/")
def home():
    return {
        "status" : "running",
        "message": "Predictive Maintenance API is live!",
        "version": "1.0.0"
    }

# ── Prediction endpoint ──────────────────────────────────────
@app.post("/predict")
def predict(data: SensorData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([{
        "temperature_c": data.temperature_c,
        "vibration_hz" : data.vibration_hz,
        "pressure_bar" : data.pressure_bar,
        "rpm"          : data.rpm,
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "machine_failure_predicted": bool(prediction),
        "failure_probability"      : round(float(probability), 4),
        "risk_level"               : (
            "HIGH"   if probability > 0.7 else
            "MEDIUM" if probability > 0.4 else
            "LOW"
        ),
        "recommendation": (
            "STOP MACHINE - Schedule immediate maintenance!"
            if prediction == 1 else
            "Machine is operating normally."
        )
    }

# ── Batch prediction endpoint ────────────────────────────────
@app.post("/predict/batch")
def predict_batch(sensors: list[SensorData]):
    results = []
    for data in sensors:
        input_df = pd.DataFrame([{
            "temperature_c": data.temperature_c,
            "vibration_hz" : data.vibration_hz,
            "pressure_bar" : data.pressure_bar,
            "rpm"          : data.rpm,
        }])
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        results.append({
            "machine_failure_predicted": bool(prediction),
            "failure_probability"      : round(float(probability), 4),
            "risk_level"               : (
                "HIGH"   if probability > 0.7 else
                "MEDIUM" if probability > 0.4 else
                "LOW"
            )
        })
    return {"predictions": results, "total": len(results)}
if __name__ == '__main__':
    import uvicorn
    import webbrowser
    import threading

    def open_browser():
        import time
        time.sleep(2)
        webbrowser.open('http://127.0.0.1:8000/docs')

    threading.Thread(target=open_browser).start()
    uvicorn.run(app, host='127.0.0.1', port=8000)