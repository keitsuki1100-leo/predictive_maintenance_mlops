# start_demo.py
# Run this file to open entire live demo automatically!

import webbrowser
import subprocess
import threading
import time
import os

os.chdir(r'C:\Users\MAYANK\Desktop\predictive_maintenance_mlops')

print("=" * 50)
print("  Predictive Maintenance MLOps - Live Demo")
print("=" * 50)

# ── Start FastAPI ───────────────────────────────────
print("\n1. Starting FastAPI server...")
fastapi_process = subprocess.Popen(
    ['py', '-m', 'uvicorn', 'src.predict_api:app',
     '--reload', '--port', '8000'],
    creationflags=subprocess.CREATE_NEW_CONSOLE
)

# ── Start MLflow UI ─────────────────────────────────
print("2. Starting MLflow dashboard...")
mlflow_process = subprocess.Popen(
    ['py', '-m', 'mlflow', 'ui',
     '--backend-store-uri', 'sqlite:///mlflow_local.db',
     '--port', '5000'],
    creationflags=subprocess.CREATE_NEW_CONSOLE
)

# ── Wait for servers to start ───────────────────────
print("3. Waiting for servers to start...")
print("   Please wait 5 seconds...")
time.sleep(5)

# ── Open all browser tabs automatically ─────────────
print("4. Opening all demo tabs in browser...")

# FastAPI docs
webbrowser.open('http://127.0.0.1:8000/docs')
time.sleep(1)

# MLflow dashboard
webbrowser.open('http://127.0.0.1:5000')
time.sleep(1)

# Jenkins pipeline
webbrowser.open('http://localhost:8080')
time.sleep(1)

# GitHub repository
webbrowser.open(
    'https://github.com/keitsuki1100-leo/predictive_maintenance_mlops'
)
time.sleep(1)

# GitHub Actions
webbrowser.open(
    'https://github.com/keitsuki1100-leo/predictive_maintenance_mlops/actions'
)

print("\n" + "=" * 50)
print("  All demo tabs opened successfully!")
print("=" * 50)
print("\nBrowser tabs opened:")
print("  1. FastAPI    -> http://127.0.0.1:8000/docs")
print("  2. MLflow     -> http://127.0.0.1:5000")
print("  3. Jenkins    -> http://localhost:8080")
print("  4. GitHub     -> your repository")
print("  5. Actions    -> GitHub Actions pipeline")
print("\nPress Ctrl+C to stop all servers when done.")
print("=" * 50)

# ── Keep running ────────────────────────────────────
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping all servers...")
    fastapi_process.terminate()
    mlflow_process.terminate()
    print("Done! Goodbye!")