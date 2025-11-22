#!/usr/bin/env python3
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature



if os.getenv("GITHUB_ACTIONS") == "true":
    MLFLOW_TRACKING_URI = "file:./mlruns"
    print("🔧 Running in GitHub Actions → using local MLflow store:", MLFLOW_TRACKING_URI)
else:
    # Jika run lokal → pakai MLflow Server dari docker-compose
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    print("🏠 Running locally → using MLflow Server:", MLFLOW_TRACKING_URI)
    
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "california_housing_exp")

DATA_DIR = "california_housing_data/namadataset_preprocessing"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

OUTPUT_DIR = "california_housing_data/artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("📂 Memuat dataset...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGET_COL = "MedHouseVal"

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"🔍 MLflow tracking: {MLFLOW_TRACKING_URI}")


models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

def train_model(name, model):
    print(f"\n🚀 Training model: {name}")
    t0 = time.time()

    model.fit(X_train, y_train)
    duration = time.time() - t0

    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_name", name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("training_duration_sec", duration)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

    return {"model": name, "mse": mse, "mae": mae, "r2": r2}

def main():
    results = []
    for name, model in models.items():
        results.append(train_model(name, model))

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "benchmark_results.csv"), index=False)
    print("\n📊 Hasil training disimpan.")

if __name__ == "__main__":
    main()
