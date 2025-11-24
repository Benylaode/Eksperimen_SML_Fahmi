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


# Atur tracking URI
if os.getenv("GITHUB_ACTIONS") == "true":
    mlflow_tracking_uri = os.path.join(os.getcwd(), "mlruns")  # absolut path
    mlflow.set_tracking_uri(f"file:{mlflow_tracking_uri}")
    print("🔧 Running in GitHub Actions → using local MLflow store:", mlflow_tracking_uri)
else:
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


# Mulai MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"🔍 MLflow tracking: {MLFLOW_TRACKING_URI}")


# Daftar model
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}


def train_model(name, model):
    print(f"\n🚀 Training model: {name}")

    with mlflow.start_run(run_name=name):
        print(f"📌 Logging manual → model {name}")

        t0 = time.time()

        # Train
        model.fit(X_train, y_train)

        duration = time.time() - t0
        pred = model.predict(X_test)

        # Hitung metrik
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        # === Logging manual ===
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("training_time_sec", duration)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Plot prediction
        plt.figure()
        plt.scatter(y_test, pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{name} Prediction Plot")

        plot_path = os.path.join(OUTPUT_DIR, f"{name}_plot.png")
        plt.savefig(plot_path)
        plt.close()

        # Log artifact (gambar)
        mlflow.log_artifact(plot_path)

        return {"model": name, "mse": mse, "mae": mae, "r2": r2}


def main():
    results = []
    for name, model in models.items():
        results.append(train_model(name, model))

    # Simpan benchmark
    results_df = pd.DataFrame(results)
    benchmark_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    results_df.to_csv(benchmark_path, index=False)

    print("\n📊 Hasil training disimpan → benchmark_results.csv")

    with mlflow.start_run(run_name="benchmark_summary"):
        mlflow.log_artifact(benchmark_path)


if __name__ == "__main__":
    main()
