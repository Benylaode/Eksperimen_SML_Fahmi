import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

DATA_DIR = "california_housing_data/namadataset_preprocessing"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")

df = pd.read_csv(TRAIN_PATH)
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5],
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

with mlflow.start_run(run_name="Tuning_RandomForest"):
    grid.fit(X, y)
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("best_score", grid.best_score_)
    mlflow.sklearn.log_model(grid.best_estimator_, "best_model")
