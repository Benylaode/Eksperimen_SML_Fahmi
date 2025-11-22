import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import joblib



BASE_DIR = os.path.join("..", "..", "Membangun_model", "california_housing_data")

RAW_DATA_PATH = os.path.join(BASE_DIR, "california_housing.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "namadataset_preprocessing")

SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.joblib")
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_raw_data():
    """
    Memuat data mentah atau membuat simulasi California housing dataset jika belum ada.
    """
    print("Memuat data mentah...")

    if not os.path.exists(RAW_DATA_PATH):
        print("Data mentah tidak ditemukan, membuat data mentah simulasi...")
        housing = fetch_california_housing()
        data = pd.DataFrame(housing.data, columns=housing.feature_names)
        data["MedHouseVal"] = housing.target
        data.to_csv(RAW_DATA_PATH, index=False)
        print(f"Data mentah simulasi dibuat di {RAW_DATA_PATH}")

    return pd.read_csv(RAW_DATA_PATH)

def preprocess_data(data):
    """
    Melakukan preprocessing: split, scaling, dan menyimpan output.
    """
    print("Memulai preprocessing...")

    X = data.drop("MedHouseVal", axis=1)
    y = data["MedHouseVal"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df["MedHouseVal"] = y_train.values

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df["MedHouseVal"] = y_test.values

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)

    print("Preprocessing selesai.")
    print(f"Data train: {TRAIN_PATH}")
    print(f"Data test: {TEST_PATH}")
    print(f"Scaler: {SCALER_PATH}")

    return train_df, test_df

def main():
    raw_data = load_raw_data()
    preprocess_data(raw_data)

if __name__ == "__main__":
    main()
