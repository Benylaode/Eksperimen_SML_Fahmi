import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import joblib
from scipy import stats

BASE_DIR = os.path.join("..", "..", "Membangun_model", "california_housing_data")

RAW_DATA_PATH = os.path.join(BASE_DIR, "california_housing.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "namadataset_preprocessing")

SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.joblib")
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_raw_data():
    print("Memuat data mentah...")
    if not os.path.exists(RAW_DATA_PATH):
        print("Data mentah tidak ditemukan, membuat data simulasi...")
        housing = fetch_california_housing()
        data = pd.DataFrame(housing.data, columns=housing.feature_names)
        data["MedHouseVal"] = housing.target
        data.to_csv(RAW_DATA_PATH, index=False)
        print(f"Data simulasi dibuat di {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH).rename(columns={'MedHouseVal': 'MedHouseVal'})

def handle_outliers(data):
    print("Menangani outlier dengan metode Z-score (threshold 3)...")


    numerical_features_for_outliers = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'MedHouseVal' in numerical_features_for_outliers:
        numerical_features_for_outliers.remove('MedHouseVal')
        
    z_scores = np.abs(stats.zscore(data[numerical_features_for_outliers]))
    threshold = 3
    
    data_cleaned = data[(z_scores < threshold).all(axis=1)]
    
    print(f"Jumlah baris sebelum outlier handling: {len(data)}")
    print(f"Jumlah baris setelah outlier handling: {len(data_cleaned)}")
    
    return data_cleaned

def preprocess_data(data):
    data_cleaned = handle_outliers(data)
    
    print("Preprocessing mengikuti BEST PRACTICE (fit scaler pada data training saja)...")

    X = data_cleaned.drop("MedHouseVal", axis=1)
    y = data_cleaned["MedHouseVal"]

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

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)

    print("Preprocessing selesai.")
    print(f"Train disimpan di: {TRAIN_PATH}")
    print(f"Test disimpan di: {TEST_PATH}")
    print(f"Scaler disimpan di: {SCALER_PATH}")


def main():
    raw = load_raw_data()
    preprocess_data(raw)


if __name__ == "__main__":
    main()
