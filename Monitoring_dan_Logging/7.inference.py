#!/usr/bin/env python3
import joblib
import pandas as pd
import json
import os

MODEL_PATH = "../Membangun_model/california_housing_data/artifacts/model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def predict_single(data: dict):
    model = load_model()
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"prediction": float(pred)}

if __name__ == "__main__":
    sample = {
        "MedInc": 8.3,
        "HouseAge": 20,
        "AveRooms": 5.5,
        "AveBedrms": 1.0,
        "Population": 1500,
        "AveOccup": 3.0,
        "Latitude": 37.5,
        "Longitude": -122.0
    }
    print(json.dumps(predict_single(sample), indent=4))
