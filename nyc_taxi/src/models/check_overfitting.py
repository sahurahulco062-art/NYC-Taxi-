
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "models" / "stacking_regressor.joblib"
TRAIN_PATH = ROOT / "data" / "processed" / "build-features" / "train.csv"
VAL_PATH = ROOT / "data" / "processed" / "build-features" / "val.csv"

TARGET = "trip_duration"


def evaluate(model, data, dataset_name):
    X = data.drop(columns=[TARGET])
    y_actual = pd.to_numeric(data[TARGET]).to_numpy(dtype=float)

    # The model predicts natural-log trip duration.
    y_pred = np.exp(
        np.asarray(model.predict(X), dtype=float).reshape(-1)
    )

    # If training data has already undergone preprocessing,
    # ensure this matches the target units used by your pipeline.
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)

    print(f"\n{dataset_name} metrics")
    print(f"RMSE: {rmse:.4f} minutes")
    print(f"MAE:  {mae:.4f} minutes")
    print(f"R²:   {r2:.4f}")

    return rmse, mae, r2


def main():
    model = joblib.load(MODEL_PATH)
    train_data = pd.read_csv(TRAIN_PATH)
    val_data = pd.read_csv(VAL_PATH)

    train_metrics = evaluate(model, train_data, "TRAINING")
    val_metrics = evaluate(model, val_data, "VALIDATION")

    print("\nMetric comparison")
    print(f"RMSE gap: {val_metrics[0] - train_metrics[0]:.4f} minutes")
    print(f"MAE gap:  {val_metrics[1] - train_metrics[1]:.4f} minutes")
    print(f"R² gap:   {train_metrics[2] - val_metrics[2]:.4f}")


if __name__ == "__main__":
    main()
