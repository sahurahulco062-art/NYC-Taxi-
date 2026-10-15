
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, transformer, data_path, dataset_name):
    # Load the prepared dataset
    data = pd.read_csv(data_path)

    X = data.drop(columns=["trip_duration"])
    y_transformed = data["trip_duration"].to_numpy().reshape(-1, 1)

    # Predict in the transformed target space
    predictions_transformed = model.predict(X).reshape(-1, 1)

    # Convert actual and predicted durations back to minutes
    y_actual = transformer.inverse_transform(y_transformed).ravel()
    y_predicted = transformer.inverse_transform(
        predictions_transformed
    ).ravel()

    # Calculate metrics in minutes
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    mae = mean_absolute_error(y_actual, y_predicted)
    r2 = r2_score(y_actual, y_predicted)

    print(f"\n{dataset_name} Results")
    print("-" * 40)
    print(f"Rows: {len(data):,}")
    print(f"RMSE: {rmse:.4f} minutes")
    print(f"MAE:  {mae:.4f} minutes")
    print(f"R²:   {r2:.4f}")

    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    root = Path(__file__).resolve().parents[2]

    model_path = root / "models" / "models" / "xgbreg.joblib"
    transformer_path = (
        root / "models" / "transformers" / "output_transformer.joblib"
    )
    train_path = root / "data" / "processed" / "final" / "train.csv"
    val_path = root / "data" / "processed" / "final" / "val.csv"

    # Load the existing trained artifacts
    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path)

    print("Evaluating the existing XGBoost model")
    print("No model retraining will be performed.")

    train_metrics = evaluate_model(
        model, transformer, train_path, "TRAINING"
    )
    val_metrics = evaluate_model(
        model, transformer, val_path, "VALIDATION"
    )

    print("\nOVERFITTING CHECK")
    print("-" * 40)
    print(
        f"RMSE gap (validation - training): "
        f"{val_metrics['rmse'] - train_metrics['rmse']:.4f} minutes"
    )
    print(
        f"MAE gap (validation - training): "
        f"{val_metrics['mae'] - train_metrics['mae']:.4f} minutes"
    )
    print(
        f"R² gap (training - validation): "
        f"{train_metrics['r2'] - val_metrics['r2']:.4f}"
    )

    print(
        "\nCompare the gaps alongside the absolute metrics. "
        "A larger training-validation gap can indicate overfitting."
    )


if __name__ == "__main__":
    main()
