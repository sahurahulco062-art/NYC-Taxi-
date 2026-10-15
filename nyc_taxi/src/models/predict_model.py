import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.logger import CustomLogger, create_log_path
from src.features.data_preprocessing import TARGET


# Preserve the custom logging convention used by the original project.
log_file_path = create_log_path("predict_model")
predict_logger = CustomLogger(
    logger_name="predict_model",
    log_filename=log_file_path,
)
predict_logger.set_log_level(level=logging.INFO)

MODEL_NAME = "xgbreg.joblib"
TRANSFORMER_NAME = "output_transformer.joblib"


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a prepared dataset from CSV."""
    return pd.read_csv(path)


def make_X_y(dataframe: pd.DataFrame, target_column: str):
    """Split a dataframe into features and target."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")
    X = dataframe.drop(columns=target_column)
    y = dataframe[target_column]
    return X, y


def calculate_metrics(y_actual_minutes, y_predicted_minutes) -> dict:
    """Calculate evaluation metrics in the original target unit: minutes."""
    y_actual_minutes = np.asarray(y_actual_minutes).ravel()
    y_predicted_minutes = np.asarray(y_predicted_minutes).ravel()

    return {
        "rmse_minutes": float(np.sqrt(mean_squared_error(
            y_actual_minutes, y_predicted_minutes
        ))),
        "mae_minutes": float(mean_absolute_error(
            y_actual_minutes, y_predicted_minutes
        )),
        "r2": float(r2_score(y_actual_minutes, y_predicted_minutes)),
    }


def main():
    root_path = Path(__file__).resolve().parents[2]
    data_path = root_path / "data" / "processed" / "final"
    model_path = root_path / "models" / "models" / MODEL_NAME
    transformer_path = (
        root_path / "models" / "transformers" / TRANSFORMER_NAME
    )
    metrics_path = root_path / "reports" / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) < 3:
        raise ValueError(
            "Usage: python src/models/predict_model.py train.csv val.csv"
        )

    # Keep the original pipeline's command-line interface. The training file
    # argument is accepted for compatibility; final metrics use held-out val.
    filenames = sys.argv[1:3]
    if "val.csv" not in filenames:
        raise ValueError("The validation dataset 'val.csv' must be supplied.")

    val_path = data_path / "val.csv"
    if not val_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {val_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    if not transformer_path.exists():
        raise FileNotFoundError(
            f"Target transformer not found: {transformer_path}"
        )

    predict_logger.save_logs(f"Loading trained model from {model_path}")
    model = joblib.load(model_path)
    target_transformer = joblib.load(transformer_path)

    val_data = load_dataframe(val_path)
    X_val, y_val_transformed = make_X_y(val_data, TARGET)

    predict_logger.save_logs(
        f"Loaded validation data: {val_path}; rows={len(val_data)}"
    )
    predictions_transformed = model.predict(X_val)

    # The preprocessing pipeline stores both y and predictions in
    # Yeo-Johnson-transformed space. Convert both back to trip-duration minutes.
    y_val_minutes = target_transformer.inverse_transform(
        y_val_transformed.to_numpy().reshape(-1, 1)
    ).ravel()
    predictions_minutes = target_transformer.inverse_transform(
        np.asarray(predictions_transformed).reshape(-1, 1)
    ).ravel()

    metrics = calculate_metrics(y_val_minutes, predictions_minutes)

    for metric_name, metric_value in metrics.items():
        predict_logger.save_logs(f"Validation {metric_name}: {metric_value:.6f}")

    results = {
        "validation_metrics": metrics,
        "validation_rows": int(len(y_val_minutes)),
        "target_unit": "minutes",
    }
    metrics_file = metrics_path / "validation_metrics.json"
    import json
    with open(metrics_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    predict_logger.save_logs(f"Validation metrics saved to {metrics_file}")
    print("Validation metrics (trip duration in minutes):")
    print(f"RMSE: {metrics['rmse_minutes']:.4f} minutes")
    print(f"MAE:  {metrics['mae_minutes']:.4f} minutes")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"Saved metrics: {metrics_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        predict_logger.save_logs(
            "Prediction/evaluation pipeline failed",
            log_level="exception",
        )
        raise
