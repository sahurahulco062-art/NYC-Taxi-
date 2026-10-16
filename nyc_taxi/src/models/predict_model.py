"""Evaluate the natural-log stacking regressor on the held-out validation split.

The stacking model's base estimators perform outlier filtering and feature
preprocessing internally. Therefore, this script must receive the
feature-engineered validation data before learned preprocessing:
    data/processed/build-features/val.csv

The target in that file is expected to remain in trip-duration minutes.
The model predicts log(minutes), so predictions are converted back with exp.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.logger import CustomLogger, create_log_path


TARGET = "trip_duration"
MODEL_NAME = "stacking_regressor.joblib"

log_file_path = create_log_path("predict_model")
predict_logger = CustomLogger(
    logger_name="predict_model",
    log_filename=log_file_path,
)
predict_logger.set_log_level(level=logging.INFO)


def calculate_metrics(y_actual_minutes, y_predicted_minutes) -> dict[str, float]:
    """Calculate regression metrics in the original unit: trip-duration minutes."""
    y_actual = np.asarray(y_actual_minutes, dtype=np.float64).reshape(-1)
    y_predicted = np.asarray(y_predicted_minutes, dtype=np.float64).reshape(-1)

    if len(y_actual) != len(y_predicted):
        raise ValueError("Actual targets and predictions have different lengths.")
    if len(y_actual) == 0:
        raise ValueError("Cannot calculate metrics for an empty validation set.")
    if not np.isfinite(y_actual).all() or not np.isfinite(y_predicted).all():
        raise ValueError("Actual targets or predictions contain NaN or infinity.")

    return {
        "rmse_minutes": float(np.sqrt(mean_squared_error(y_actual, y_predicted))),
        "mae_minutes": float(mean_absolute_error(y_actual, y_predicted)),
        "r2": float(r2_score(y_actual, y_predicted)),
    }


def load_validation_data(root: Path, cli_args: list[str]) -> tuple[Path, pd.DataFrame]:
    """Load feature-engineered validation data; accept the legacy DVC arguments."""
    if cli_args:
        # The existing DVC stage calls this script as: train.csv val.csv.
        # The train filename is retained for command compatibility only.
        if len(cli_args) not in (1, 2):
            raise ValueError(
                "Usage: python src/models/predict_model.py [train.csv] [val.csv]"
            )
        supplied_name = Path(cli_args[-1]).name
        if supplied_name.lower() != "val.csv":
            raise ValueError(
                "The validation argument must identify val.csv. The model needs "
                "data/processed/build-features/val.csv, before learned preprocessing."
            )

    validation_path = root / "data" / "processed" / "build-features" / "val.csv"
    if not validation_path.exists():
        raise FileNotFoundError(
            f"Feature-engineered validation data not found: {validation_path}. "
            "Run the DVC pipeline through build_features first."
        )

    dataframe = pd.read_csv(validation_path)
    if TARGET not in dataframe.columns:
        raise ValueError(
            f"Required target column '{TARGET}' is missing from {validation_path}."
        )
    if dataframe.empty:
        raise ValueError(f"Validation dataset is empty: {validation_path}")
    return validation_path, dataframe


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    model_path = root / "models" / "models" / MODEL_NAME
    metrics_dir = root / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained stacking model not found: {model_path}. "
            "Run src/models/train_model.py first."
        )

    validation_path, validation_data = load_validation_data(root, sys.argv[1:])
    y_val_minutes = pd.to_numeric(
        validation_data[TARGET], errors="raise"
    ).to_numpy(dtype=np.float64)
    X_val = validation_data.drop(columns=[TARGET])

    if not np.isfinite(y_val_minutes).all() or (y_val_minutes <= 0).any():
        raise ValueError(
            "Validation trip durations must be finite, positive values in minutes."
        )

    predict_logger.save_logs(f"Loading stacking model from {model_path}")
    model = joblib.load(model_path)

    # Confirm compatible feature schema and restore the exact training column order.
    # Validate the feature count.
    if hasattr(model, "n_features_in_"):
        if X_val.shape[1] != model.n_features_in_:
             raise ValueError(
            f"Feature count mismatch: model expects "
            f"{model.n_features_in_} features, but validation data "
            f"contains {X_val.shape[1]}."
        )

    non_numeric = list(X_val.select_dtypes(exclude=[np.number]).columns)
    if non_numeric:
        raise ValueError(f"Validation features contain non-numeric columns: {non_numeric}")
    X_values = X_val.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(X_values).all():
        raise ValueError("Validation features contain NaN or infinite values.")

    predict_logger.save_logs(
        f"Loaded validation data from {validation_path}; rows={len(validation_data):,}; "
        f"features={X_val.shape[1]}."
    )
    predictions_log = np.asarray(model.predict(X_val), dtype=np.float64).reshape(-1)
    if not np.isfinite(predictions_log).all():
        raise ValueError("The stacking model produced NaN or infinite log predictions.")

    # The trained model predicts log(duration in minutes); restore minutes for scoring.
    with np.errstate(over="raise", invalid="raise"):
        try:
            predictions_minutes = np.exp(predictions_log)
        except FloatingPointError as exc:
            raise ValueError(
                "Exponentiating the model's log predictions overflowed; inspect model output."
            ) from exc

    metrics = calculate_metrics(y_val_minutes, predictions_minutes)
    for name, value in metrics.items():
        predict_logger.save_logs(f"Validation {name}: {value:.6f}")

    results = {
        "model": "StackingRegressor",
        "model_path": str(model_path),
        "validation_data": str(validation_path),
        "validation_rows": int(len(y_val_minutes)),
        "feature_count": int(X_val.shape[1]),
        "target_transform": "natural_log",
        "target_unit": "minutes",
        "validation_metrics": metrics,
    }
    metrics_path = metrics_dir / "stacking_validation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    predict_logger.save_logs(f"Validation metrics saved to {metrics_path}")
    print("Stacking model validation metrics (trip duration in minutes):")
    print(f"RMSE: {metrics['rmse_minutes']:.4f} minutes")
    print(f"MAE:  {metrics['mae_minutes']:.4f} minutes")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        predict_logger.save_logs(
            "Stacking prediction/evaluation pipeline failed.",
            log_level="exception",
        )
        raise
