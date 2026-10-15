import json
from pathlib import Path
import logging

import joblib
import numpy as np
import optuna
import pandas as pd
from yaml import safe_load
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

from src.features.data_preprocessing import (
    COLUMN_NAMES,
    TARGET,
    remove_outliers,
    train_preprocessor,
    transform_data,
    transform_output,
)
from src.logger import CustomLogger, create_log_path
from src.features.distances import (
    haversine_distance,
    euclidean_distance,
    manhattan_distance,
)


# Keep the same custom logging convention used by the original project.
log_file_path = create_log_path("train_model")
train_logger = CustomLogger(logger_name="train_model", log_filename=log_file_path)
train_logger.set_log_level(level=logging.INFO)


def apply_original_cleaning_and_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the deterministic cleaning and feature-engineering operations
    from the original modify_features.py and build_features.py.

    Input is one fold's raw split from data/interim/train.csv. No statistics
    are learned here; learned preprocessing is fitted separately below.
    """
    df = data.copy()

    # Original modify_features.py: remove columns available in train/validation.
    columns_to_drop = [
        col for col in ["id", "dropoff_datetime", "store_and_fwd_flag"]
        if col in df.columns
    ]
    df = df.drop(columns=columns_to_drop)

    # Original modify_features.py: retain passenger counts from 1 through 6.
    df = df.loc[df["passenger_count"].isin(range(1, 7))].copy()

    # Original modify_features.py: derive datetime features and drop timestamp.
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_date"] = df["pickup_datetime"].dt.day
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day"] = df["pickup_datetime"].dt.weekday
    df["is_weekend"] = (df["pickup_day"] >= 5).astype(int)
    df = df.drop(columns=["pickup_datetime"])

    # Original modify_features.py: seconds to minutes and fixed domain cutoff.
    if TARGET in df.columns:
        df[TARGET] = df[TARGET] / 60.0
        df = df.loc[df[TARGET] <= 200].copy()

    # Original build_features.py: same three distance calculations.
    df["haversine_distance"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["euclidean_distance"] = euclidean_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["manhattan_distance"] = manhattan_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    return df


def calculate_rmse_minutes(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def main():
    root_path = Path(__file__).resolve().parents[2]
    train_logger.save_logs("Training pipeline started")

    with open(root_path / "params.yaml", encoding="utf-8") as file:
        params = safe_load(file)

    settings = params["train_model"]
    n_trials = int(settings["n_trials"])
    n_splits = int(settings["cv_folds"])
    random_state = int(settings["random_state"])
    percentiles = list(params["data_preprocessing"]["percentiles"])

    # CV starts BEFORE modify_features.py, build_features.py and data_preprocessing.py.
    # This lets each fold run those operations independently.
    cv_path = root_path / "data" / "interim" / "train.csv"
    raw_train = pd.read_csv(cv_path)
    train_logger.save_logs(
        f"Loaded raw training data for cross-validation: {cv_path}; rows={len(raw_train)}"
    )
    train_logger.save_logs(
        f"Starting Optuna optimization: trials={n_trials}, folds={n_splits}, random_state={random_state}"
    )

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        model_params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True

            ),

            "tree_method": "hist",
            "device": "cuda",
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "reg:squarederror",
            "random_state": random_state,
            "verbosity": 0,
        }

        fold_rmses = []
        train_logger.save_logs(f"Optuna trial {trial.number + 1}/{n_trials} started")

        for fold_number, (fold_train_idx, fold_val_idx) in enumerate(kfold.split(raw_train), start=1):
            # Split raw training data first; validation rows do not fit transformers.
            fold_train_raw = raw_train.iloc[fold_train_idx].copy()
            fold_val_raw = raw_train.iloc[fold_val_idx].copy()

            # Re-run the original deterministic cleaning + feature engineering.
            fold_train = apply_original_cleaning_and_feature_engineering(fold_train_raw)
            fold_val = apply_original_cleaning_and_feature_engineering(fold_val_raw)

            X_train = fold_train.drop(columns=TARGET)
            y_train = fold_train[TARGET]
            X_val = fold_val.drop(columns=TARGET)
            y_val = fold_val[TARGET]

            # Fit outlier thresholds ONLY on this fold's training features.
            outlier_transformer = remove_outliers(
                dataframe=X_train,
                percentiles=percentiles,
                column_names=COLUMN_NAMES,
            )
            X_train_filtered = transform_data(outlier_transformer, X_train)
            X_val_filtered = transform_data(outlier_transformer, X_val)

            # Keep targets aligned with rows retained by outlier removal.
            y_train_filtered = y_train.loc[X_train_filtered.index]
            y_val_filtered = y_val.loc[X_val_filtered.index]

            if X_train_filtered.empty or X_val_filtered.empty:
                raise ValueError(
                    "A CV fold has no rows remaining after outlier removal."
                )
            train_logger.save_logs(
                f"Trial {trial.number + 1}, fold {fold_number}: rows after cleaning/outlier removal "
                f"train={len(X_train_filtered)}, validation={len(X_val_filtered)}"
            )

            # Fit feature transformations ONLY on the fold's training rows.
            preprocessor = train_preprocessor(X_train_filtered)
            X_train_processed = transform_data(preprocessor, X_train_filtered)
            X_val_processed = transform_data(preprocessor, X_val_filtered)

            # Fit Yeo-Johnson ONLY on fold-training targets.
            target_transformer = transform_output(y_train_filtered)
            y_train_transformed = transform_data(
                target_transformer,
                y_train_filtered.to_numpy().reshape(-1, 1),
            ).ravel()

            model = XGBRegressor(**model_params)
            model.fit(X_train_processed, y_train_transformed)

            pred_transformed = model.predict(X_val_processed)
            pred_minutes = target_transformer.inverse_transform(
                pred_transformed.reshape(-1, 1)
            ).ravel()

            fold_rmse = calculate_rmse_minutes(y_val_filtered, pred_minutes)
            fold_rmses.append(fold_rmse)
            train_logger.save_logs(
                f"Trial {trial.number + 1}, fold {fold_number}/{n_splits}: RMSE={fold_rmse:.4f} minutes"
            )

        mean_rmse = float(np.mean(fold_rmses))
        train_logger.save_logs(
            f"Trial {trial.number + 1} completed: mean CV RMSE={mean_rmse:.4f} minutes; params={model_params}"
        )
        return mean_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    train_logger.save_logs(f"Optuna optimization completed. Best parameters: {study.best_params}")
    train_logger.save_logs(f"Best mean CV RMSE: {study.best_value:.4f} minutes")

    # Final training uses the complete, already-prepared dataset from the
    # original GitHub pipeline. Validation data is NOT loaded or evaluated here.
    final_train_path = root_path / "data" / "processed" / "final" / "train.csv"
    train_data = pd.read_csv(final_train_path)
    train_logger.save_logs(
        f"Loaded prepared training data for final fit: {final_train_path}; rows={len(train_data)}"
    )
    X_train_final = train_data.drop(columns=TARGET)
    y_train_final = train_data[TARGET]

    final_model_params = {
        **study.best_params,
        "objective": "reg:squarederror",
        "random_state": random_state,
        "tree_method": "hist",
        "device": "cuda",
        "verbosity": 0,
    }
    final_model = XGBRegressor(**final_model_params)
    train_logger.save_logs("Training final XGBoost model on the complete prepared training dataset")
    final_model.fit(X_train_final, y_train_final)

    model_dir = root_path / "models" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_dir / "xgbreg.joblib")
    train_logger.save_logs(f"Final model saved to {model_dir / 'xgbreg.joblib'}")

    metrics_dir = root_path / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "best_params": study.best_params,
        "cv_rmse_minutes": float(study.best_value),
        "cv_folds": n_splits,
        "n_trials": n_trials,
    }
    with open(metrics_dir / "optuna_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    train_logger.save_logs(f"Optuna results saved to {metrics_dir / 'optuna_results.json'}")
    train_logger.save_logs("Training pipeline completed successfully")
    print("Best parameters:", study.best_params)
    print(f"Mean CV RMSE (minutes): {study.best_value:.4f}")
    print("Saved model:", model_dir / "xgbreg.joblib")
    print("Saved tuning results:", metrics_dir / "optuna_results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        train_logger.save_logs("Training pipeline failed", log_level="exception")
        raise
