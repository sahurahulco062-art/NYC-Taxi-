"""Leakage-aware Optuna tuning and final StackingRegressor training.

Hyperparameter tuning:
- Uses data/processed/build-features/train.csv (before learned preprocessing).
- Uses separate data subsets for base-model tuning and Ridge-alpha tuning.
- Fits outlier thresholds and feature preprocessing inside every CV training fold.
- Uses Optuna for XGBoost, LightGBM, and Ridge tuning.

Final training:
- Fits a plain StackingRegressor on data/processed/final/train.csv.
- Assumes trip_duration is already natural-log transformed by data_preprocessing.py.
- Does not evaluate validation predictions here; that belongs in predict_model.py.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline as ImblearnPipeline
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor
from yaml import safe_load

from src.logger import CustomLogger, create_log_path
from src.models.model_components import (
    ExistingProjectPreprocessor,
    fold_local_outlier_filter,
)


# ---------------------------------------------------------------------------
# Project logging
# ---------------------------------------------------------------------------
log_file_path = create_log_path("train_model")
train_logger = CustomLogger(logger_name="train_model", log_filename=log_file_path)
train_logger.set_log_level(level=logging.INFO)


def log(message: str) -> None:
    """Write a progress message to the project's existing logger."""
    train_logger.save_logs(message)


TARGET = "trip_duration"
OUTLIER_COLUMNS = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
ONE_HOT_COLUMNS = ["vendor_id"]
MIN_MAX_COLUMNS = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
STANDARD_COLUMNS = [
    "haversine_distance",
    "euclidean_distance",
    "manhattan_distance",
]


def natural_log_target(y_minutes: pd.Series) -> pd.Series:
    """Apply the explicitly required natural logarithm to trip duration in minutes."""
    values = pd.Series(
        np.asarray(y_minutes, dtype=np.float64).reshape(-1),
        index=y_minutes.index,
        name=TARGET,
    )
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Target contains NaN or infinite values.")
    if (values <= 0).any():
        raise ValueError("All trip_duration values must be positive before np.log.")
    return np.log(values).rename(TARGET)


def negative_rmse_minutes_scorer(estimator, X, y_log) -> float:
    """Scikit-learn scorer: higher is better; predictions are scored in minutes."""
    y_true_log = np.asarray(y_log, dtype=np.float64).reshape(-1)
    y_pred_log = np.asarray(estimator.predict(X), dtype=np.float64).reshape(-1)
    y_true_minutes = np.exp(y_true_log)
    y_pred_minutes = np.exp(y_pred_log)
    rmse = np.sqrt(np.mean(np.square(y_true_minutes - y_pred_minutes)))
    return -float(rmse)



def build_fold_safe_pipeline(
    model_name: str,
    model_params: dict[str, Any],
    percentiles: list[float],
) -> ImblearnPipeline:
    """Use existing project functions while fitting learned transforms per CV fold."""
    sampler = FunctionSampler(
        func=fold_local_outlier_filter,
        kw_args={
            "percentiles": list(percentiles),
            "column_names": list(OUTLIER_COLUMNS),
        },
        validate=False,
    )

    if model_name == "xgboost":
        model = XGBRegressor(**model_params)
    elif model_name == "lightgbm":
        model = LGBMRegressor(**model_params)
    else:
        raise ValueError(f"Unsupported base model: {model_name}")

    return ImblearnPipeline(
        steps=[
            ("outlier_filter", sampler),
            ("preprocessor", ExistingProjectPreprocessor()),
            ("model", model),
        ]
    )


def make_xgb_params(trial: optuna.Trial, random_state: int) -> dict[str, Any]:
    """Optuna search space plus fixed project/hardware settings for XGBoost."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
        "random_state": random_state,
        "n_jobs": 1,
        "verbosity": 0,
    }


def make_lgbm_params(
    trial: optuna.Trial,
    random_state: int,
    n_jobs: int,
) -> dict[str, Any]:
    """Optuna search space plus fixed CPU settings for LightGBM."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10, 12]),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "objective": "regression",
        "device_type": "cpu",
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbosity": -1,
    }


def tune_base_model(
    model_name: str,
    X_tune: pd.DataFrame,
    y_tune_log: pd.Series,
    n_trials: int,
    cv_folds: int,
    random_state: int,
    percentiles: list[float],
    lightgbm_n_jobs: int,
) -> tuple[dict[str, Any], float, optuna.Study]:
    """Tune one base model with Optuna and scikit-learn CV.

    cross_val_score creates CV folds; the imbalanced-learn pipeline fits
    outlier thresholds and feature transformations on each training fold only.
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    study_name = f"{model_name}_tuning"
    log(
        f"Starting {model_name} tuning: trials={n_trials}, folds={cv_folds}, "
        f"rows={len(X_tune):,}."
    )

    def objective(trial: optuna.Trial) -> float:
        if model_name == "xgboost":
            model_params = make_xgb_params(trial, random_state)
        else:
            model_params = make_lgbm_params(trial, random_state, lightgbm_n_jobs)

        estimator = build_fold_safe_pipeline(model_name, model_params, percentiles)
        log(
            f"{model_name} trial {trial.number + 1}/{n_trials} started; "
            "preprocessing will be fitted within each CV training fold."
        )
        scores = cross_val_score(
            estimator,
            X_tune,
            y_tune_log,
            cv=cv,
            scoring=negative_rmse_minutes_scorer,
            n_jobs=1,
            error_score="raise",
        )
        fold_rmse = (-scores).astype(float)
        mean_rmse = float(np.mean(fold_rmse))
        log(
            f"{model_name} trial {trial.number + 1}/{n_trials} completed: "
            f"fold RMSE minutes={np.round(fold_rmse, 4).tolist()}, "
            f"mean={mean_rmse:.4f}."
        )
        return mean_rmse

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    best_search_params = dict(study.best_params)

    if model_name == "xgboost":
        best_params = {
            **best_search_params,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cuda",
            "random_state": random_state,
            "n_jobs": 1,
            "verbosity": 0,
        }
    else:
        best_params = {
            **best_search_params,
            "objective": "regression",
            "device_type": "cpu",
            "random_state": random_state,
            "n_jobs": lightgbm_n_jobs,
            "verbosity": -1,
        }
        # The search uses subsampling, so enable it in the final estimator too.
        best_params["subsample_freq"] = 1

    log(
        f"{model_name} tuning finished: best CV RMSE={study.best_value:.4f} minutes; "
        f"best search parameters={study.best_params}."
    )
    return best_params, float(study.best_value), study


def tune_ridge_alpha(
    X_ridge: pd.DataFrame,
    y_ridge_log: pd.Series,
    xgb_params: dict[str, Any],
    lgbm_params: dict[str, Any],
    n_trials: int,
    outer_cv_folds: int,
    stack_internal_folds: int,
    random_state: int,
    percentiles: list[float],
) -> tuple[float, float, optuna.Study]:
    """Tune Ridge alpha on data disjoint from base-model hyperparameter tuning.

    Each alpha is scored by outer CV around the entire StackingRegressor.
    Base-model parameters are fixed from the separate base-tuning subset.
    Within each stack fit, StackingRegressor creates OOF base predictions for
    Ridge. Each pipeline refits outlier thresholds and feature preprocessing
    within its own training fold.
    """
    outer_cv = KFold(
        n_splits=outer_cv_folds,
        shuffle=True,
        random_state=random_state + 1,
    )
    study = optuna.create_study(
        direction="minimize",
        study_name="ridge_meta_model_tuning",
        sampler=optuna.samplers.TPESampler(seed=random_state + 2),
    )
    log(
        f"Starting Ridge alpha tuning on disjoint data: rows={len(X_ridge):,}, "
        f"trials={n_trials}, outer folds={outer_cv_folds}, "
        f"stack internal folds={stack_internal_folds}."
    )

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-4, 1e4, log=True)
        estimators = [
            (
                "xgboost",
                build_fold_safe_pipeline("xgboost", xgb_params, percentiles),
            ),
            (
                "lightgbm",
                build_fold_safe_pipeline("lightgbm", lgbm_params, percentiles),
            ),
        ]
        stack = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=alpha),
            cv=KFold(
                n_splits=stack_internal_folds,
                shuffle=True,
                random_state=random_state,
            ),
            n_jobs=1,
            passthrough=False,
        )

        log(
            f"Ridge trial {trial.number + 1}/{n_trials}: alpha={alpha:.6g}; "
            "evaluating the complete stack with outer CV."
        )
        scores = cross_val_score(
            stack,
            X_ridge,
            y_ridge_log,
            cv=outer_cv,
            scoring=negative_rmse_minutes_scorer,
            n_jobs=1,
            error_score="raise",
        )
        fold_rmse = (-scores).astype(float)
        mean_rmse = float(np.mean(fold_rmse))
        log(
            f"Ridge trial {trial.number + 1}/{n_trials} completed: "
            f"outer-fold RMSE minutes={np.round(fold_rmse, 4).tolist()}, "
            f"mean={mean_rmse:.4f}, alpha={alpha:.6g}."
        )
        return mean_rmse

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    best_alpha = float(study.best_params["alpha"])
    log(
        f"Ridge tuning finished: best alpha={best_alpha:.6g}; "
        f"best outer-CV RMSE={study.best_value:.4f} minutes."
    )
    return best_alpha, float(study.best_value), study


def validate_numeric_features(X: pd.DataFrame, context: str) -> None:
    """Fail early when feature-engineering output contains invalid model inputs."""
    non_numeric = list(X.select_dtypes(exclude=[np.number]).columns)
    if non_numeric:
        raise ValueError(f"{context} has non-numeric feature columns: {non_numeric}")
    values = X.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{context} contains NaN or infinite feature values.")


def save_study(study: optuna.Study, path: Path) -> None:
    """Persist a compact, JSON-safe record of every Optuna trial."""
    payload = [
        {
            "number": trial.number,
            "value": None if trial.value is None else float(trial.value),
            "params": trial.params,
            "state": trial.state.name,
        }
        for trial in study.trials
    ]
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    started_at = time.time()
    root = Path(__file__).resolve().parents[2]
    log("Leakage-aware stacking training pipeline started.")
    log("Target: natural-log of trip duration in minutes (explicit np.log).")
    log("XGBoost device: CUDA; LightGBM and Ridge: CPU.")

    params_path = root / "params.yaml"
    with params_path.open("r", encoding="utf-8") as file:
        params = safe_load(file) or {}

    stack_cfg = params.get("stacking", {})
    xgb_cfg = stack_cfg.get("xgboost", {})
    lgbm_cfg = stack_cfg.get("lightgbm", {})
    ridge_cfg = stack_cfg.get("ridge", {})
    legacy_cfg = params.get("train_model", {})

    random_state = int(stack_cfg.get("random_state", legacy_cfg.get("random_state", 42)))
    stack_internal_folds = int(stack_cfg.get("cv_folds", 5))
    xgb_trials = int(xgb_cfg.get("n_trials", 20))
    lgbm_trials = int(lgbm_cfg.get("n_trials", 20))
    ridge_trials = int(ridge_cfg.get("n_trials", 20))
    xgb_cv_folds = int(xgb_cfg.get("cv_folds", stack_internal_folds))
    lgbm_cv_folds = int(lgbm_cfg.get("cv_folds", stack_internal_folds))
    ridge_outer_folds = int(ridge_cfg.get("cv_folds", 3))
    lightgbm_n_jobs = int(lgbm_cfg.get("n_jobs", -1))
    tuning_max_rows = 0
    base_tuning_fraction = float(stack_cfg.get("base_tuning_fraction", 0.7))
    percentiles = list(params.get("data_preprocessing", {}).get("percentiles", [0.002, 0.998]))

    if len(percentiles) != 2 or not (0 <= percentiles[0] < percentiles[1] <= 1):
        raise ValueError("data_preprocessing.percentiles must contain two increasing values from 0 to 1.")
    if min(xgb_trials, lgbm_trials, ridge_trials) < 1:
        raise ValueError("All stacking Optuna n_trials settings must be at least 1.")
    if min(xgb_cv_folds, lgbm_cv_folds, ridge_outer_folds, stack_internal_folds) < 2:
        raise ValueError("All stacking CV fold settings must be at least 2.")
    if tuning_max_rows != 0 and tuning_max_rows < max(xgb_cv_folds, lgbm_cv_folds, ridge_outer_folds, stack_internal_folds) * 2:
        raise ValueError("stacking.tuning_max_rows is too small for the configured CV folds.")
    if not 0.5 <= base_tuning_fraction <= 0.9:
        raise ValueError("stacking.base_tuning_fraction must be between 0.5 and 0.9.")

    # Use preprocessed-feature engineering output so every learned preprocessing
    # operation can be fitted afresh inside the CV training folds.
    raw_train_path = root / "data" / "processed" / "build-features" / "train.csv"
    if not raw_train_path.exists():
        raise FileNotFoundError(
            f"Leakage-safe tuning input not found: {raw_train_path}. "
            "Run the DVC stages through build_features before training."
        )
    raw_data = pd.read_csv(raw_train_path)
    if TARGET not in raw_data.columns:
        raise ValueError(f"Required target column '{TARGET}' missing from {raw_train_path}.")
    raw_y_minutes = pd.to_numeric(raw_data.pop(TARGET), errors="raise").astype("float64")
    if not np.isfinite(raw_y_minutes.to_numpy()).all() or (raw_y_minutes <= 0).any():
        raise ValueError("Raw training target must contain only finite, positive durations in minutes.")

    # Tuning sample is only a computational convenience; final training uses
    # all prepared training rows. Keep target and feature row order aligned.
    tuning_frame = raw_data.copy()
    tuning_frame[TARGET] = raw_y_minutes
    if tuning_max_rows > 0 and len(tuning_frame) > tuning_max_rows:
        tuning_frame = tuning_frame.sample(
            n=tuning_max_rows,
            random_state=random_state,
        ).reset_index(drop=True)
        log(
            f"Using deterministic tuning sample of {len(tuning_frame):,} rows "
            f"from {len(raw_data):,} available rows."
        )
    else:
        tuning_frame = tuning_frame.reset_index(drop=True)
        log(f"Using all {len(tuning_frame):,} rows for hyperparameter tuning.")

    X_tune_all = tuning_frame.drop(columns=[TARGET])
    y_tune_log_all = natural_log_target(tuning_frame[TARGET])
    validate_numeric_features(X_tune_all, "Raw feature-engineering output")

    # Critical separation: base-model hyperparameters never see the rows used
    # to tune/evaluate Ridge alpha.
    X_base_tune, X_ridge_tune, y_base_tune, y_ridge_tune = train_test_split(
        X_tune_all,
        y_tune_log_all,
        test_size=1.0 - base_tuning_fraction,
        random_state=random_state,
        shuffle=True,
    )
    log(
        f"Separated tuning data: base-model tuning rows={len(X_base_tune):,}; "
        f"Ridge alpha tuning rows={len(X_ridge_tune):,}."
    )

    xgb_params, xgb_cv_rmse, xgb_study = tune_base_model(
        model_name="xgboost",
        X_tune=X_base_tune,
        y_tune_log=y_base_tune,
        n_trials=xgb_trials,
        cv_folds=xgb_cv_folds,
        random_state=random_state,
        percentiles=percentiles,
        lightgbm_n_jobs=lightgbm_n_jobs,
    )
    lgbm_params, lgbm_cv_rmse, lgbm_study = tune_base_model(
        model_name="lightgbm",
        X_tune=X_base_tune,
        y_tune_log=y_base_tune,
        n_trials=lgbm_trials,
        cv_folds=lgbm_cv_folds,
        random_state=random_state,
        percentiles=percentiles,
        lightgbm_n_jobs=lightgbm_n_jobs,
    )

    best_alpha, ridge_cv_rmse, ridge_study = tune_ridge_alpha(
        X_ridge=X_ridge_tune,
        y_ridge_log=y_ridge_tune,
        xgb_params=xgb_params,
        lgbm_params=lgbm_params,
        n_trials=ridge_trials,
        outer_cv_folds=ridge_outer_folds,
        stack_internal_folds=stack_internal_folds,
        random_state=random_state,
        percentiles=percentiles,
    )

    # Final training uses the full feature-engineered training split, before
    # learned preprocessing. Each base estimator is a fold-safe imblearn
    # pipeline, so StackingRegressor's internal OOF folds fit outlier thresholds
    # and feature preprocessing only on each fold's training partition.
    X_final = raw_data.copy()
    y_final_log = natural_log_target(raw_y_minutes)
    validate_numeric_features(X_final, "Final feature-engineering training data")
    if len(X_final) < stack_internal_folds:
        raise ValueError(
            f"Final training rows ({len(X_final)}) are fewer than "
            f"stacking.cv_folds ({stack_internal_folds})."
        )

    log(
        f"Starting final fold-safe StackingRegressor fit: rows={len(X_final):,}, "
        f"features={X_final.shape[1]}, internal folds={stack_internal_folds}. "
        "Preprocessing will be fitted within each internal training fold."
    )
    final_stack = StackingRegressor(
        estimators=[
            ("xgboost", build_fold_safe_pipeline("xgboost", xgb_params, percentiles)),
            ("lightgbm", build_fold_safe_pipeline("lightgbm", lgbm_params, percentiles)),
        ],
        final_estimator=Ridge(alpha=best_alpha),
        cv=KFold(
            n_splits=stack_internal_folds,
            shuffle=True,
            random_state=random_state,
        ),
        n_jobs=1,
        passthrough=False,
    )
    final_stack.fit(X_final, y_final_log)
    log("Final fold-safe StackingRegressor fit completed.")

    model_dir = root / "models" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "stacking_regressor.joblib"
    joblib.dump(final_stack, model_path)
    log(f"Saved final stacking model to {model_path}.")

    metrics_dir = root / "reports" / "metrics"
    studies_dir = metrics_dir / "optuna"
    studies_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "model": "StackingRegressor",
        "base_models": ["XGBRegressor", "LGBMRegressor"],
        "meta_model": "Ridge",
        "target_transform": "natural_log",
        "prediction_inverse_transform": "exp",
        "random_state": random_state,
        "leakage_controls": {
            "tuning_and_final_training_input": "data/processed/build-features/train.csv",
            "learned_preprocessing_fitted_inside_each_cv_training_fold": True,
            "outlier_filtering_fits_quantiles_inside_each_cv_training_fold": True,
            "final_stack_base_estimators_include_fold_safe_preprocessing_pipelines": True,
            "ridge_tuning_data_disjoint_from_base_model_tuning_data": True,
            "ridge_score_evaluates_complete_stacking_regressor_with_outer_cv": True,
            "validation_dataset_used_during_tuning": False,
            "final_model_input": "feature-engineered data before learned preprocessing; pass matching build-features validation/test features at prediction time",
        },
        "tuning": {
            "rows_sampled_for_tuning": int(len(tuning_frame)),
            "base_model_tuning_rows": int(len(X_base_tune)),
            "ridge_alpha_tuning_rows": int(len(X_ridge_tune)),
            "base_tuning_fraction": base_tuning_fraction,
            "xgboost_trials": xgb_trials,
            "xgboost_cv_folds": xgb_cv_folds,
            "lightgbm_trials": lgbm_trials,
            "lightgbm_cv_folds": lgbm_cv_folds,
            "ridge_trials": ridge_trials,
            "ridge_outer_cv_folds": ridge_outer_folds,
            "stacking_internal_cv_folds": stack_internal_folds,
        },
        "xgboost": {
            "best_params": xgb_params,
            "best_cv_rmse_minutes": xgb_cv_rmse,
        },
        "lightgbm": {
            "best_params": lgbm_params,
            "best_cv_rmse_minutes": lgbm_cv_rmse,
        },
        "ridge": {
            "best_alpha": best_alpha,
            "best_outer_cv_rmse_minutes": ridge_cv_rmse,
        },
        "final_training_rows": int(len(X_final)),
        "final_feature_count": int(X_final.shape[1]),
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    metrics_path = metrics_dir / "stacking_results.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    log(f"Saved tuning results to {metrics_path}.")

    for study_name, study in (
        ("xgboost_study.json", xgb_study),
        ("lightgbm_study.json", lgbm_study),
        ("ridge_study.json", ridge_study),
    ):
        study_path = studies_dir / study_name
        save_study(study, study_path)
        log(f"Saved Optuna trial history to {study_path}.")

    elapsed = time.time() - started_at
    log(f"Leakage-aware stacking training pipeline completed in {elapsed:.2f} seconds.")
    print("Final stacking model saved:", model_path)
    print("Tuning results saved:", metrics_path)
    print(f"XGBoost best CV RMSE (minutes): {xgb_cv_rmse:.4f}")
    print(f"LightGBM best CV RMSE (minutes): {lgbm_cv_rmse:.4f}")
    print(f"Ridge best outer-CV RMSE (minutes): {ridge_cv_rmse:.4f}")
    print(f"Best Ridge alpha: {best_alpha:.6g}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        train_logger.save_logs("Stacking training pipeline failed.", log_level="exception")
        raise
