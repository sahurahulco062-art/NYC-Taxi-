
"""Create validation diagnostics for the natural-log NYC Taxi stacking model.

Inputs:
- models/models/stacking_regressor.joblib
- data/processed/build-features/val.csv (feature-engineered, before learned preprocessing)

Output:
- plots/model_results/results.png

The model predicts log(trip_duration in minutes), so predictions are exponentiated
before metrics and plots are computed. All reported metrics use the full validation
split; the scatter plot samples at most 30,000 points for readability.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.logger import CustomLogger, create_log_path


TARGET = "trip_duration"
MODEL_NAME = "stacking_regressor.joblib"
MAX_SCATTER_POINTS = 30_000
RANDOM_STATE = 42

log_file_path = create_log_path("plot_results")
plot_logger = CustomLogger(logger_name="plot_results", log_filename=log_file_path)
plot_logger.set_log_level(level=logging.INFO)


def load_validation_data(root: Path, cli_args: list[str]) -> tuple[Path, pd.DataFrame]:
    """Load the feature-engineered validation split, retaining legacy CLI compatibility."""
    if cli_args and Path(cli_args[-1]).name.lower() != "val.csv":
        raise ValueError(
            "Expected val.csv as the final argument. The stacking model requires "
            "data/processed/build-features/val.csv before learned preprocessing."
        )

    validation_path = root / "data" / "processed" / "build-features" / "val.csv"
    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation data not found: {validation_path}. Run the DVC stages "
            "through build_features first."
        )

    dataframe = pd.read_csv(validation_path)
    if dataframe.empty:
        raise ValueError(f"Validation data is empty: {validation_path}")
    if TARGET not in dataframe.columns:
        raise ValueError(f"Target column '{TARGET}' is missing from {validation_path}")
    return validation_path, dataframe


def prepare_features(model, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Validate the feature schema and positive target values."""
    y_actual_minutes = pd.to_numeric(dataframe[TARGET], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(y_actual_minutes).all() or (y_actual_minutes <= 0).any():
        raise ValueError("Validation targets must be finite, positive trip durations in minutes.")

    X = dataframe.drop(columns=[TARGET])
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is not None:
        expected_features = list(expected_features)
        missing = sorted(set(expected_features) - set(X.columns))
        unexpected = sorted(set(X.columns) - set(expected_features))
        if missing or unexpected:
            raise ValueError(
                "Validation feature schema differs from model training. "
                f"Missing columns: {missing}; unexpected columns: {unexpected}."
            )
        X = X.loc[:, expected_features]

    non_numeric = list(X.select_dtypes(exclude=[np.number]).columns)
    if non_numeric:
        raise ValueError(f"Validation features contain non-numeric columns: {non_numeric}")
    if not np.isfinite(X.to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("Validation features contain NaN or infinite values.")
    return X, y_actual_minutes


def calculate_metrics(y_actual_minutes: np.ndarray, y_predicted_minutes: np.ndarray) -> dict[str, float]:
    """Calculate all metrics in the original unit: minutes."""
    if len(y_actual_minutes) != len(y_predicted_minutes):
        raise ValueError("Actual and predicted target arrays have different lengths.")
    if len(y_actual_minutes) == 0:
        raise ValueError("Cannot calculate metrics for an empty validation split.")
    if not np.isfinite(y_predicted_minutes).all():
        raise ValueError("Model predictions contain NaN or infinity.")

    return {
        "rmse_minutes": float(np.sqrt(mean_squared_error(y_actual_minutes, y_predicted_minutes))),
        "mae_minutes": float(mean_absolute_error(y_actual_minutes, y_predicted_minutes)),
        "r2": float(r2_score(y_actual_minutes, y_predicted_minutes)),
    }


def make_plot(
    y_actual_minutes: np.ndarray,
    y_predicted_minutes: np.ndarray,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    """Render metric cards, actual-vs-predicted scatter, and a residual histogram."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    residuals = y_predicted_minutes - y_actual_minutes

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    fig.suptitle("NYC Taxi Trip Duration — Stacking Regressor", fontsize=18, fontweight="bold")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 4])

    # Metric cards across the top.
    cards_ax = fig.add_subplot(grid[0, :])
    cards_ax.set_axis_off()
    card_specs = [
        ("Validation RMSE", f"{metrics['rmse_minutes']:.4f} min"),
        ("Validation MAE", f"{metrics['mae_minutes']:.4f} min"),
        ("Validation R²", f"{metrics['r2']:.4f}"),
    ]
    positions = [0.035, 0.355, 0.675]
    for x, (label, value) in zip(positions, card_specs):
        card = FancyBboxPatch(
            (x, 0.10), 0.29, 0.78,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            transform=cards_ax.transAxes,
            facecolor="#f2f4f7",
            edgecolor="#d1d5db",
            linewidth=1.1,
        )
        cards_ax.add_patch(card)
        cards_ax.text(x + 0.145, 0.63, label, ha="center", va="center",
                      fontsize=11, color="#374151", transform=cards_ax.transAxes)
        cards_ax.text(x + 0.145, 0.35, value, ha="center", va="center",
                      fontsize=19, fontweight="bold", color="#111827",
                      transform=cards_ax.transAxes)

    # Scatter: draw a reproducible sample for manageable rendering time.
    scatter_ax = fig.add_subplot(grid[1, 0])
    n_rows = len(y_actual_minutes)
    if n_rows > MAX_SCATTER_POINTS:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_indices = np.sort(rng.choice(n_rows, size=MAX_SCATTER_POINTS, replace=False))
        x_scatter = y_actual_minutes[sample_indices]
        y_scatter = y_predicted_minutes[sample_indices]
    else:
        x_scatter = y_actual_minutes
        y_scatter = y_predicted_minutes

    scatter_ax.scatter(x_scatter, y_scatter, s=9, alpha=0.22, edgecolors="none")
    axis_max = float(max(np.max(x_scatter), np.max(y_scatter), 1.0))
    scatter_ax.plot([0, axis_max], [0, axis_max], linestyle="--", linewidth=1.5,
                    label="Ideal prediction")
    scatter_ax.set_xlim(left=0, right=axis_max)
    scatter_ax.set_ylim(bottom=0, top=axis_max)
    scatter_ax.set_xlabel("Actual trip duration (minutes)")
    scatter_ax.set_ylabel("Predicted trip duration (minutes)")
    scatter_ax.set_title(f"Actual vs. predicted ({len(x_scatter):,} plotted rows)")
    scatter_ax.grid(True, alpha=0.25)
    scatter_ax.legend(loc="upper left")

    # Residual histogram is clipped to its 1st–99th percentile range for readability.
    residual_ax = fig.add_subplot(grid[1, 1])
    low, high = np.quantile(residuals, [0.01, 0.99])
    central_residuals = residuals[(residuals >= low) & (residuals <= high)]
    residual_ax.hist(central_residuals, bins=60, edgecolor="white", linewidth=0.35)
    residual_ax.axvline(0, linestyle="--", linewidth=1.5, label="Zero error")
    residual_ax.set_xlabel("Prediction error (predicted − actual, minutes)")
    residual_ax.set_ylabel("Count")
    residual_ax.set_title("Residual distribution (central 98% shown)")
    residual_ax.grid(True, axis="y", alpha=0.25)
    residual_ax.legend()

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    model_path = root / "models" / "models" / MODEL_NAME
    output_path = root / "plots" / "model_results" / "results.png"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Stacking model not found: {model_path}. Run src/models/train_model.py first."
        )

    validation_path, validation_data = load_validation_data(root, sys.argv[1:])
    model = joblib.load(model_path)
    X_val, y_actual_minutes = prepare_features(model, validation_data)

    plot_logger.save_logs(
        f"Loaded stacking model={model_path}; validation={validation_path}; "
        f"rows={len(validation_data):,}; features={X_val.shape[1]}."
    )
    predictions_log = np.asarray(model.predict(X_val), dtype=np.float64).reshape(-1)
    if not np.isfinite(predictions_log).all():
        raise ValueError("Model produced NaN or infinite log-duration predictions.")
    with np.errstate(over="raise", invalid="raise"):
        try:
            predictions_minutes = np.exp(predictions_log)
        except FloatingPointError as exc:
            raise ValueError("Exponentiating log predictions overflowed.") from exc

    metrics = calculate_metrics(y_actual_minutes, predictions_minutes)
    make_plot(y_actual_minutes, predictions_minutes, metrics, output_path)

    plot_logger.save_logs(
        f"Saved stacking validation plot to {output_path}; "
        f"RMSE={metrics['rmse_minutes']:.6f} min; "
        f"MAE={metrics['mae_minutes']:.6f} min; R2={metrics['r2']:.6f}."
    )
    print("Stacking model validation diagnostics:")
    print(f"RMSE: {metrics['rmse_minutes']:.4f} minutes")
    print(f"MAE:  {metrics['mae_minutes']:.4f} minutes")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        plot_logger.save_logs("Plotting pipeline failed.", log_level="exception")
        raise
