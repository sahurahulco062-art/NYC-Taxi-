
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # Project paths
    root_path = Path(__file__).resolve().parents[2]

    metrics_path = root_path / "reports" / "metrics" / "validation_metrics.json"
    validation_path = root_path / "data" / "processed" / "final" / "val.csv"
    model_path = root_path / "models" / "models" / "xgbreg.joblib"
    transformer_path = (
        root_path / "models" / "transformers" / "output_transformer.joblib"
    )

    output_dir = root_path / "plots" / "model_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.png"

    target_column = "trip_duration"

    # Load validation metrics
    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    validation_metrics = metrics["validation_metrics"]
    rmse = validation_metrics["rmse_minutes"]
    mae = validation_metrics["mae_minutes"]
    r2 = validation_metrics["r2"]

    # Load the trained model, target transformer, and validation data
    model = joblib.load(model_path)
    target_transformer = joblib.load(transformer_path)
    validation_data = pd.read_csv(validation_path)

    X_val = validation_data.drop(columns=[target_column])
    y_val_transformed = validation_data[target_column].to_numpy().reshape(-1, 1)

    # Convert actual and predicted targets back to minutes
    y_actual = target_transformer.inverse_transform(
        y_val_transformed
    ).ravel()

    y_pred_transformed = model.predict(X_val).reshape(-1, 1)
    y_predicted = target_transformer.inverse_transform(
        y_pred_transformed
    ).ravel()

    # Create dashboard
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 4])

    fig.suptitle(
        "NYC Taxi Trip Duration | XGBoost Validation Dashboard",
        fontsize=17,
        fontweight="bold",
    )

    # Separate metric cards
    metric_cards = [
        ("RMSE", f"{rmse:.4f} min", "Lower is better"),
        ("MAE", f"{mae:.4f} min", "Lower is better"),
        ("R²", f"{r2:.4f}", "Closer to 1 is better"),
    ]

    for column, (name, value, description) in enumerate(metric_cards):
        ax = fig.add_subplot(grid[0, column])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)

        ax.text(
            0.5, 0.72, name,
            ha="center", va="center",
            fontsize=13, fontweight="bold",
        )
        ax.text(
            0.5, 0.43, value,
            ha="center", va="center",
            fontsize=20, fontweight="bold",
        )
        ax.text(
            0.5, 0.14, description,
            ha="center", va="center",
            fontsize=9,
        )

    # Actual vs predicted chart
    ax = fig.add_subplot(grid[1, :])

    # Sample points if the validation set is large
    rng = np.random.default_rng(42)
    sample_size = min(10000, len(y_actual))
    sample_indices = rng.choice(
        len(y_actual), size=sample_size, replace=False
    )

    actual_sample = y_actual[sample_indices]
    predicted_sample = y_predicted[sample_indices]

    ax.scatter(
        actual_sample,
        predicted_sample,
        alpha=0.25,
        s=12,
        label=f"Validation samples (n={sample_size:,})",
    )

    # Ideal predictions lie on y = x
    lower = min(actual_sample.min(), predicted_sample.min())
    upper = max(actual_sample.max(), predicted_sample.max())

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=2,
        label="Ideal prediction",
    )

    ax.set_title("Actual vs. Predicted Trip Duration", fontsize=13)
    ax.set_xlabel("Actual trip duration (minutes)")
    ax.set_ylabel("Predicted trip duration (minutes)")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logging.info("Dashboard saved to %s", output_path)
    print(f"Dashboard saved: {output_path}")


if __name__ == "__main__":
    main()

