
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def calculate_rmse(actual_transformed, predicted_transformed, transformer):
    actual = transformer.inverse_transform(
        actual_transformed.reshape(-1, 1)
    ).ravel()

    predicted = transformer.inverse_transform(
        predicted_transformed.reshape(-1, 1)
    ).ravel()

    return float(np.sqrt(mean_squared_error(actual, predicted)))


def main():
    root = Path(__file__).resolve().parents[2]

    model_path = root / "models" / "models" / "xgbreg.joblib"
    transformer_path = (
        root / "models" / "transformers" / "output_transformer.joblib"
    )
    train_path = root / "data" / "processed" / "final" / "train.csv"
    val_path = root / "data" / "processed" / "final" / "val.csv"

    output_dir = root / "plots" / "model_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing artifacts; do not retrain
    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path)

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    target = "trip_duration"
    X_train = train.drop(columns=[target])
    y_train = train[target].to_numpy()
    X_val = val.drop(columns=[target])
    y_val = val[target].to_numpy()

    # Use a deterministic training sample to keep repeated predictions practical.
    # Validation metrics use the full validation dataset.
    sample_size = min(100_000, len(train))
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        len(train), size=sample_size, replace=False
    )

    X_train_sample = X_train.iloc[sample_indices]
    y_train_sample = y_train[sample_indices]

    total_trees = model.get_params().get("n_estimators", 100)
    checkpoints = sorted(
        set(
            np.linspace(
                1, total_trees, num=min(20, total_trees), dtype=int
            ).tolist()
        )
    )

    results = []

    print(f"Model trees: {total_trees}")
    print(f"Training sample: {sample_size:,} rows")
    print(f"Validation dataset: {len(val):,} rows")
    print("Evaluating successive boosting iterations...")

    for iteration in checkpoints:
        train_predictions = model.predict(
            X_train_sample,
            iteration_range=(0, iteration),
        )
        val_predictions = model.predict(
            X_val,
            iteration_range=(0, iteration),
        )

        train_rmse = calculate_rmse(
            y_train_sample, train_predictions, transformer
        )
        val_rmse = calculate_rmse(
            y_val, val_predictions, transformer
        )

        results.append(
            {
                "trees": iteration,
                "train_rmse_minutes": train_rmse,
                "validation_rmse_minutes": val_rmse,
                "rmse_gap_minutes": val_rmse - train_rmse,
            }
        )

        print(
            f"Trees={iteration:4d} | "
            f"Train RMSE={train_rmse:.4f} | "
            f"Validation RMSE={val_rmse:.4f} | "
            f"Gap={val_rmse - train_rmse:.4f} minutes"
        )

    results_df = pd.DataFrame(results)

    csv_path = output_dir / "learning_curve.csv"
    results_df.to_csv(csv_path, index=False)

    # Plot both RMSE curves
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        results_df["trees"],
        results_df["train_rmse_minutes"],
        marker="o",
        label="Training RMSE (100k-row sample)",
    )
    ax.plot(
        results_df["trees"],
        results_df["validation_rmse_minutes"],
        marker="o",
        label="Validation RMSE (full dataset)",
    )

    ax.set_title("XGBoost Overfitting Analysis")
    ax.set_xlabel("Number of boosting trees")
    ax.set_ylabel("RMSE (minutes)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plot_path = output_dir / "learning_curve.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # A simple diagnostic, not a definitive statistical test
    best_row = results_df.loc[
        results_df["validation_rmse_minutes"].idxmin()
    ]

    print("\nLearning-curve analysis complete.")
    print(f"Best checkpoint: {int(best_row['trees'])} trees")
    print(
        "Lowest validation RMSE across checkpoints: "
        f"{best_row['validation_rmse_minutes']:.4f} minutes"
    )
    print(f"Plot saved: {plot_path}")
    print(f"Results saved: {csv_path}")
    print(
        "\nNote: the lowest checkpoint is diagnostic. "
        "The saved model has not been changed."
    )


if __name__ == "__main__":
    main()

