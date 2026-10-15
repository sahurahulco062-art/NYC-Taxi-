
import pandas as pd

from src.models.train_model import (
    apply_original_cleaning_and_feature_engineering,
)
from src.features.modify_features import (
    input_modifications,
    target_modifications,
)
from src.features.build_features import implement_distances

# Load a small sample from the original training split.
raw = pd.read_csv("data/interim/train.csv", nrows=1000)

# Process it using the original project functions.
original = input_modifications(raw.copy())
original = target_modifications(original)
original = implement_distances(
    original,
    original["pickup_latitude"],
    original["pickup_longitude"],
    original["dropoff_latitude"],
    original["dropoff_longitude"],
)

# Process the same sample using the CV helper.
cv_version = apply_original_cleaning_and_feature_engineering(raw.copy())

# Compare columns, row counts, and values.
print("Original shape:", original.shape)
print("CV helper shape:", cv_version.shape)
print("Same columns:", set(original.columns) == set(cv_version.columns))
print("Same row count:", len(original) == len(cv_version))

if set(original.columns) == set(cv_version.columns):
    original = original.sort_index(axis=1)
    cv_version = cv_version.sort_index(axis=1)

    print(
        "Same values:",
        original.sort_index().equals(cv_version.sort_index()),
    )

    print("Original columns:", sorted(original.columns))
    print("CV columns:", sorted(cv_version.columns))
