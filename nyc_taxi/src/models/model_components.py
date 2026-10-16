
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from src.features.data_preprocessing import (
    remove_outliers,
    train_preprocessor,
    transform_data,
)


OUTLIER_COLUMNS = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]


def fold_local_outlier_filter(
    X: pd.DataFrame,
    y=None,
    percentiles: list[float] | tuple[float, float] = (0.002, 0.998),
    column_names: list[str] | tuple[str, ...] = tuple(OUTLIER_COLUMNS),
):
    """Remove outliers using thresholds fitted on the current training fold."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "Fold-local outlier filtering expects a pandas DataFrame."
        )

    outlier_transformer = remove_outliers(
        dataframe=X,
        percentiles=list(percentiles),
        column_names=list(column_names),
    )

    X_filtered = transform_data(outlier_transformer, X)

    if len(X_filtered) == 0:
        raise ValueError(
            "Outlier filtering removed every row in a CV training fold."
        )

    if y is None:
        return X_filtered

    if isinstance(y, pd.Series):
        if not X_filtered.index.isin(y.index).all():
            raise ValueError(
                "Filtered feature indices do not match target indices."
            )
        y_filtered = y.loc[X_filtered.index]
    else:
        y_series = pd.Series(
            np.asarray(y).reshape(-1),
            index=X.index,
        )
        if not X_filtered.index.isin(y_series.index).all():
            raise ValueError(
                "Filtered feature indices do not match target indices."
            )
        y_filtered = y_series.loc[X_filtered.index].to_numpy()

    if len(X_filtered) != len(y_filtered):
        raise ValueError(
            "Fold-local outlier filtering produced mismatched X and y lengths."
        )

    return X_filtered, y_filtered


class ExistingProjectPreprocessor(BaseEstimator, TransformerMixin):
    """Use the project's existing preprocessing functions."""

    def fit(self, X, y=None):
        self.preprocessor_ = train_preprocessor(X)
        return self

    def transform(self, X):
        return transform_data(self.preprocessor_, X)
