"""Lightweight modeling helpers."""
from typing import Iterable, Tuple


def build_regression_model(
    numeric_features: Iterable[str] | Tuple[str, ...] = (),
    categorical_features: Iterable[str] | Tuple[str, ...] = (),
):
    """Create a simple regression pipeline with optional preprocessing."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    transformers = []
    if numeric_features:
        transformers.append(("numeric", "passthrough", list(numeric_features)))
    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(categorical_features),
            )
        )

    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers)
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        )

    return LinearRegression()


def train_regression_model(model, features, target, **fit_kwargs):
    """Fit ``model`` on ``features`` and ``target`` and return it."""
    model.fit(features, target, **fit_kwargs)
    return model
