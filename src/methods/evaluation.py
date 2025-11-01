"""Simple evaluation helpers."""

def evaluate_regression(model, features, target):
    """Return a dictionary with a few basic regression metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    predictions = model.predict(features)
    return {
        "r2": r2_score(target, predictions),
        "rmse": mean_squared_error(target, predictions, squared=False),
        "mae": mean_absolute_error(target, predictions),
    }
