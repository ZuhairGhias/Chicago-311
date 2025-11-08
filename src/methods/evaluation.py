"""Simple evaluation helpers."""

def evaluate_regression(model, features, target):
    # return basic regression metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    predictions = model.predict(features)
    mse = mean_squared_error(target, predictions)
    return {
        "r2": r2_score(target, predictions),
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(target, predictions),
    }


def evaluate_with_predictions(y_true, y_pred, model_name="Model"):
    # evaluate predictions and return metrics dictionary
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_samples': len(y_true)
    }

    return metrics


def print_evaluation_metrics(metrics):
    # print metrics in readable format
    print(f"\n{'='*60}")
    print(f"{metrics['model_name']} Evaluation Results")
    print(f"{'='*60}")
    print(f"  R² Score:     {metrics['r2']:.6f}")
    print(f"  RMSE:         {metrics['rmse']:.4f} days")
    print(f"  MAE:          {metrics['mae']:.4f} days")
    print(f"  MSE:          {metrics['mse']:.4f}")
    print(f"  Samples:      {metrics['n_samples']:,}")
    print(f"{'='*60}\n")


def compare_models(metrics_list):
    # compare multiple models and return dataframe sorted by mae
    import pandas as pd

    comparison_df = pd.DataFrame(metrics_list)
    comparison_df = comparison_df.sort_values('mae', ascending=True)

    return comparison_df[['model_name', 'r2', 'rmse', 'mae', 'mse']]
