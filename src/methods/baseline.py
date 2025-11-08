import pandas as pd
import numpy as np


def get_all_prior_year_average(test_df, train_df, feature_columns):
    # calculate baseline predictions using historical averages from training data
    # if feature_columns is empty, returns global mean
    # otherwise returns mean grouped by the specified features
    global_mean = train_df['RESPONSE_TIME_DAYS'].mean()

    if not feature_columns:
        return pd.Series(global_mean, index=test_df.index)

    if len(feature_columns) == 1:
        feature_col = feature_columns[0]
        historical_means = train_df.groupby(feature_col)['RESPONSE_TIME_DAYS'].mean()
        predictions = test_df[feature_col].map(historical_means).fillna(global_mean)

    else:
        historical_means = train_df.groupby(feature_columns)['RESPONSE_TIME_DAYS'].mean().to_dict()

        predictions = []
        for _, row in test_df.iterrows():
            key = tuple(row[col] for col in feature_columns)
            pred = historical_means.get(key, global_mean)
            predictions.append(pred)

        predictions = pd.Series(predictions, index=test_df.index)

    return predictions


def evaluate_baseline_methods(test_df, train_df, actual_column='RESPONSE_TIME_DAYS'):
    # evaluate all 4 baseline methods and return results
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    results_df = test_df.copy()

    results_df['BASELINE_PREDICTION_DAYS_0'] = get_all_prior_year_average(
        test_df, train_df, []
    )
    results_df['BASELINE_PREDICTION_DAYS_1'] = get_all_prior_year_average(
        test_df, train_df, ['SR_TYPE']
    )
    results_df['BASELINE_PREDICTION_DAYS_2'] = get_all_prior_year_average(
        test_df, train_df, ['ZIP_CODE']
    )
    results_df['BASELINE_PREDICTION_DAYS_3'] = get_all_prior_year_average(
        test_df, train_df, ['SR_TYPE', 'ZIP_CODE']
    )

    actual = results_df[actual_column]
    metrics = []

    for i in range(4):
        col = f'BASELINE_PREDICTION_DAYS_{i}'
        pred = results_df[col]

        valid_mask = ~pred.isna()

        if valid_mask.sum() > 0:
            mse = mean_squared_error(actual[valid_mask], pred[valid_mask])
            mae = mean_absolute_error(actual[valid_mask], pred[valid_mask])

            metrics.append({
                'Baseline': col,
                'MSE': mse,
                'MAE': mae,
                'Valid_Predictions': valid_mask.sum()
            })

    metrics_df = pd.DataFrame(metrics)

    return results_df, metrics_df
