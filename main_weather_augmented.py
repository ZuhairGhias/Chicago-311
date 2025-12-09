#!/usr/bin/env python3
"""
Chicago 311 Response Time Prediction - Weather-Augmented Model

This script runs the same modeling pipeline as main.py but uses the weather-augmented
dataset with 7 additional weather features:
    - Base metrics: temp_mean, precipitation, snowfall
    - Derived features: extreme_cold, heavy_precipitation, snow_day, temp_deviation

Usage:
    python main_weather_augmented.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path

# import model functions
from src.methods.baseline import get_all_prior_year_average
from src.methods.xgboost_model import (
    prepare_features_xgb, train_xgboost_model,
    predict_xgboost, encode_test_features_xgb
)
from src.methods.random_forest_model import (
    prepare_features_rf, train_random_forest_model,
    predict_random_forest, encode_test_features
)
from src.methods.lightgbm_model import (
    prepare_features, train_lightgbm_model, predict_lightgbm
)
from src.methods.evaluation import evaluate_with_predictions


def load_weather_augmented_data(sample_size=None):
    """
    Load the weather-augmented dataset and perform temporal train/test split.

    Args:
        sample_size: Number of samples to use (None for full dataset)

    Returns:
        train_df, test_df: Train and test DataFrames
    """
    print("Loading weather-augmented dataset...")
    data_path = Path('data/processed/311_Service_Requests_Since_2020_with_weather.csv')

    if not data_path.exists():
        raise FileNotFoundError(
            f"Weather-augmented dataset not found at {data_path}\n"
            f"Please run: python scripts/data_augmentation.py"
        )

    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df):,} rows with {df.shape[1]} columns")

    # Convert dates
    df['CREATED_DATE'] = pd.to_datetime(df['CREATED_DATE'], errors='coerce')
    df['CLOSED_DATE'] = pd.to_datetime(df['CLOSED_DATE'], errors='coerce')

    # Calculate response time
    df['RESPONSE_TIME_DAYS'] = (df['CLOSED_DATE'] - df['CREATED_DATE']).dt.total_seconds() / 86400

    # Filter valid response times
    df = df[df['RESPONSE_TIME_DAYS'] > 0].copy()
    df = df[df['RESPONSE_TIME_DAYS'].notna()].copy()

    # Apply 99th percentile capping
    p99 = df['RESPONSE_TIME_DAYS'].quantile(0.99)
    print(f"✓ 99th percentile: {p99:.2f} days")
    df = df[df['RESPONSE_TIME_DAYS'] <= p99].copy()

    # Remove rows with missing weather data
    weather_cols = ['temp_mean', 'precipitation', 'snowfall']
    before_drop = len(df)
    df = df.dropna(subset=weather_cols)
    after_drop = len(df)
    print(f"✓ Dropped {before_drop - after_drop:,} rows with missing weather data")

    # Temporal split (train: before 2024, test: 2024+)
    cutoff_date = pd.Timestamp('2024-01-01')
    train_df = df[df['CREATED_DATE'] < cutoff_date].copy()
    test_df = df[df['CREATED_DATE'] >= cutoff_date].copy()

    print(f"✓ Train set: {len(train_df):,} rows (before {cutoff_date.date()})")
    print(f"✓ Test set: {len(test_df):,} rows (on/after {cutoff_date.date()})")

    # Sample if requested
    if sample_size is not None:
        if len(train_df) > sample_size:
            train_df = train_df.sample(n=sample_size, random_state=42)
            print(f"✓ Sampled train set to {sample_size:,} rows")

        # Keep proportional test set
        test_proportion = min(1.0, sample_size / len(test_df))
        if test_proportion < 1.0:
            test_sample_size = int(len(test_df) * test_proportion)
            test_df = test_df.sample(n=test_sample_size, random_state=42)
            print(f"✓ Sampled test set to {len(test_df):,} rows")

    return train_df, test_df


def get_ts_cv_splits(df, n_splits=5):
    """Get time-series cross-validation splits.

    Returns list of (train_idx, val_idx) tuples for temporal CV.
    Data must be sorted by date before calling.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(df))


def run_baseline_methods(train_df, test_df, y_test):
    # run all 4 baseline configurations
    print("="*70)
    print("RUNNING BASELINE METHODS")
    print("="*70)

    baseline_configs = [
        {'name': 'Baseline_0_Global', 'features': []},
        {'name': 'Baseline_1_SR_TYPE', 'features': ['SR_TYPE']},
        {'name': 'Baseline_2_ZIP', 'features': ['ZIP_CODE']},
        {'name': 'Baseline_3_SR_ZIP', 'features': ['SR_TYPE', 'ZIP_CODE']}
    ]

    baseline_results = []
    for config in baseline_configs:
        preds = get_all_prior_year_average(test_df, train_df, config['features'])
        metrics = evaluate_with_predictions(y_test, preds.values, config['name'])
        baseline_results.append(metrics)
        print(f"{config['name']:25s} MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")

    print()
    return baseline_results


def run_xgboost(train_df, test_df, y_test, categorical_cols, numeric_cols, n_splits=5):
    """Run XGBoost with time-series CV to find optimal iterations."""
    print("="*70)
    print("RUNNING XGBOOST (Time-Series CV) - WITH WEATHER FEATURES")
    print("="*70)

    params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'max_depth': 6,
        'learning_rate': 0.02,
        'n_estimators': 800,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 150,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }

    target_col = 'RESPONSE_TIME_DAYS'
    cv_splits = get_ts_cv_splits(train_df, n_splits=n_splits)
    best_iterations = []

    # CV to find optimal iterations
    print(f"Running {n_splits}-fold time-series CV...")
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]

        X_train, encoders = prepare_features_xgb(fold_train, categorical_cols, numeric_cols)
        X_val, _ = prepare_features_xgb(fold_val, categorical_cols, numeric_cols)

        y_train_log = np.log1p(fold_train[target_col].values)
        y_val_log = np.log1p(fold_val[target_col].values)

        model = train_xgboost_model(X_train, y_train_log, X_val, y_val_log, params)
        best_iterations.append(model.best_iteration)
        print(f"  Fold {fold+1}: best_iteration = {model.best_iteration}")

    # Train final model on full training data with averaged iterations
    avg_iterations = int(np.mean(best_iterations))
    print(f"  Average best iterations: {avg_iterations}")

    final_params = params.copy()
    final_params['n_estimators'] = avg_iterations

    X_train_full, encoders = prepare_features_xgb(train_df, categorical_cols, numeric_cols)
    y_train_full_log = np.log1p(train_df[target_col].values)
    X_test = encode_test_features_xgb(test_df[categorical_cols + numeric_cols],
                                       categorical_cols, encoders)

    # Train without early stopping (fixed iterations)
    from xgboost import XGBRegressor
    final_model = XGBRegressor(**final_params)
    final_model.fit(X_train_full, y_train_full_log)

    y_pred_log = predict_xgboost(final_model, X_test)
    y_pred = np.expm1(y_pred_log)

    metrics = evaluate_with_predictions(y_test, y_pred, 'XGB_Conservative')
    print(f"XGB_Conservative          MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}\n")

    return metrics


def run_random_forest(train_df, test_df, y_test, categorical_cols, numeric_cols):
    """Run Random Forest on full training data (no early stopping needed)."""
    print("="*70)
    print("RUNNING RANDOM FOREST - WITH WEATHER FEATURES")
    print("="*70)

    params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 100,
        'min_samples_leaf': 50,
        'max_features': 0.6,
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1
    }

    target_col = 'RESPONSE_TIME_DAYS'
    y_train_log = np.log1p(train_df[target_col].values)

    X_train, encoders = prepare_features_rf(train_df, categorical_cols, numeric_cols)
    X_test = encode_test_features(test_df[categorical_cols + numeric_cols],
                                   categorical_cols, encoders)

    model = train_random_forest_model(X_train, y_train_log, params)
    y_pred_log = predict_random_forest(model, X_test)
    y_pred = np.expm1(y_pred_log)

    metrics = evaluate_with_predictions(y_test, y_pred, 'RF_Conservative')
    print(f"RF_Conservative           MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}\n")

    return metrics


def run_lightgbm_ensemble(train_df, test_df, y_test, categorical_cols, numeric_cols, n_splits=5):
    """Run LightGBM ensemble with time-series CV to find optimal iterations."""
    print("="*70)
    print("RUNNING LIGHTGBM ENSEMBLE (Time-Series CV) - WITH WEATHER FEATURES")
    print("="*70)

    params = {
        'objective': 'mae',
        'metric': 'mae',
        'num_leaves': 45,
        'max_depth': 9,
        'learning_rate': 0.04,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'min_data_in_leaf': 40,
        'lambda_l1': 0.2,
        'lambda_l2': 0.2,
        'verbose': -1
    }

    target_col = 'RESPONSE_TIME_DAYS'
    cv_splits = get_ts_cv_splits(train_df, n_splits=n_splits)

    # CV to find optimal iterations
    print(f"Running {n_splits}-fold time-series CV to find optimal iterations...")
    best_iterations = []
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]

        X_train = prepare_features(fold_train, categorical_cols, numeric_cols)
        X_val = prepare_features(fold_val, categorical_cols, numeric_cols)

        y_train_log = np.log1p(fold_train[target_col].values)
        y_val_log = np.log1p(fold_val[target_col].values)

        params['random_state'] = 42
        model = train_lightgbm_model(X_train, y_train_log, X_val, y_val_log, params,
                                     num_iterations=1000, early_stopping_rounds=50)
        best_iterations.append(model.best_iteration)
        print(f"  Fold {fold+1}: best_iteration = {model.best_iteration}")

    avg_iterations = int(np.mean(best_iterations))
    print(f"  Average best iterations: {avg_iterations}")

    # Train ensemble on full data with averaged iterations
    seeds = [42, 123, 456, 789, 1011]
    all_preds = []

    print(f"Training {len(seeds)}-model ensemble with {avg_iterations} iterations...")
    X_train_full = prepare_features(train_df, categorical_cols, numeric_cols)
    X_test = prepare_features(test_df, categorical_cols, numeric_cols)
    y_train_full_log = np.log1p(train_df[target_col].values)

    import lightgbm as lgb
    for seed in seeds:
        params['random_state'] = seed
        train_data = lgb.Dataset(X_train_full, label=y_train_full_log)
        model = lgb.train(params, train_data, num_boost_round=avg_iterations)
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        all_preds.append(y_pred)

    # average predictions
    y_pred_final = np.mean(all_preds, axis=0)
    metrics = evaluate_with_predictions(y_test, y_pred_final, 'LightGBM_Ensemble')
    print(f"LightGBM_Ensemble         MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}\n")

    return metrics


def create_visualization(all_results, output_suffix='_weather'):
    # create comparison visualization
    print("="*70)
    print("CREATING VISUALIZATION")
    print("="*70)

    # select best models for visualization
    results_df = pd.DataFrame(all_results)
    selected_models = ['XGB_Conservative', 'RF_Conservative', 'LightGBM_Ensemble', 'Baseline_1_SR_TYPE']
    plot_df = results_df[results_df['model_name'].isin(selected_models)].copy()
    plot_df = plot_df.sort_values('mae', ascending=True)

    # clean labels
    plot_df['clean_label'] = plot_df['model_name'].map({
        'Baseline_1_SR_TYPE': 'Baseline (SR_TYPE)',
        'LightGBM_Ensemble': 'LightGBM',
        'XGB_Conservative': 'XGBoost',
        'RF_Conservative': 'Random Forest'
    })

    # define colors
    def get_color(name):
        if 'Baseline' in name:
            return '#e74c3c'
        elif 'XGB' in name:
            return '#3498db'
        elif 'RF' in name:
            return '#2ecc71'
        elif 'LightGBM' in name:
            return '#9b59b6'
        return '#95a5a6'

    colors = [get_color(name) for name in plot_df['model_name']]

    # create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # plot MAE
    bars1 = ax1.barh(range(len(plot_df)), plot_df['mae'], color=colors,
                     alpha=0.85, edgecolor='black', linewidth=1.2)
    ax1.set_yticks(range(len(plot_df)))
    ax1.set_yticklabels(plot_df['clean_label'], fontsize=11, fontweight='bold')
    ax1.set_xlabel('Mean Absolute Error (days)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Comparison: MAE (Weather-Augmented)', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()

    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax1.text(row['mae'] + 0.2, i, f"{row['mae']:.2f}", va='center',
                fontsize=10, fontweight='bold')

    # plot RMSE
    bars2 = ax2.barh(range(len(plot_df)), plot_df['rmse'], color=colors,
                     alpha=0.85, edgecolor='black', linewidth=1.2)
    ax2.set_yticks(range(len(plot_df)))
    ax2.set_yticklabels(plot_df['clean_label'], fontsize=11, fontweight='bold')
    ax2.set_xlabel('Root Mean Squared Error (days)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Comparison: RMSE (Weather-Augmented)', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()

    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax2.text(row['rmse'] + 0.3, i, f"{row['rmse']:.2f}", va='center',
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path = f'results/final_model_comparison{output_suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {output_path}\n")
    plt.close()


def print_summary(all_results):
    # print final summary
    print("="*70)
    print("FINAL SUMMARY - WEATHER-AUGMENTED MODELS")
    print("="*70)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('mae', ascending=True)

    print(f"\n{'Model':<25} {'MAE (days)':<15} {'RMSE (days)':<15} {'R²':<12}")
    print("-"*70)

    for _, row in results_df.iterrows():
        print(f"{row['model_name']:<25} {row['mae']:<15.4f} {row['rmse']:<15.4f} {row['r2']:<12.6f}")

    print("-"*70)

    best = results_df.iloc[0]
    baseline_best = results_df[results_df['model_name'].str.contains('Baseline')].iloc[0]
    improvement = ((baseline_best['mae'] - best['mae']) / baseline_best['mae'] * 100)

    print(f"\nBest Model: {best['model_name']}")
    print(f"  MAE: {best['mae']:.4f} days")
    print(f"  RMSE: {best['rmse']:.4f} days")
    print(f"  R²: {best['r2']:.6f}")
    print(f"  Improvement over best baseline: {improvement:.1f}%")
    print("\n" + "="*70)


def main():
    # main execution
    print("\n" + "="*70)
    print("CHICAGO 311 RESPONSE TIME PREDICTION - WEATHER-AUGMENTED")
    print("="*70 + "\n")

    # load weather-augmented data with temporal split
    train_df, test_df = load_weather_augmented_data(sample_size=5000)

    # sort training data by date for time-series CV
    train_df = train_df.sort_values('CREATED_DATE').reset_index(drop=True)

    # define features (base + weather features)
    categorical_cols = ['SR_TYPE', 'ZIP_CODE', 'ORIGIN']

    # Add weather features to numeric columns
    numeric_cols = [
        # Base temporal features
        'CREATED_HOUR', 'CREATED_DAY_OF_WEEK', 'CREATED_MONTH',
        # Weather base metrics
        'temp_mean', 'precipitation', 'snowfall',
        # Weather derived features
        'extreme_cold', 'heavy_precipitation', 'snow_day', 'temp_deviation'
    ]

    target_col = 'RESPONSE_TIME_DAYS'

    print(f"\nFeature Configuration:")
    print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"  Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"  Total features: {len(categorical_cols) + len(numeric_cols)}")
    print(f"  Target: {target_col}\n")

    # prepare test target
    y_test = test_df[target_col].values

    # run all models (each handles its own time-series CV internally)
    all_results = []

    baseline_results = run_baseline_methods(train_df, test_df, y_test)
    all_results.extend(baseline_results)

    xgb_result = run_xgboost(train_df, test_df, y_test, categorical_cols, numeric_cols)
    all_results.append(xgb_result)

    rf_result = run_random_forest(train_df, test_df, y_test, categorical_cols, numeric_cols)
    all_results.append(rf_result)

    lgb_result = run_lightgbm_ensemble(train_df, test_df, y_test, categorical_cols, numeric_cols)
    all_results.append(lgb_result)

    # create visualization
    create_visualization(all_results, output_suffix='_weather')

    # print summary
    print_summary(all_results)

    # save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('results/model_results_weather.csv', index=False)
    print("\nResults saved to results/model_results_weather.csv")
    print("All done!\n")


if __name__ == "__main__":
    main()
