"""
Variance Analysis and Statistical Testing
Runs models multiple times with different seeds and performs t-tests
"""

import sys
from pathlib import Path

# add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from src.processes.temporal_split import load_and_split_data
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
from src.visualizations.interpretability import plot_shap_analysis, plot_feature_importance_comparison


def run_xgboost_trials(train_df, val_df, test_df, seeds, categorical_cols, numeric_cols, target_col):
    # run xgboost with multiple seeds
    print("\nRunning XGBoost trials...")
    results = []
    saved_model = None
    saved_X_test = None

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
        'n_jobs': -1
    }

    for i, seed in enumerate(seeds, 1):
        print(f"  Trial {i}/{len(seeds)} (seed={seed})...")

        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)

        # prepare and train
        X_train, encoders = prepare_features_xgb(train_df, categorical_cols, numeric_cols)
        X_val, _ = prepare_features_xgb(val_df, categorical_cols, numeric_cols)
        X_test = encode_test_features_xgb(test_df[categorical_cols + numeric_cols],
                                          categorical_cols, encoders)

        params['random_state'] = seed
        model = train_xgboost_model(X_train, y_train_log, X_val, y_val_log, params)

        # predict and evaluate
        y_pred_log = predict_xgboost(model, X_test)
        y_pred = np.expm1(y_pred_log)
        metrics = evaluate_with_predictions(y_test, y_pred, f'XGB_seed{seed}')

        results.append(metrics)

        # save first trial model for SHAP analysis
        if i == 1:
            saved_model = model
            saved_X_test = X_test

    return results, saved_model, saved_X_test


def run_random_forest_trials(train_df, test_df, seeds, categorical_cols, numeric_cols, target_col):
    # run random forest with multiple seeds
    print("\nRunning Random Forest trials...")
    results = []
    saved_model = None
    saved_X_test = None

    params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 100,
        'min_samples_leaf': 50,
        'max_features': 0.6,
        'bootstrap': True,
        'n_jobs': -1
    }

    for i, seed in enumerate(seeds, 1):
        print(f"  Trial {i}/{len(seeds)} (seed={seed})...")

        y_train = train_df[target_col].values
        y_test = test_df[target_col].values
        y_train_log = np.log1p(y_train)

        # prepare and train
        X_train, encoders = prepare_features_rf(train_df, categorical_cols, numeric_cols)
        X_test = encode_test_features(test_df[categorical_cols + numeric_cols],
                                      categorical_cols, encoders)

        params['random_state'] = seed
        model = train_random_forest_model(X_train, y_train_log, params)

        # predict and evaluate
        y_pred_log = predict_random_forest(model, X_test)
        y_pred = np.expm1(y_pred_log)
        metrics = evaluate_with_predictions(y_test, y_pred, f'RF_seed{seed}')

        results.append(metrics)

        # save first trial model for SHAP analysis
        if i == 1:
            saved_model = model
            saved_X_test = X_test

    return results, saved_model, saved_X_test


def run_lightgbm_trials(train_df, val_df, test_df, seeds, categorical_cols, numeric_cols, target_col):
    # run lightgbm with multiple seeds
    print("\nRunning LightGBM trials...")
    results = []
    saved_model = None
    saved_X_test = None

    params = {
        'objective': 'mae',
        'metric': 'mae',
        'num_leaves': 31,
        'max_depth': 8,
        'learning_rate': 0.03,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'min_data_in_leaf': 100,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'verbose': -1
    }

    for i, seed in enumerate(seeds, 1):
        print(f"  Trial {i}/{len(seeds)} (seed={seed})...")

        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)

        # prepare and train
        X_train = prepare_features(train_df, categorical_cols, numeric_cols)
        X_val = prepare_features(val_df, categorical_cols, numeric_cols)
        X_test = prepare_features(test_df, categorical_cols, numeric_cols)

        params['random_state'] = seed
        model = train_lightgbm_model(X_train, y_train_log, X_val, y_val_log, params,
                                     num_iterations=1000, early_stopping_rounds=50)

        # predict and evaluate
        y_pred_log = predict_lightgbm(model, X_test)
        y_pred = np.expm1(y_pred_log)
        metrics = evaluate_with_predictions(y_test, y_pred, f'LGB_seed{seed}')

        results.append(metrics)

        # save first trial model for SHAP analysis
        if i == 1:
            saved_model = model
            saved_X_test = X_test

    return results, saved_model, saved_X_test


def analyze_variance(results_dict):
    # analyze variance across trials
    print("\n" + "="*70)
    print("VARIANCE ANALYSIS")
    print("="*70)

    summary_stats = []

    for model_name, results in results_dict.items():
        maes = [r['mae'] for r in results]
        rmses = [r['rmse'] for r in results]

        stats = {
            'Model': model_name,
            'MAE_mean': np.mean(maes),
            'MAE_std': np.std(maes),
            'MAE_min': np.min(maes),
            'MAE_max': np.max(maes),
            'RMSE_mean': np.mean(rmses),
            'RMSE_std': np.std(rmses),
            'n_trials': len(results)
        }
        summary_stats.append(stats)

        print(f"\n{model_name}:")
        print(f"  MAE: {stats['MAE_mean']:.4f} ± {stats['MAE_std']:.4f} (min={stats['MAE_min']:.4f}, max={stats['MAE_max']:.4f})")
        print(f"  RMSE: {stats['RMSE_mean']:.4f} ± {stats['RMSE_std']:.4f}")

    return pd.DataFrame(summary_stats)


def perform_ttests(results_dict):
    # perform pairwise t-tests
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS (T-tests)")
    print("="*70)

    model_names = list(results_dict.keys())
    ttest_results = []

    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]

            mae1 = [r['mae'] for r in results_dict[model1]]
            mae2 = [r['mae'] for r in results_dict[model2]]

            t_stat, p_value = ttest_ind(mae1, mae2)

            result = {
                'Model_1': model1,
                'Model_2': model2,
                'Mean_Diff': np.mean(mae1) - np.mean(mae2),
                't_statistic': t_stat,
                'p_value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            }
            ttest_results.append(result)

            print(f"\n{model1} vs {model2}:")
            print(f"  Mean difference: {result['Mean_Diff']:.4f} days")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant at α=0.05: {result['Significant']}")

    return pd.DataFrame(ttest_results)


def plot_variance_results(results_dict, save_path='results/interpretability/variance_analysis.png'):
    # visualize variance across trials
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    model_names = list(results_dict.keys())
    positions = range(len(model_names))

    # box plot
    mae_data = [[r['mae'] for r in results_dict[model]] for model in model_names]
    bp = ax1.boxplot(mae_data, positions=positions, labels=model_names,
                     patch_artist=True, widths=0.6)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax1.set_ylabel('MAE (days)', fontweight='bold')
    ax1.set_title('MAE Distribution Across Multiple Seeds', fontweight='bold', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)

    # bar plot with error bars
    means = [np.mean([r['mae'] for r in results_dict[model]]) for model in model_names]
    stds = [np.std([r['mae'] for r in results_dict[model]]) for model in model_names]

    bars = ax2.bar(positions, means, yerr=stds, capsize=10, alpha=0.8,
                   color=['steelblue', 'darkgreen', 'purple'], edgecolor='black', linewidth=1.5)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(model_names)
    ax2.set_ylabel('MAE (days)', fontweight='bold')
    ax2.set_title('Mean MAE with Standard Deviation', fontweight='bold', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)

    # add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax2.text(i, mean + std + 0.1, f'{mean:.2f}±{std:.2f}',
                ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVariance visualization saved: {save_path}")


def main():
    print("\n" + "="*70)
    print("MODEL VARIANCE ANALYSIS AND STATISTICAL TESTING")
    print("="*70)

    # load data with temporal split
    data_path = Path(__file__).parent.parent / "data/raw/311_Service_Requests_2020.csv"
    train_df, val_df, test_df = load_and_split_data(data_path=data_path, sample_size=5000, use_validation=True)

    categorical_cols = ['SR_TYPE', 'ZIP_CODE', 'ORIGIN']
    numeric_cols = ['CREATED_HOUR', 'CREATED_DAY_OF_WEEK', 'CREATED_MONTH']
    target_col = 'RESPONSE_TIME_DAYS'

    # define seeds for trials
    n_trials = 10
    seeds = list(range(42, 42 + n_trials))
    print(f"\nRunning {n_trials} trials with seeds: {seeds}")

    # run trials for each model
    results_dict = {}
    models_dict = {}
    X_test_dict = {}

    xgb_results, xgb_model, xgb_X_test = run_xgboost_trials(train_df, val_df, test_df, seeds,
                                                             categorical_cols, numeric_cols, target_col)
    results_dict['XGBoost'] = xgb_results
    models_dict['XGBoost'] = xgb_model
    X_test_dict['XGBoost'] = xgb_X_test

    rf_results, rf_model, rf_X_test = run_random_forest_trials(train_df, test_df, seeds,
                                                                categorical_cols, numeric_cols, target_col)
    results_dict['Random Forest'] = rf_results
    models_dict['Random Forest'] = rf_model
    X_test_dict['Random Forest'] = rf_X_test

    lgb_results, lgb_model, lgb_X_test = run_lightgbm_trials(train_df, val_df, test_df, seeds,
                                                              categorical_cols, numeric_cols, target_col)
    results_dict['LightGBM'] = lgb_results
    models_dict['LightGBM'] = lgb_model
    X_test_dict['LightGBM'] = lgb_X_test

    # analyze variance
    variance_summary = analyze_variance(results_dict)

    # perform t-tests
    ttest_results = perform_ttests(results_dict)

    # visualize
    plot_variance_results(results_dict)

    # save results
    variance_summary.to_csv('results/interpretability/variance_summary.csv', index=False)
    ttest_results.to_csv('results/interpretability/ttest_results.csv', index=False)

    # generate shap analysis for each model
    print("\n" + "="*70)
    print("SHAP INTERPRETABILITY ANALYSIS")
    print("="*70)

    feature_names = categorical_cols + numeric_cols

    for model_name in ['XGBoost', 'Random Forest', 'LightGBM']:
        print(f"\nGenerating SHAP analysis for {model_name}...")
        model = models_dict[model_name]
        X_test = X_test_dict[model_name]

        # use subset of test data for faster SHAP computation
        X_test_sample = X_test[:500] if len(X_test) > 500 else X_test

        try:
            plot_shap_analysis(model, X_test_sample, feature_names, model_name,
                             save_path='results/interpretability/')
        except Exception as e:
            print(f"  Warning: SHAP analysis failed for {model_name}: {e}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nResults saved:")
    print("  - results/interpretability/variance_summary.csv")
    print("  - results/interpretability/ttest_results.csv")
    print("  - results/interpretability/variance_analysis.png")
    print("  - results/interpretability/XGBoost_shap_summary.png")
    print("  - results/interpretability/XGBoost_shap_bar.png")
    print("  - results/interpretability/Random Forest_shap_summary.png")
    print("  - results/interpretability/Random Forest_shap_bar.png")
    print("  - results/interpretability/LightGBM_shap_summary.png")
    print("  - results/interpretability/LightGBM_shap_bar.png")


if __name__ == "__main__":
    main()
