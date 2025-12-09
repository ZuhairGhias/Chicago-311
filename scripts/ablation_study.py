"""
Feature Ablation Study
Systematically removes feature groups to measure their contribution to model performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.processes.temporal_split import load_and_split_data
from src.methods.random_forest_model import prepare_features_rf, train_random_forest_model, predict_random_forest, encode_test_features
from src.methods.evaluation import evaluate_with_predictions


def run_ablation_experiment(train_df, test_df, y_test, categorical_cols, numeric_cols,
                            experiment_name):
    """Run Random Forest with specific feature subset."""
    print(f"\nRunning: {experiment_name}")
    print(f"  Categorical: {categorical_cols}")
    print(f"  Numeric: {numeric_cols}")

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

    metrics = evaluate_with_predictions(y_test, y_pred, experiment_name)
    print(f"  MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.6f}")

    return metrics


def main():
    print("\n" + "="*70)
    print("FEATURE ABLATION STUDY - RANDOM FOREST")
    print("="*70 + "\n")

    # Load data
    train_df, _, test_df = load_and_split_data(sample_size=None, use_validation=True)
    y_test = test_df['RESPONSE_TIME_DAYS'].values

    # Define feature groups
    all_categorical = ['SR_TYPE', 'ZIP_CODE', 'ORIGIN']
    all_numeric = ['CREATED_HOUR', 'CREATED_DAY_OF_WEEK', 'CREATED_MONTH']

    ablation_experiments = [
        {
            'name': 'Full_Model',
            'categorical': all_categorical,
            'numeric': all_numeric
        },
        {
            'name': 'No_SR_TYPE',
            'categorical': ['ZIP_CODE', 'ORIGIN'],
            'numeric': all_numeric
        },
        {
            'name': 'No_ZIP_CODE',
            'categorical': ['SR_TYPE', 'ORIGIN'],
            'numeric': all_numeric
        },
        {
            'name': 'No_ORIGIN',
            'categorical': ['SR_TYPE', 'ZIP_CODE'],
            'numeric': all_numeric
        },
        {
            'name': 'No_Temporal',
            'categorical': all_categorical,
            'numeric': []
        },
        {
            'name': 'SR_TYPE_Only',
            'categorical': ['SR_TYPE'],
            'numeric': []
        },
        {
            'name': 'ZIP_CODE_Only',
            'categorical': ['ZIP_CODE'],
            'numeric': []
        },
        {
            'name': 'Temporal_Only',
            'categorical': [],
            'numeric': all_numeric
        }
    ]

    results = []
    for exp in ablation_experiments:
        metrics = run_ablation_experiment(
            train_df, test_df, y_test,
            exp['categorical'], exp['numeric'],
            exp['name']
        )
        results.append(metrics)

    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mae')

    output_path = Path('results/ablation_results.csv')
    results_df.to_csv(output_path, index=False)

    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    print(f"\n{'Experiment':<25} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
    print("-"*70)
    for _, row in results_df.iterrows():
        print(f"{row['model_name']:<25} {row['mae']:<12.4f} {row['rmse']:<12.4f} {row['r2']:<12.6f}")
    print("-"*70)

    # Calculate impact
    full_model = results_df[results_df['model_name'] == 'Full_Model'].iloc[0]
    print(f"\n\nFeature Impact Analysis (vs Full Model MAE: {full_model['mae']:.4f}):")
    print("-"*70)
    for _, row in results_df.iterrows():
        if row['model_name'] != 'Full_Model':
            impact = row['mae'] - full_model['mae']
            pct_change = (impact / full_model['mae']) * 100
            print(f"{row['model_name']:<25} +{impact:>6.4f} days (+{pct_change:>5.2f}%)")

    print(f"\n\nResults saved to: {output_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
