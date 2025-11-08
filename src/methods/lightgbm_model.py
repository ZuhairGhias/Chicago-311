import lightgbm as lgb
import pandas as pd
import numpy as np


def build_lightgbm_model(params=None):
    # default lightgbm parameters optimized for this dataset
    default_params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.0,
        'lambda_l2': 0.0,
    }

    if params:
        default_params.update(params)

    return default_params


def prepare_features(df, categorical_cols, numeric_cols):
    # encode categorical features for lightgbm (handles categoricals natively)
    X = df[categorical_cols + numeric_cols].copy()

    for col in categorical_cols:
        X[col] = X[col].astype('category')

    return X


def train_lightgbm_model(X_train, y_train, X_val, y_val, params, num_iterations=1000, early_stopping_rounds=50):
    # train lightgbm model with early stopping
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_iterations,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=50)
        ]
    )

    return model


def predict_lightgbm(model, X):
    # make predictions
    return model.predict(X, num_iteration=model.best_iteration)


def get_feature_importance(model, feature_names=None, importance_type='gain'):
    # get feature importance from trained model
    importance = model.feature_importance(importance_type=importance_type)

    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importance))]

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return importance_df
