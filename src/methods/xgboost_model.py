import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def build_xgboost_model(params=None):
    # default xgboost parameters optimized for this dataset
    default_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'max_depth': 8,
        'learning_rate': 0.03,
        'n_estimators': 500,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 100,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'random_state': 42,
        'n_jobs': -1
    }

    if params:
        default_params.update(params)

    return default_params


def prepare_features_xgb(df, categorical_cols, numeric_cols):
    # label encode categorical features for xgboost
    X = df[categorical_cols + numeric_cols].copy()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    return X, label_encoders


def encode_test_features_xgb(X_test, categorical_cols, label_encoders):
    # encode test features using fitted encoders
    X_test_encoded = X_test.copy()

    for col in categorical_cols:
        le = label_encoders[col]
        X_test_encoded[col] = X_test_encoded[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    return X_test_encoded


def train_xgboost_model(X_train, y_train, X_val, y_val, params):
    # train xgboost model with early stopping
    n_estimators = params.pop('n_estimators', 500)

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    return model


def predict_xgboost(model, X):
    # make predictions
    return model.predict(X)


def get_feature_importance_xgb(model, feature_names=None, importance_type='gain'):
    # get feature importance from trained model
    if importance_type == 'gain':
        importance = model.feature_importances_
    elif importance_type == 'weight':
        importance = model.get_booster().get_score(importance_type='weight')
        importance = [importance.get(f'f{i}', 0) for i in range(len(feature_names or []))]
    else:
        importance = model.feature_importances_

    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importance))]

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return importance_df
