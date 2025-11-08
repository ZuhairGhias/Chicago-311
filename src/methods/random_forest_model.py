from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def build_random_forest_model(params=None):
    # default random forest parameters optimized for this dataset
    default_params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 50,
        'min_samples_leaf': 20,
        'max_features': 0.7,
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0,
    }

    if params:
        default_params.update(params)

    return default_params


def prepare_features_rf(df, categorical_cols, numeric_cols):
    # label encode categorical features for random forest
    X = df[categorical_cols + numeric_cols].copy()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    return X, label_encoders


def train_random_forest_model(X_train, y_train, params):
    # train random forest model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    return model


def predict_random_forest(model, X):
    # make predictions
    return model.predict(X)


def get_feature_importance_rf(model, feature_names=None):
    # get feature importance from trained model
    importance = model.feature_importances_

    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importance))]

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return importance_df


def encode_test_features(X_test, categorical_cols, label_encoders):
    # encode test features using fitted encoders
    X_test_encoded = X_test.copy()

    for col in categorical_cols:
        le = label_encoders[col]
        X_test_encoded[col] = X_test_encoded[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    return X_test_encoded
