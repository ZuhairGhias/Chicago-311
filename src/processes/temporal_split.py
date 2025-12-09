"""
Temporal Train-Validation-Test Split
Implements time-based splitting that respects temporal dependencies.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_preprocess_data(data_path, sample_size=None, cap_at_p99=True):
    """Load and preprocess the 311 dataset with response time calculation."""
    print("Loading data...")
    df_raw = pd.read_csv(data_path)

    df_raw['CREATED_DATE'] = pd.to_datetime(df_raw['CREATED_DATE'])
    df_raw['CLOSED_DATE'] = pd.to_datetime(df_raw['CLOSED_DATE'])
    df_raw['RESPONSE_TIME'] = df_raw['CLOSED_DATE'] - df_raw['CREATED_DATE']
    df_raw['RESPONSE_TIME_DAYS'] = df_raw['RESPONSE_TIME'].dt.total_seconds() / 86400

    cols_to_keep = ["SR_TYPE", "ORIGIN", "CREATED_DATE", "CLOSED_DATE", "ZIP_CODE",
                    "CREATED_HOUR", "CREATED_DAY_OF_WEEK", "CREATED_MONTH", "RESPONSE_TIME_DAYS"]
    df = df_raw[cols_to_keep].copy()
    df = df.dropna(subset=['RESPONSE_TIME_DAYS'])

    # cap response time at 99th percentile to handle extreme outliers
    if cap_at_p99:
        p99 = df['RESPONSE_TIME_DAYS'].quantile(0.99)
        n_before = len(df)
        df = df[df['RESPONSE_TIME_DAYS'] <= p99].copy()
        n_removed = n_before - len(df)
        print(f"Capped at P99 ({p99:.1f} days): removed {n_removed:,} outliers ({n_removed/n_before*100:.1f}%)")

    if sample_size is not None and sample_size < len(df):
        np.random.seed(42)
        df = df.sample(n=sample_size, random_state=42).copy()

    return df


def temporal_train_val_test_split(df, test_ratio=0.2, val_ratio=0.2):
    """
    Split data chronologically into train, validation, and test sets.
    Prevents data leakage by ensuring no future data is used for training.
    """
    df_sorted = df.sort_values('CREATED_DATE').copy()

    n_total = len(df_sorted)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_test - n_val

    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
    test_df = df_sorted.iloc[n_train + n_val:].copy()

    train_start = train_df['CREATED_DATE'].min()
    train_end = train_df['CREATED_DATE'].max()
    val_start = val_df['CREATED_DATE'].min()
    val_end = val_df['CREATED_DATE'].max()
    test_start = test_df['CREATED_DATE'].min()
    test_end = test_df['CREATED_DATE'].max()

    print(f"\nTemporal Split Summary:")
    print(f"  Train: {len(train_df):,} samples ({train_start.date()} to {train_end.date()})")
    print(f"  Val:   {len(val_df):,} samples ({val_start.date()} to {val_end.date()})")
    print(f"  Test:  {len(test_df):,} samples ({test_start.date()} to {test_end.date()})")
    print(f"  Total: {n_total:,} samples\n")

    return train_df, val_df, test_df


def temporal_train_test_split(df, test_ratio=0.2):
    """Split data chronologically into train and test sets only."""
    df_sorted = df.sort_values('CREATED_DATE').copy()

    n_total = len(df_sorted)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_test

    train_df = df_sorted.iloc[:n_train].copy()
    test_df = df_sorted.iloc[n_train:].copy()

    train_start = train_df['CREATED_DATE'].min()
    train_end = train_df['CREATED_DATE'].max()
    test_start = test_df['CREATED_DATE'].min()
    test_end = test_df['CREATED_DATE'].max()

    print(f"\nTemporal Split Summary:")
    print(f"  Train: {len(train_df):,} samples ({train_start.date()} to {train_end.date()})")
    print(f"  Test:  {len(test_df):,} samples ({test_start.date()} to {test_end.date()})")
    print(f"  Total: {n_total:,} samples\n")

    return train_df, test_df


def load_and_split_data(data_path=None, sample_size=5000, use_validation=True):
    """Load data and perform temporal split in one call."""
    if data_path is None:
        data_path = Path("data/raw/311_Service_Requests_2020.csv")

    df = load_and_preprocess_data(data_path, sample_size)

    if use_validation:
        return temporal_train_val_test_split(df, test_ratio=0.2, val_ratio=0.2)
    else:
        return temporal_train_test_split(df, test_ratio=0.2)
