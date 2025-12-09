"""
Exploratory Data Analysis Script
Run this before modeling to generate EDA visualizations
"""

import sys
from pathlib import Path

# add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.visualizations.eda import (
    plot_correlation_heatmap,
    plot_temporal_trends,
    plot_spatial_distribution,
    plot_target_distribution
)


def main():
    print("="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)

    # load data
    print("\nLoading data...")
    data_path = Path(__file__).parent.parent / "data/raw/311_Service_Requests_2020.csv"
    df_raw = pd.read_csv(data_path)

    # calculate response time
    df_raw['CREATED_DATE'] = pd.to_datetime(df_raw['CREATED_DATE'])
    df_raw['CLOSED_DATE'] = pd.to_datetime(df_raw['CLOSED_DATE'])
    df_raw['RESPONSE_TIME'] = df_raw['CLOSED_DATE'] - df_raw['CREATED_DATE']
    df_raw['RESPONSE_TIME_DAYS'] = df_raw['RESPONSE_TIME'].dt.total_seconds() / 86400

    # select relevant columns
    cols = ["SR_TYPE", "ORIGIN", "CREATED_DATE", "CLOSED_DATE", "ZIP_CODE",
            "CREATED_HOUR", "CREATED_DAY_OF_WEEK", "CREATED_MONTH",
            "LATITUDE", "LONGITUDE", "RESPONSE_TIME_DAYS"]
    df = df_raw[cols].copy()
    df = df.dropna(subset=['RESPONSE_TIME_DAYS'])

    # sample for faster processing (optional - remove for full dataset)
    np.random.seed(42)
    df_sample = df.sample(n=min(50000, len(df)), random_state=42).copy()

    print(f"Dataset size: {len(df_sample)} samples")

    # 1. target distribution
    print("\n1. Analyzing target variable distribution...")
    plot_target_distribution(df_sample)

    # 2. temporal trends
    print("\n2. Analyzing temporal trends...")
    plot_temporal_trends(df_sample)

    # 3. spatial distribution
    print("\n3. Analyzing spatial distribution...")
    plot_spatial_distribution(df_sample)

    # 4. correlation heatmap (encode categoricals first)
    print("\n4. Creating correlation heatmap...")
    df_encoded = df_sample.copy()

    # label encode categorical columns
    for col in ['SR_TYPE', 'ORIGIN', 'ZIP_CODE']:
        le = LabelEncoder()
        df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))

    # select features for correlation
    features = ['CREATED_HOUR', 'CREATED_DAY_OF_WEEK', 'CREATED_MONTH',
                'SR_TYPE_encoded', 'ORIGIN_encoded', 'ZIP_CODE_encoded',
                'LATITUDE', 'LONGITUDE', 'RESPONSE_TIME_DAYS']

    correlation_matrix = plot_correlation_heatmap(df_encoded, features)

    # print top correlations with target
    print("\nTop correlations with RESPONSE_TIME_DAYS:")
    target_corr = correlation_matrix['RESPONSE_TIME_DAYS'].drop('RESPONSE_TIME_DAYS').sort_values(ascending=False)
    print(target_corr)

    print("\n" + "="*70)
    print("EDA COMPLETE")
    print("="*70)
    print("\nVisualizations saved in results/eda/")
    print("  - target_distribution.png")
    print("  - temporal_trends.png")
    print("  - spatial_distribution.png")
    print("  - correlation_heatmap.png")


if __name__ == "__main__":
    main()
