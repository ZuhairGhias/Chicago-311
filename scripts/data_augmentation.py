#!/usr/bin/env python3
"""
Weather Data Augmentation Script for Chicago 311 Dataset

This script augments the Chicago 311 service request dataset with weather data
from the Open-Meteo Historical Weather API. It adds both base weather metrics
and derived weather features to improve prediction accuracy.

Usage:
    python scripts/data_augmentation.py

Features Added:
    - Base weather metrics (3): temp_mean, precipitation, snowfall
    - Derived features (4): extreme_cold, heavy_precipitation, snow_day, temp_deviation

Output:
    - Saves augmented dataset to data/processed/311_Service_Requests_Since_2020_with_weather.csv
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime

# Display settings
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)


def fetch_chicago_weather(start_date, end_date, cache_path):
    """
    Fetch historical weather data for Chicago from Open-Meteo API.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        cache_path: Path to cache the weather data

    Returns:
        pd.DataFrame: Daily weather data with columns: date, temp_mean, precipitation, snowfall
    """
    # Check if cache exists
    if Path(cache_path).exists():
        print(f"Loading cached weather data from {cache_path}...")
        weather_df = pd.read_csv(cache_path, parse_dates=['date'])
        print(f"✓ Loaded {len(weather_df)} days from cache")
        return weather_df

    print(f"Fetching weather data from Open-Meteo API ({start_date} to {end_date})...")

    url = "https://archive-api.open-meteo.com/v1/archive"

    # Only fetch features that matter for prediction:
    # - Temperature (work speed, equipment function)
    # - Precipitation (delays outdoor work)
    # - Snow (road access, delays)
    params = {
        "latitude": 41.8781,  # Chicago Loop
        "longitude": -87.6298,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_mean",      # Daily mean temp
            "precipitation_sum",         # Total precipitation
            "snowfall_sum",             # Total snowfall
        ],
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/Chicago"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"Weather API error: {response.status_code}\n{response.text}")

    data = response.json()

    # Convert to DataFrame - only keep essential features
    weather_df = pd.DataFrame({
        'date': pd.to_datetime(data['daily']['time']),
        'temp_mean': data['daily']['temperature_2m_mean'],
        'precipitation': data['daily']['precipitation_sum'],
        'snowfall': data['daily']['snowfall_sum'],
    })

    # Cache the results
    weather_df.to_csv(cache_path, index=False)
    print(f"✓ Fetched {len(weather_df)} days of weather data")
    print(f"✓ Cached to {cache_path}")

    return weather_df


def create_derived_weather_features(df):
    """
    Create derived weather features from base weather metrics.

    Args:
        df: DataFrame with base weather columns (temp_mean, precipitation, snowfall, CREATED_MONTH)

    Returns:
        DataFrame with additional derived features added in-place
    """
    print("Creating derived weather features...\n")

    # 1. Extreme cold (below 20�F mean) - impacts work speed and equipment
    df['extreme_cold'] = (df['temp_mean'] < 20).fillna(0).astype(int)
    extreme_cold_count = (df['temp_mean'] < 20).sum()
    print(f"✓ extreme_cold: {extreme_cold_count:,} requests on extreme cold days")

    # 2. Heavy precipitation (>0.5 inches) - delays outdoor work
    df['heavy_precipitation'] = (df['precipitation'] > 0.5).fillna(0).astype(int)
    heavy_precip_count = (df['precipitation'] > 0.5).sum()
    print(f"✓ heavy_precipitation: {heavy_precip_count:,} requests on heavy rain days")

    # 3. Snow day (any measurable snowfall) - affects road access and work
    df['snow_day'] = (df['snowfall'] > 0).fillna(0).astype(int)
    snow_day_count = (df['snowfall'] > 0).sum()
    print(f"✓ snow_day: {snow_day_count:,} requests on snow days")

    # 4. Temperature deviation from monthly normal - unusual weather disrupts operations
    # Chicago average temps by month (historical normals)
    monthly_normal_temps = {
        1: 27, 2: 30, 3: 41, 4: 52, 5: 63, 6: 72,
        7: 77, 8: 75, 9: 68, 10: 56, 11: 43, 12: 31
    }
    df['temp_normal'] = df['CREATED_MONTH'].map(monthly_normal_temps)
    df['temp_deviation'] = (df['temp_mean'] - df['temp_normal']).fillna(0)

    print(f"✓ temp_deviation calculated")
    print(f"  Range: {df['temp_deviation'].min():.1f}�F to {df['temp_deviation'].max():.1f}�F")

    print("\n✓ Created 4 derived weather features")

    return df


def main():
    """Main execution function."""
    print("=" * 60)
    print("CHICAGO 311 WEATHER DATA AUGMENTATION")
    print("=" * 60)
    print()

    # ========================================================================
    # 1. LOAD AND PREPARE BASE DATASET
    # ========================================================================
    print("STEP 1: Loading base dataset")
    print("-" * 60)

    data_path = Path('data/raw/311_Service_Requests_Since_2020.csv')

    print(f"Loading data from: {data_path}")
    print(f"File exists: {data_path.exists()}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Read CSV
    df = pd.read_csv(data_path)

    print(f"\n✓ Loaded {len(df):,} rows")
    print(f"✓ Columns: {df.shape[1]}")

    # Convert date columns to datetime
    print("\nConverting date columns...")
    df['CREATED_DATE'] = pd.to_datetime(df['CREATED_DATE'], errors='coerce')
    df['CLOSED_DATE'] = pd.to_datetime(df['CLOSED_DATE'], errors='coerce')

    print(f"✓ Converted dates")
    print(f"CREATED_DATE range: {df['CREATED_DATE'].min()} to {df['CREATED_DATE'].max()}")
    print(f"CLOSED_DATE range: {df['CLOSED_DATE'].min()} to {df['CLOSED_DATE'].max()}")

    # Extract base temporal features
    df['CREATED_HOUR'] = df['CREATED_DATE'].dt.hour
    df['CREATED_DAY_OF_WEEK'] = df['CREATED_DATE'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['CREATED_MONTH'] = df['CREATED_DATE'].dt.month
    df['CREATED_DATE_ONLY'] = df['CREATED_DATE'].dt.date

    print("\n✓ Extracted base temporal features")

    # ========================================================================
    # 2. FETCH WEATHER DATA
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Fetching weather data")
    print("-" * 60)

    # Determine date range from our data
    min_date = df['CREATED_DATE'].min().date()
    max_date = df['CREATED_DATE'].max().date()

    print(f"Data date range: {min_date} to {max_date}")
    print(f"Total days needed: {(max_date - min_date).days + 1}")

    # Fetch weather for full date range
    cache_path = Path('data/raw/chicago_weather_full.csv')
    weather_df = fetch_chicago_weather(
        start_date=str(min_date),
        end_date=str(max_date),
        cache_path=cache_path
    )

    print("\nWeather data statistics:")
    print(weather_df.describe())

    # ========================================================================
    # 3. MERGE WEATHER DATA
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Merging weather data")
    print("-" * 60)

    # Prepare weather data for merge (date only, no time)
    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date

    print(f"Main dataset rows before merge: {len(df):,}")
    print(f"Weather data dates: {len(weather_df):,}")

    df = df.merge(weather_df, left_on='CREATED_DATE_ONLY', right_on='date', how='left')

    print(f"Main dataset rows after merge: {len(df):,}")

    # Check for missing weather data
    missing_weather = df['temp_mean'].isna().sum()
    print(f"Rows with missing weather: {missing_weather:,} ({missing_weather/len(df)*100:.2f}%)")

    if missing_weather > 0:
        # Show which dates are missing
        missing_dates = df[df['temp_mean'].isna()]['CREATED_DATE_ONLY'].unique()
        print(f"Missing dates sample: {sorted(missing_dates)[:5]}")

    print("\n✓ Weather data merged successfully")

    # ========================================================================
    # 4. CREATE DERIVED FEATURES
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Creating derived weather features")
    print("-" * 60)

    df = create_derived_weather_features(df)

    # ========================================================================
    # 5. SAVE AUGMENTED DATASET
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Saving augmented dataset")
    print("-" * 60)

    # Create output directory if it doesn't exist
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / '311_Service_Requests_Since_2020_with_weather.csv'
    df.to_csv(output_path, index=False)

    print(f"✓ Saved weather-augmented dataset to {output_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {df.shape[1]}")

    # ========================================================================
    # 6. SUMMARY
    # ========================================================================
    print("\n" + "=" * 60)
    print("WEATHER DATA AUGMENTATION COMPLETE")
    print("=" * 60)

    weather_features = [
        'temp_mean',
        'precipitation',
        'snowfall',
        'extreme_cold',
        'heavy_precipitation',
        'snow_day',
        'temp_deviation'
    ]

    print(f"\nTotal weather features added: {len(weather_features)}")
    print("\nFeature list:")
    for i, feat in enumerate(weather_features, 1):
        print(f"  {i}. {feat}")

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Missing weather data: {df['temp_mean'].isna().sum():,} ({df['temp_mean'].isna().sum()/len(df)*100:.2f}%)")

    # Print weather feature statistics
    print("\nWeather Feature Statistics:")
    print("-" * 60)
    for col in weather_features:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"\n{col}:")
            if df[col].dtype in ['int64', 'float64']:
                print(f"  Count: {non_null:,}")
                print(f"  Mean: {df[col].mean():.2f}")
                print(f"  Min: {df[col].min():.2f}")
                print(f"  Max: {df[col].max():.2f}")

    print("\n" + "=" * 60)
    print("✓ ALL STEPS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
