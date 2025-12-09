# Weather Data Augmentation Summary

## What Was Done

A weather data augmentation pipeline was implemented to enhance the Chicago 311 service request dataset with historical weather information. The implementation includes:

### 1. Data Augmentation Script (`scripts/data_augmentation.py`)
- Fetches historical weather data from Open-Meteo Historical Weather API for Chicago (2020-2025)
- Merges daily weather metrics with 311 requests based on `CREATED_DATE`
- Creates both base weather metrics and derived features

### 2. Weather Features Added (7 total)

**Base Metrics (3):**
- `temp_mean`: Daily mean temperature in Fahrenheit
- `precipitation`: Total daily precipitation in inches
- `snowfall`: Total daily snowfall in inches

**Derived Features (4):**
- `extreme_cold`: Binary flag for days with mean temp < 20°F
- `heavy_precipitation`: Binary flag for days with >0.5 inches precipitation
- `snow_day`: Binary flag for days with any measurable snowfall
- `temp_deviation`: Deviation from monthly normal temperature (Chicago historical averages)

### 3. Weather-Augmented Modeling Pipeline (`main_weather_augmented.py`)
- Modified version of `main.py` that uses the weather-augmented dataset
- Same model architectures (XGBoost, Random Forest, LightGBM) with weather features included
- Uses temporal train/test split (train: before 2024, test: 2024+)
- Sample size: 5,000 requests (same as base models)

## Why It Was Done

The hypothesis was that weather conditions significantly impact 311 service response times:
- **Extreme cold**: Slows work speed, affects equipment functionality
- **Precipitation**: Delays outdoor work and field operations
- **Snow**: Affects road access and transportation, causing delays
- **Temperature deviations**: Unusual weather disrupts normal operations

By incorporating weather features, we expected models to better capture temporal patterns and improve prediction accuracy.

## Results

### Weather-Augmented Model Performance

| Model | MAE (days) | RMSE (days) | R² |
|-------|------------|-------------|-----|
| XGB_Weather | 8.07 | 18.43 | 0.097 |
| RF_Weather | 8.32 | 18.15 | 0.124 |
| LightGBM_Weather | 8.51 | 18.53 | 0.087 |
| Baseline_1_SR_TYPE | 11.63 | 19.52 | -0.013 |

### Comparison to Base Models (without weather)

| Model | Base MAE | Weather MAE | Improvement |
|-------|----------|-------------|-------------|
| XGBoost | 9.43 | 8.07 | **-14.4%** (1.36 days better) |
| Random Forest | 9.58 | 8.32 | **-13.2%** (1.26 days better) |
| LightGBM | 10.31 | 8.51 | **-17.5%** (1.80 days better) |
| Baseline | 14.50 | 11.63 | **-19.8%** (2.87 days better) |

**Note**: The baseline also improved significantly, suggesting the weather-augmented dataset may have different characteristics or the temporal split resulted in a different test set composition.

## Why Results Are Not As Expected

While the weather-augmented models show **absolute improvements** in MAE (1.3-1.8 days better), the results are not as expected for several critical reasons:

### 1. **Baseline Also Improved Dramatically (Most Critical Issue)**
- Base baseline: 14.50 days MAE
- Weather baseline: 11.63 days MAE (19.8% improvement = 2.87 days better)
- **This is the key problem**: The baseline improved MORE than the ML models, suggesting:
  - The improvement is **not primarily due to weather features**
  - Different data composition after filtering rows with missing weather data
  - Different temporal split characteristics or test set composition
  - Potential data quality differences (filtering may have removed outliers/problematic cases)

### 2. **Relative Improvement Over Baseline Actually Decreased**
- Base models: ~35% improvement over baseline (9.43 vs 14.50)
- Weather models: ~30-31% improvement over baseline (8.07 vs 11.63)
- The **relative** improvement decreased, indicating weather features are not providing the expected signal boost
- If weather were truly helpful, we'd expect the gap between ML models and baseline to widen, not narrow

### 3. **Cannot Isolate Weather Feature Impact**
- The dramatic baseline improvement makes it impossible to determine how much of the ML model improvement is due to:
  - Weather features themselves
  - Data filtering/cleaning effects
  - Different test set characteristics
- A proper comparison would require using the **exact same train/test split** and only adding weather features

### 4. **Low R² Values Persist**
- Weather models still show very low R² (0.087-0.124)
- This suggests weather features, while helpful, are not capturing the majority of variance in response times
- Most variance remains unexplained, indicating weather is not a dominant factor

### 5. **Weather Signal May Be Weak or Context-Dependent**
- Weather conditions may have less impact on response times than hypothesized
- Service request type and location (SR_TYPE, ZIP_CODE) may dominate the signal
- Weather effects may be non-linear or only matter for certain request types (e.g., outdoor vs indoor requests)
- The simple binary flags may not capture nuanced weather effects

### 6. **Feature Engineering Limitations**
- Simple binary flags (extreme_cold, snow_day) may not capture nuanced weather effects
- Temperature deviation may not be the most predictive weather feature
- Interaction effects between weather and request type/location were not explored
- Multi-day weather patterns (e.g., consecutive snow days) were not considered

## Recommendations

1. **Investigate data composition differences** between base and weather-augmented datasets
2. **Compare on identical test sets** to isolate weather feature impact
3. **Feature importance analysis** to verify weather features are actually being used by models
4. **Explore interaction features** (e.g., weather × request type, weather × location)
5. **Consider more sophisticated weather features** (e.g., multi-day weather patterns, weather severity scores)
6. **Analyze weather impact by request type** to identify which services are most weather-sensitive

## Files Changed/Added

**Staged files:**
- `scripts/data_augmentation.py` - Weather data fetching and merging
- `main_weather_augmented.py` - Modeling pipeline with weather features
- `notebooks/data_augmentation.ipynb` - Exploratory notebook
- `results/model_results_weather.csv` - Weather model results
- `results/final_model_comparison_weather.png` - Visualization
- `README.md` - Updated with weather augmentation instructions

**Output:**
- `data/processed/311_Service_Requests_Since_2020_with_weather.csv` - Augmented dataset
- `data/raw/chicago_weather_full.csv` - Cached weather data

