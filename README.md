# Chicago 311 Response Time Prediction

Machine learning models to predict Chicago 311 service request response times using gradient boosting and ensemble methods.

## Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd Chicago-311
```

### 2. Install Dependencies

**Option A: Using pip (recommended)**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Option B: Using the project**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Download Data
Download the Chicago 311 data and place it at:
```
data/raw/311_Service_Requests_Since_2020.csv
```

Data source: [Chicago Data Portal](https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy)

### 4. Run Analysis Scripts

**Run EDA (Exploratory Data Analysis):**
```bash
python scripts/eda.py
```
Generates correlation heatmaps, temporal trends, spatial distribution, and target distribution plots.

**Run main models (base features only):**
```bash
python main.py
```
Trains all models (baseline, XGBoost, Random Forest, LightGBM) and generates comparison visualizations.

**Run weather data augmentation:**
```bash
python scripts/data_augmentation.py
```
Fetches historical weather data and adds 7 weather features (temp, precipitation, snowfall, etc.) to the dataset.

**Run models with weather features:**
```bash
python main_weather_augmented.py
```
Trains models with weather-augmented dataset to compare performance impact.

**Run variance analysis and SHAP interpretability:**
```bash
python scripts/variance_analysis.py
```
Runs each model 10 times with different seeds, performs t-tests, and generates SHAP analysis plots.

## Project Structure

```
Chicago-311/
├── main.py                      # Main entry point - runs all models
├── main_weather_augmented.py    # Runs models with weather features
├── requirements.txt             # Python dependencies
├── scripts/                     # Analysis scripts
│   ├── eda.py                  # Exploratory data analysis
│   ├── data_augmentation.py    # Weather data augmentation
│   └── variance_analysis.py    # Variance testing and SHAP analysis
├── data/
│   ├── raw/                     # Place data here (gitignored)
│   └── processed/               # Weather-augmented datasets
├── results/                     # Model outputs and visualizations
│   ├── eda/                    # EDA visualizations
│   ├── interpretability/       # SHAP plots and variance analysis
│   ├── *.csv                   # Performance metrics
│   └── *.png                   # Comparison plots
├── src/
│   ├── methods/                # Model implementations
│   │   ├── baseline.py         # Historical average baselines
│   │   ├── xgboost_model.py
│   │   ├── random_forest_model.py
│   │   └── lightgbm_model.py
│   ├── visualizations/         # Visualization modules
│   │   ├── eda.py              # EDA plotting functions
│   │   └── interpretability.py # SHAP and feature importance
│   ├── processes/              # Data processing utilities
│   └── config.py               # Project configuration
├── notebooks/                  # Jupyter notebooks for exploration
└── tests/                      # Unit tests
```

## Models Implemented

1. **Baseline Methods** - Historical averages by service type and location
2. **XGBoost** - Gradient boosting with L1/L2 regularization
3. **Random Forest** - Ensemble decision trees
4. **LightGBM** - Fast gradient boosting framework

## Results

All models significantly outperform baseline methods using temporal train-test split with response times capped at 99th percentile (147.8 days) to remove outliers:
- **XGBoost**: 9.36 days MAE, 20.22 RMSE (25% improvement over baseline)
- **Random Forest**: 9.37 days MAE, 19.93 RMSE (25% improvement)
- **LightGBM**: 9.45 days MAE, 20.21 RMSE (24% improvement)
- **Best Baseline**: 12.48 days MAE (by service type)

Models use proper temporal validation (train on past data, predict future requests) to ensure realistic performance estimates.

See `results/` directory for detailed metrics and visualizations.

## Development

### Running Tests
```bash
pytest tests/
```

### Adding New Models
1. Create model file in `src/methods/`
2. Implement training and prediction functions
3. Add to `main.py` for evaluation

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib (for visualizations)
- xgboost, lightgbm (gradient boosting libraries)

See `requirements.txt` for specific versions.
