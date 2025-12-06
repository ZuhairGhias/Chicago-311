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

### 4. Run the Models
```bash
python main.py
```

This will:
- Train all models (baseline, XGBoost, Random Forest, LightGBM)
- Generate comparison visualizations
- Save results to `results/` directory

## Project Structure

```
Chicago-311/
├── main.py                  # Main entry point - runs all models
├── requirements.txt         # Python dependencies
├── data/
│   └── raw/                 # Place data here (gitignored)
├── results/                 # Model outputs and visualizations
│   ├── *.csv               # Performance metrics
│   └── *.png               # Comparison plots
├── src/
│   ├── methods/            # Model implementations
│   │   ├── baseline.py     # Historical average baselines
│   │   ├── xgboost_model.py
│   │   ├── random_forest_model.py
│   │   └── lightgbm_model.py
│   ├── processes/          # Data processing utilities
│   └── config.py           # Project configuration
├── notebooks/              # Jupyter notebooks for exploration
└── tests/                  # Unit tests
```

## Models Implemented

1. **Baseline Methods** - Historical averages by service type and location
2. **XGBoost** - Gradient boosting with L1/L2 regularization
3. **Random Forest** - Ensemble decision trees
4. **LightGBM** - Fast gradient boosting framework

## Results

All models significantly outperform baseline methods:
- **XGBoost**: 9.43 days MAE (35% improvement over baseline)
- **Random Forest**: 9.58 days MAE (34% improvement)
- **LightGBM**: 10.31 days MAE (29% improvement)
- **Baseline**: 14.50 days MAE

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
