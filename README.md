# Inventory Demand Forecasting Dashboard

AI-powered inventory demand forecasting system using LightGBM with a hurdle model architecture.

## Overview

This project provides end-to-end inventory demand forecasting with:
- **Modular Python backend** for data processing, feature engineering, and predictions
- **Interactive Streamlit dashboard** for stakeholder-friendly forecast generation
- **Multi-day rolling forecasts** with safety stock recommendations
- **Tier-specific models** optimized for different product categories

## Demo

![Dashboard Demo](gif/inventory_forecast.gif)

*Interactive dashboard demonstration showing forecast generation and visualization*

## Features

### Backend (Production-Ready)
-  Modular architecture with clean separation of concerns
-  Data loading and preprocessing pipeline
-  Comprehensive feature engineering (20+ features)
-  Model interface supporting tier-specific predictions

### Frontend (Streamlit Dashboard)
-  Interactive forecast configuration
-  Multi-day forecast horizon
-  Summary metrics and detailed prediction tables
-  Interactive visualizations with Plotly
-  Model performance metrics display
-  SKU-level deep dive analysis
-  CSV/PDF export functionality

## Project Structure

```
inventory-forecast/
   backend/                     # Production backend modules
      __init__.py
      config.py               # Configuration and constants
      data_loader.py          # Data loading and preprocessing
      feature_engineering.py  # Feature creation pipeline
      model.py                # Model loading and prediction
      forecaster.py           # Main forecasting orchestrator
      utils.py                # Utility functions

   app.py                      # Streamlit dashboard (main entry point)
   requirements.txt            # Python dependencies
   README.md                   # This file

   Data/                       # Data files (not included - confidential)
   models/                     # Trained models
      lightgbm_final.pkl
      model_metadata.json

   notebooks/                  # Jupyter notebooks (development)
       Inventory_Forecasting_Final.ipynb

   Generated Report/           # Exported PDF reports
```

## Installation

### Prerequisites
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Clone or navigate to the project directory:
```bash
cd inventory-forecast
```

2. Install dependencies using uv:
```bash
uv sync
```

This will automatically create a virtual environment and install all required dependencies from `uv.lock`.

## Usage

### Running the Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your web browser at `http://localhost:8501`.

### Dashboard Workflow

1. **Generate Forecast**:
   - Click "Generate Forecast" button
   - Wait for processing (first run may take ~10-20 seconds to load data and forecast for all ~200 SKUs)

2. **Explore Results** (4 Tabs):
   - **Tab 1 - Forecast Results**: Summary metrics, detailed table, CSV/PDF export
   - **Tab 2 - Visualizations**: Charts and graphs
   - **Tab 3 - Model Performance**: ML model metrics
   - **Tab 4 - SKU Deep Dive**: Individual SKU analysis

## Model Architecture

### Hurdle Model (Two-Stage)
1. **Binary Classifier**: Predicts whether there will be any demand (yes/no)
2. **Quantile Regressors**:
   - Q50 (Median): Expected demand level for order planning
   - Q90 (Upper Bound): Conservative estimate for safety stock

### Tier-Specific Training
- **High Volume** (Tiers 1-2): More complex models (64 leaves, depth 8)
- **Low Volume** (Tiers 3-4): More regularized models (31 leaves, depth 6)
- **Tier 5**: Simple baseline (moving average)

### Product Tiers
- **Tier 1**: >99 sales records (highest volume)
- **Tier 2**: 50-99 sales records
- **Tier 3**: 20-49 sales records
- **Tier 4**: 10-19 sales records
- **Tier 5**: <10 sales records (baseline only)

## Features

The model uses 26 features across multiple categories:

- **Temporal**: Month, quarter, day of week, seasonality
- **Historical**: Expanding mean, std, coefficient of variation
- **Lag Features**: 30, 60, 90 days + year-over-year
- **Rolling Statistics**: 30, 60 day means and stds
- **EMA Features**: 7, 14, 30 day exponential moving averages
- **Trends**: Short-term and long-term momentum
- **Recency**: Days since last sale
- **Product**: Manufacturing time, lead time, product type

## Model Performance

Test set metrics (June-Nov 2025):

- **WAPE**: 27.3%
- **R2**: 0.86
- **MAE**: 1.26 units
- **Calibration (Q90)**: 86.0% (safety stock covers actual demand 86% of the time)

Performance by tier:
| Tier | WAPE | MAE | Calibration |
|------|------|-----|-------------|
| 1    | 23.6% | 3.49 | 81.5% |
| 2    | 22.3% | 1.41 | 86.9% |
| 3    | 33.1% | 1.03 | 87.6% |
| 4    | 30.5% | 0.66 | 84.7% |

## Data Requirements

### Input Files
1. **Sales Data**: `auftraege_mit_sku_Teilnehmer.xlsx`

2. **Manufacturing Data**: `Produktionszeit_Warenfluss_Hackathon_Teilnehmer.xlsx`

**Note**: Sample data files are not included in this repository due to confidentiality.

### Trained Models
- `lightgbm_final.pkl`: Trained LightGBM models
- `model_metadata.json`: Model configuration and metadata

## Configuration

Edit `backend/config.py` to customize:
- File paths
- Feature columns
- Tier thresholds
- High season months
- Safety buffer days
- Forecast parameters
