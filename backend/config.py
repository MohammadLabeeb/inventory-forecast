"""
Configuration file for Inventory Forecasting
Contains paths, feature definitions, and model parameters
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'Data'
MODEL_DIR = BASE_DIR / 'models'

# Input data files
RAW_DATA_FILE = DATA_DIR / 'raw' / 'Unterlagen Teilnehmer' / 'auftraege_mit_sku_Teilnehmer.xlsx'
PRODUCT_FILE = DATA_DIR / 'raw' / 'Unterlagen Teilnehmer' / 'Produktionszeit_Warenfluss_Hackathon_Teilnehmer.xlsx'

# Model files
MODEL_FILE = MODEL_DIR / 'lightgbm_final.pkl'
METADATA_FILE = MODEL_DIR / 'model_metadata.json'

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# All features used for model training (from notebook)
FEATURE_COLUMNS = [
    # Product and time features
    'SKU', 'month', 'quarter', 'day_of_week',
    'manufacturing_time', 'lead_time_weeks',

    # Historical statistics
    'historical_avg_sales', 'historical_std_sales', 'demand_cv',

    # Lag features
    'lag_30', 'lag_60', 'lag_90', 'same_month_last_year',

    # Rolling statistics
    'rolling_mean_30', 'rolling_mean_60', 'rolling_std_30',

    # Seasonality and calendar
    'is_high_season', 'is_weekend',

    # Price and recency
    'price_per_unit', 'days_since_last_sale',

    # EMA features
    'ema_7', 'ema_14', 'ema_30',
    'ema_trend_short', 'ema_trend_long', 'recent_volatility'
]

# Categorical features (need special encoding in LightGBM)
CATEGORICAL_FEATURES = ['SKU', 'month', 'quarter', 'day_of_week']

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Tier groups for tier-specific models
TIER_GROUPS = {
    'high_volume': [1, 2],  # Tiers 1-2: More complex models
    'low_volume': [3, 4]     # Tiers 3-4: More regularized models
}

# Tier definitions based on historical sales volume
TIER_THRESHOLDS = {
    1: 100,  # >= 100 sales records
    2: 50,   # 50-99 sales records
    3: 20,   # 20-49 sales records
    4: 10,   # 10-19 sales records
    5: 0     # < 10 sales records (baseline model)
}

# High season months (identified from data analysis)
HIGH_SEASON_MONTHS = [3, 4, 5]  # March, April, May

# ============================================================================
# BUSINESS PARAMETERS
# ============================================================================

# Safety buffer days (added to manufacturing time)
SAFETY_BUFFER_DAYS = 3

# Default forecast horizon (days)
DEFAULT_FORECAST_HORIZON = 30

# Probability threshold for binary classifier
DEMAND_PROBABILITY_THRESHOLD = 0.3

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================

# Date range for historical data
MIN_DATE = '2024-01-26'
MAX_DATE = '2025-11-24'

# Column mappings from raw data
RAW_DATA_COLUMNS = {
    'date': 'Auftragsdatum',
    'quantity': 'Auftragspositionen/Menge',
    'total_price': 'Gesamtpreis €',
    'sku': 'SKU'
}

# Manufacturing time columns to sum
MANUFACTURING_TIME_COLUMNS = [
    'Produzent 1', 'Produzent 2', 'Produzent 3', 'Produzent 4',
    'Duplexbeschichter', 'Verzinken (=Produzent 2)',
    'ELEO Lager', 'Annahme Lagerbestand Status Quo'
]

# Product type sheet names
PRODUCT_SHEETS = {
    'product_1': 'Produktsparte 1_Produktionszeit',
    'product_2': 'Produktsparte 2_Produktionzeit',
    'product_3': 'Produktsparte 3_Produktionszeit'
}
