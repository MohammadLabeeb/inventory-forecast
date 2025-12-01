"""
Model interface for loading trained models and generating predictions
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from backend.config import (
    MODEL_FILE, METADATA_FILE,
    FEATURE_COLUMNS, CATEGORICAL_FEATURES,
    TIER_GROUPS, DEMAND_PROBABILITY_THRESHOLD
)


class InventoryForecaster:
    """
    Interface for loading trained models and generating predictions
    """

    def __init__(self, model_path=None, metadata_path=None):
        """
        Initialize the forecaster

        Args:
            model_path: Path to saved model file (default: from config)
            metadata_path: Path to model metadata (default: from config)
        """
        self.model_path = model_path or MODEL_FILE
        self.metadata_path = metadata_path or METADATA_FILE

        self.models = None
        self.metadata = None

        # Load models automatically
        self.load_models()

    def load_models(self):
        """
        Load trained LightGBM models from pickle file
        """
        print(f"Loading models from {self.model_path}...")

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load models
        self.models = joblib.load(self.model_path)

        # Load metadata if available
        if Path(self.metadata_path).exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata from {self.metadata_path}")

        print("Models loaded successfully!")
        print(f"  - High volume models: {list(self.models['high_volume'].keys())}")
        print(f"  - Low volume models: {list(self.models['low_volume'].keys())}")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction

        Args:
            df: DataFrame with all calculated features

        Returns:
            pd.DataFrame: Features ready for model input
        """
        # Select only the required feature columns
        X = df[FEATURE_COLUMNS].copy()

        # Convert categorical features to 'category' dtype for LightGBM
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                X[col] = X[col].astype('category')

        return X

    def predict_for_date(self, df: pd.DataFrame, prediction_date: pd.Timestamp) -> pd.DataFrame:
        """
        Generate predictions for all SKUs on a specific date

        Args:
            df: DataFrame with all features
            prediction_date: Date to generate predictions for

        Returns:
            pd.DataFrame: Predictions for all SKUs on the given date
        """
        # Filter to the prediction date
        df_date = df[df['date'] == prediction_date].copy()

        if len(df_date) == 0:
            raise ValueError(f"No data available for date: {prediction_date}")

        # Initialize prediction columns
        df_date['prob_nonzero'] = 0.0
        df_date['pred_median'] = 0.0
        df_date['pred_upper'] = 0.0

        # Generate predictions for each tier group
        for group_name, tiers in TIER_GROUPS.items():
            # Filter data for this tier group
            mask = df_date['tier'].isin(tiers)
            df_group = df_date[mask]

            if len(df_group) == 0:
                continue

            # Prepare features
            X = self._prepare_features(df_group)

            # Get models for this group
            classifier = self.models[group_name]['classifier']
            regressor_q50 = self.models[group_name]['regressor_q50']
            regressor_q90 = self.models[group_name]['regressor_q90']

            # Step 1: Predict probability of non-zero demand
            prob_nonzero = classifier.predict(X)

            # Step 2: Predict demand levels
            pred_q50_raw = regressor_q50.predict(X)
            pred_q90_raw = regressor_q90.predict(X)

            # Step 3: Combine predictions (hurdle model)
            # If probability < threshold, predict zero; otherwise use quantile predictions
            pred_median = np.where(
                prob_nonzero < DEMAND_PROBABILITY_THRESHOLD,
                0,
                np.maximum(0, pred_q50_raw)
            )
            pred_upper = np.where(
                prob_nonzero < DEMAND_PROBABILITY_THRESHOLD,
                0,
                np.maximum(0, pred_q90_raw)
            )

            # Assign predictions back to dataframe
            df_date.loc[mask, 'prob_nonzero'] = prob_nonzero
            df_date.loc[mask, 'pred_median'] = pred_median
            df_date.loc[mask, 'pred_upper'] = pred_upper

        # Handle Tier 5 with baseline predictions
        mask_tier5 = df_date['tier'] == 5
        if mask_tier5.sum() > 0:
            # Simple baseline: use historical average
            df_date.loc[mask_tier5, 'pred_median'] = df_date.loc[mask_tier5, 'historical_avg_sales']
            df_date.loc[mask_tier5, 'pred_upper'] = df_date.loc[mask_tier5, 'historical_avg_sales'] * 1.5
            df_date.loc[mask_tier5, 'prob_nonzero'] = (df_date.loc[mask_tier5, 'historical_avg_sales'] > 0).astype(float)

        return df_date

    def predict_for_sku(self, df: pd.DataFrame, sku: str, prediction_date: pd.Timestamp) -> dict:
        """
        Generate prediction for a single SKU on a specific date

        Args:
            df: DataFrame with all features
            sku: SKU to predict for
            prediction_date: Date to generate prediction for

        Returns:
            dict: Prediction results for the SKU
        """
        # Get predictions for all SKUs on this date
        df_predictions = self.predict_for_date(df, prediction_date)

        # Filter to the requested SKU
        sku_data = df_predictions[df_predictions['SKU'] == sku]

        if len(sku_data) == 0:
            raise ValueError(f"SKU '{sku}' not found in data")

        # Extract results
        row = sku_data.iloc[0]

        result = {
            'sku': sku,
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'tier': int(row['tier']),
            'product_type': row['product_type'],
            'manufacturing_time': int(row['manufacturing_time']),
            'probability_demand': float(row['prob_nonzero']),
            'predicted_median': float(row['pred_median']),
            'predicted_upper': float(row['pred_upper']),
            'historical_avg': float(row['historical_avg_sales']),
            'recent_trend': float(row['ema_trend_short'])
        }

        return result

    def predict_multi_date(self, df: pd.DataFrame, start_date: pd.Timestamp,
                          horizon_days: int = 30) -> pd.DataFrame:
        """
        Generate predictions for multiple dates (rolling forecast)

        Args:
            df: DataFrame with all features
            start_date: Starting date for forecast
            horizon_days: Number of days to forecast

        Returns:
            pd.DataFrame: Predictions for all dates in the horizon
        """
        print(f"Generating {horizon_days}-day forecast starting from {start_date.date()}...")

        all_predictions = []

        for i in range(horizon_days):
            current_date = start_date + pd.Timedelta(days=i)

            try:
                df_pred = self.predict_for_date(df, current_date)
                all_predictions.append(df_pred)
            except ValueError as e:
                print(f"Warning: {e}")
                continue

        if len(all_predictions) == 0:
            raise ValueError(f"No valid predictions could be generated for the date range")

        # Combine all predictions
        df_all = pd.concat(all_predictions, ignore_index=True)

        print(f"Generated {len(all_predictions)} days of predictions for {df_all['SKU'].nunique()} SKUs")

        return df_all


# Convenience function
def load_forecaster() -> InventoryForecaster:
    """
    Convenience function to load the forecaster

    Returns:
        InventoryForecaster: Loaded forecaster instance
    """
    return InventoryForecaster()
