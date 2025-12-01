"""
Main forecasting pipeline orchestrator
Combines data loading, feature engineering, and model prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Union

from backend.data_loader import DataLoader
from backend.feature_engineering import FeatureEngine
from backend.model import InventoryForecaster


class ForecastingPipeline:
    """
    Main orchestrator for the forecasting pipeline
    Handles end-to-end forecasting from data loading to prediction
    """

    def __init__(self):
        """Initialize the forecasting pipeline"""
        self.data_loader = DataLoader()
        self.feature_engine = FeatureEngine()
        self.forecaster = None

        self.df_prepared = None
        self.df_features = None

    def prepare_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Prepare data for forecasting (load, clean, engineer features)

        Args:
            use_cache: If True, use cached data if available

        Returns:
            pd.DataFrame: Data with all features ready for prediction
        """
        if use_cache and self.df_features is not None:
            print("Using cached feature data...")
            return self.df_features

        print("\n" + "="*80)
        print("FORECASTING PIPELINE: DATA PREPARATION")
        print("="*80)

        # Step 1: Load and prepare raw data
        self.df_prepared = self.data_loader.load_and_prepare_data()

        # Step 2: Engineer features
        self.df_features = self.feature_engine.create_all_features(self.df_prepared)

        print("\nData preparation complete!")
        print(f"Total records: {len(self.df_features):,}")
        print(f"Total SKUs: {self.df_features['SKU'].nunique()}")
        print(f"Date range: {self.df_features['date'].min()} to {self.df_features['date'].max()}")

        return self.df_features

    def load_forecaster(self):
        """
        Load the trained forecasting model
        """
        if self.forecaster is None:
            print("\n" + "="*80)
            print("FORECASTING PIPELINE: MODEL LOADING")
            print("="*80)
            self.forecaster = InventoryForecaster()

    def generate_forecast(
        self,
        prediction_date: Union[str, datetime, pd.Timestamp],
        sku: Optional[str] = None,
        horizon_days: int = 1
    ) -> dict:
        """
        Generate forecast for specified date and SKU(s)

        Args:
            prediction_date: Date to generate forecast for (can be string, datetime, or Timestamp)
            sku: Specific SKU to forecast (if None, forecasts all SKUs)
            horizon_days: Number of days to forecast (1 for single-day, >1 for multi-day)

        Returns:
            dict: Forecast results containing predictions DataFrame and metadata
        """
        # Convert prediction_date to Timestamp
        if isinstance(prediction_date, str):
            prediction_date = pd.Timestamp(prediction_date)
        elif isinstance(prediction_date, datetime):
            prediction_date = pd.Timestamp(prediction_date)

        # Ensure data is prepared
        if self.df_features is None:
            self.prepare_data()

        # Ensure forecaster is loaded
        if self.forecaster is None:
            self.load_forecaster()

        print("\n" + "="*80)
        print("FORECASTING PIPELINE: GENERATING PREDICTIONS")
        print("="*80)
        print(f"Prediction Date: {prediction_date.date()}")
        print(f"Forecast Horizon: {horizon_days} day(s)")
        print(f"SKU Filter: {sku if sku else 'All SKUs'}")

        # Generate predictions
        if horizon_days == 1:
            # Single-day forecast
            if sku:
                # Single SKU
                pred_dict = self.forecaster.predict_for_sku(self.df_features, sku, prediction_date)
                df_predictions = pd.DataFrame([pred_dict])
            else:
                # All SKUs
                df_predictions = self.forecaster.predict_for_date(self.df_features, prediction_date)
        else:
            # Multi-day forecast
            df_predictions = self.forecaster.predict_multi_date(
                self.df_features,
                prediction_date,
                horizon_days
            )

            if sku:
                # Filter to specific SKU
                df_predictions = df_predictions[df_predictions['SKU'] == sku]

        # Calculate summary statistics
        summary = self._calculate_summary(df_predictions)

        # Prepare result
        result = {
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'forecast_horizon': horizon_days,
            'sku_filter': sku,
            'predictions': df_predictions,
            'summary': summary,
            'generated_at': datetime.now().isoformat()
        }

        print("\n" + "="*80)
        print("FORECAST GENERATION COMPLETE")
        print("="*80)
        print(f"Total predictions: {len(df_predictions)}")
        print(f"SKUs with expected demand: {(df_predictions['pred_median'] > 0).sum()}")
        print(f"Total predicted demand (Q50): {df_predictions['pred_median'].sum():.0f} units")
        print(f"Total safety stock needed (Q90): {df_predictions['pred_upper'].sum():.0f} units")

        return result

    def _calculate_summary(self, df_predictions: pd.DataFrame) -> dict:
        """
        Calculate summary statistics from predictions

        Args:
            df_predictions: DataFrame with predictions

        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_skus': int(df_predictions['SKU'].nunique()),
            'skus_with_demand': int((df_predictions['pred_median'] > 0).sum()),
            'total_predicted_demand_q50': float(df_predictions['pred_median'].sum()),
            'total_safety_stock_q90': float(df_predictions['pred_upper'].sum()),
            'avg_manufacturing_time': float(df_predictions['manufacturing_time'].mean()),
            'predictions_by_tier': df_predictions.groupby('tier').agg({
                'SKU': 'nunique',
                'pred_median': 'sum',
                'pred_upper': 'sum'
            }).to_dict(),
            'predictions_by_product_type': df_predictions.groupby('product_type').agg({
                'SKU': 'nunique',
                'pred_median': 'sum',
                'pred_upper': 'sum'
            }).to_dict()
        }

        return summary

    def get_historical_data(self, sku: str, days: int = 90) -> pd.DataFrame:
        """
        Get historical sales data for a specific SKU

        Args:
            sku: SKU to get history for
            days: Number of days of history to retrieve

        Returns:
            pd.DataFrame: Historical data for the SKU
        """
        if self.df_features is None:
            self.prepare_data()

        # Filter to SKU
        df_sku = self.df_features[self.df_features['SKU'] == sku].copy()

        # Get last N days
        df_sku = df_sku.sort_values('date').tail(days)

        return df_sku[['date', 'SKU', 'quantity', 'manufacturing_time', 'product_type', 'tier']]


# Convenience function for quick forecasting
def quick_forecast(prediction_date: str, sku: Optional[str] = None, horizon_days: int = 1) -> dict:
    """
    Convenience function for quick forecasting

    Args:
        prediction_date: Date to forecast for (YYYY-MM-DD format)
        sku: Optional SKU filter
        horizon_days: Number of days to forecast

    Returns:
        dict: Forecast results
    """
    pipeline = ForecastingPipeline()
    return pipeline.generate_forecast(prediction_date, sku, horizon_days)
