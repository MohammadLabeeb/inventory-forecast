"""
Feature engineering module
Creates all features needed for inventory demand forecasting
"""

import pandas as pd
import numpy as np
from backend.config import HIGH_SEASON_MONTHS


class FeatureEngine:
    """
    Handles all feature creation for the forecasting model
    """

    def __init__(self):
        """Initialize the FeatureEngine"""
        pass

    @staticmethod
    def calculate_target_variable(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate target variable for each observation
        Target = sum of sales during manufacturing lead time window

        Args:
            df: DataFrame with complete time series

        Returns:
            pd.DataFrame: Data with target variables added
        """
        print("Calculating target variables...")

        def _calculate_for_group(group):
            group = group.sort_values('date').reset_index(drop=True)
            manufacturing_time = int(group['manufacturing_time'].iloc[0])

            # Calculate cumulative demand over lead time window
            group['target_lead_time_demand'] = group['quantity'].shift(-1).rolling(
                window=manufacturing_time,
                min_periods=manufacturing_time
            ).sum()

            # Binary target for hurdle model
            group['target_binary'] = (group['target_lead_time_demand'] > 0).astype(int)

            return group

        df = df.groupby('SKU', group_keys=False).apply(_calculate_for_group)

        print(f"Valid observations: {df['target_lead_time_demand'].notna().sum():,}")

        return df

    @staticmethod
    def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from date column

        Args:
            df: DataFrame with date column

        Returns:
            pd.DataFrame: Data with temporal features added
        """
        print("Creating temporal features...")

        # Basic temporal features
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Lead time features
        df['lead_time_weeks'] = df['manufacturing_time'] / 7

        # High season indicator
        df['is_high_season'] = df['month'].isin(HIGH_SEASON_MONTHS).astype(int)

        return df

    @staticmethod
    def create_historical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create expanding historical statistics

        Args:
            df: DataFrame sorted by date

        Returns:
            pd.DataFrame: Data with historical features added
        """
        print("Calculating historical statistics...")

        def _calculate_for_group(group):
            group = group.sort_values('date').reset_index(drop=True)

            # Expanding mean and std
            group['historical_avg_sales'] = group['quantity'].expanding().mean().shift(1)
            group['historical_std_sales'] = group['quantity'].expanding().std().shift(1)

            return group

        df = df.groupby('SKU', group_keys=False).apply(_calculate_for_group)

        # Coefficient of variation
        df['demand_cv'] = df['historical_std_sales'] / (df['historical_avg_sales'] + 0.01)

        return df

    @staticmethod
    def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features

        Args:
            df: DataFrame sorted by date

        Returns:
            pd.DataFrame: Data with lag features added
        """
        print("Creating lag features...")

        def _calculate_for_group(group):
            group = group.sort_values('date').reset_index(drop=True)

            # Lag features
            group['lag_30'] = group['quantity'].shift(30)
            group['lag_60'] = group['quantity'].shift(60)
            group['lag_90'] = group['quantity'].shift(90)

            # Year-over-year seasonality
            group['same_month_last_year'] = group['quantity'].shift(364)

            return group

        df = df.groupby('SKU', group_keys=False).apply(_calculate_for_group)

        return df

    @staticmethod
    def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window statistics

        Args:
            df: DataFrame sorted by date

        Returns:
            pd.DataFrame: Data with rolling features added
        """
        print("Creating rolling features...")

        def _calculate_for_group(group):
            group = group.sort_values('date').reset_index(drop=True)

            # Rolling statistics
            group['rolling_mean_30'] = group['quantity'].rolling(
                window=30, min_periods=1
            ).mean().shift(30)

            group['rolling_mean_60'] = group['quantity'].rolling(
                window=60, min_periods=1
            ).mean().shift(60)

            group['rolling_std_30'] = group['quantity'].rolling(
                window=30, min_periods=1
            ).std().shift(30)

            return group

        df = df.groupby('SKU', group_keys=False).apply(_calculate_for_group)

        return df

    @staticmethod
    def create_ema_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Exponential Moving Average (EMA) features

        Args:
            df: DataFrame sorted by date

        Returns:
            pd.DataFrame: Data with EMA features added
        """
        print("Creating EMA and trend features...")

        def _calculate_for_group(group):
            group = group.sort_values('date').reset_index(drop=True)

            # EMAs with shift to prevent data leakage
            group['ema_7'] = group['quantity'].shift(1).ewm(span=7, adjust=False).mean()
            group['ema_14'] = group['quantity'].shift(1).ewm(span=14, adjust=False).mean()
            group['ema_30'] = group['quantity'].shift(1).ewm(span=30, adjust=False).mean()

            # Trend features
            group['ema_trend_short'] = group['ema_7'] - group['ema_14']
            group['ema_trend_long'] = group['ema_14'] - group['ema_30']

            # Volatility
            group['recent_volatility'] = group['quantity'].shift(1).rolling(14).std()

            return group

        df = df.groupby('SKU', group_keys=False).apply(_calculate_for_group)

        return df

    @staticmethod
    def create_recency_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate days since last sale

        Args:
            df: DataFrame sorted by date

        Returns:
            pd.DataFrame: Data with recency features added
        """
        print("Calculating recency features...")

        def _calculate_for_group(group):
            group = group.sort_values('date').reset_index(drop=True)

            # Find last sale date for each row
            last_sale_idx = (group['quantity'] > 0).cumsum()
            last_sale_date = group.groupby(last_sale_idx)['date'].transform('first')
            group['days_since_last_sale'] = (group['date'] - last_sale_date).dt.days

            return group

        df = df.groupby('SKU', group_keys=False).apply(_calculate_for_group)

        return df

    @staticmethod
    def assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign product tiers based on historical sales volume

        Args:
            df: DataFrame with sales data

        Returns:
            pd.DataFrame: Data with tier column added
        """
        print("Assigning product tiers...")

        # Count sales records per SKU
        sku_sales_count = df[df['quantity'] > 0].groupby('SKU').size().reset_index(name='sales_records')

        def assign_tier(sales_count):
            if sales_count >= 100:
                return 1
            elif sales_count >= 50:
                return 2
            elif sales_count >= 20:
                return 3
            elif sales_count >= 10:
                return 4
            else:
                return 5

        sku_sales_count['tier'] = sku_sales_count['sales_records'].apply(assign_tier)

        # Merge tiers back to main dataset
        df = df.merge(sku_sales_count[['SKU', 'tier']], on='SKU', how='left')
        df['tier'] = df['tier'].fillna(5).astype(int)

        # Print tier distribution
        tier_summary = df.groupby('tier')['SKU'].nunique()
        print("\nTier Distribution:")
        for tier, count in tier_summary.items():
            print(f"  Tier {tier}: {count} SKUs")

        return df

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features in the correct order

        Args:
            df: Complete time series DataFrame

        Returns:
            pd.DataFrame: Data with all features added
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*80)

        # Calculate target variable first
        df = self.calculate_target_variable(df)

        # Create temporal features
        df = self.create_temporal_features(df)

        # Create historical statistics
        df = self.create_historical_features(df)

        # Create lag features
        df = self.create_lag_features(df)

        # Create rolling features
        df = self.create_rolling_features(df)

        # Create EMA features
        df = self.create_ema_features(df)

        # Create recency features
        df = self.create_recency_features(df)

        # Assign tiers
        df = self.assign_tiers(df)

        # Fill any remaining NaN values with 0
        df = df.fillna(0)

        print(f"\nFeature engineering complete!")
        print(f"Total features created: {len(df.columns)}")
        print("="*80)

        return df


# Convenience function
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to create all features

    Args:
        df: Complete time series DataFrame

    Returns:
        pd.DataFrame: Data with all features
    """
    engine = FeatureEngine()
    return engine.create_all_features(df)
