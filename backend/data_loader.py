"""
Data loading and preprocessing module
Handles loading raw sales and manufacturing data,  SKU normalization, and skeleton creation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from backend.config import (
    RAW_DATA_FILE, PRODUCT_FILE,
    RAW_DATA_COLUMNS, MANUFACTURING_TIME_COLUMNS, PRODUCT_SHEETS
)


class DataLoader:
    """
    Handles all data loading and preprocessing operations
    """

    def __init__(self):
        """Initialize the DataLoader"""
        self.df_sales = None
        self.df_manufacturing = None
        self.df_complete = None

    def load_sales_data(self) -> pd.DataFrame:
        """
        Load raw sales transaction data from Excel

        Returns:
            pd.DataFrame: Sales data with normalized SKUs
        """
        print("Loading raw sales data...")
        df = pd.read_excel(RAW_DATA_FILE, sheet_name='Auftragsdaten')

        # Remove location column (not needed)
        df = df.drop(columns=['Ort'], errors='ignore')

        # Convert date column to datetime
        df['Auftragsdatum'] = pd.to_datetime(df['Auftragsdatum'])

        # Normalize SKU column
        # Step 1: Remove dots from SKUs
        mask = df['SKU'].str.contains('.', regex=False, na=False)
        df.loc[mask, 'SKU'] = df.loc[mask, 'SKU'].str.replace('.', '', regex=False)

        # Step 2: Convert to lowercase
        df['SKU'] = df['SKU'].str.lower()

        print(f"Loaded {len(df):,} transactions")
        print(f"Date range: {df['Auftragsdatum'].min()} to {df['Auftragsdatum'].max()}")
        print(f"Unique products: {df['SKU'].nunique()}")

        self.df_sales = df
        return df

    def load_manufacturing_times(self) -> pd.DataFrame:
        """
        Load manufacturing time data from product details Excel

        Returns:
            pd.DataFrame: Manufacturing times for each SKU
        """
        print("\nLoading manufacturing time data...")

        all_products = []

        for product_type, sheet_name in PRODUCT_SHEETS.items():
            # Load sheet
            df = pd.read_excel(PRODUCT_FILE, sheet_name=sheet_name, header=1)

            # Calculate total manufacturing time
            df['manufacturing_time'] = df[MANUFACTURING_TIME_COLUMNS].fillna(0).sum(axis=1)

            # Add product type
            df['product_type'] = product_type

            # Keep only relevant columns
            df = df[['SKU', 'manufacturing_time', 'product_type']]

            all_products.append(df)

        # Combine all product types
        df_manufacturing = pd.concat(all_products, ignore_index=True)

        # Normalize SKU to lowercase
        df_manufacturing['SKU'] = df_manufacturing['SKU'].str.lower()

        print(f"Loaded manufacturing times for {df_manufacturing['SKU'].nunique()} unique SKUs")
        print(f"Manufacturing time range: {df_manufacturing['manufacturing_time'].min():.0f} - {df_manufacturing['manufacturing_time'].max():.0f} days")

        self.df_manufacturing = df_manufacturing
        return df_manufacturing

    def merge_data(self) -> pd.DataFrame:
        """
        Merge sales data with manufacturing times

        Returns:
            pd.DataFrame: Merged and cleaned dataset
        """
        if self.df_sales is None:
            self.load_sales_data()
        if self.df_manufacturing is None:
            self.load_manufacturing_times()

        print("\nMerging sales with manufacturing times...")

        # Merge
        df = self.df_sales.merge(
            self.df_manufacturing,
            on='SKU',
            how='left'
        )

        # Check for unmatched SKUs
        unmatched_count = df['manufacturing_time'].isna().sum()
        if unmatched_count > 0:
            unmatched_skus = df[df['manufacturing_time'].isna()]['SKU'].unique()
            print(f"Warning: {unmatched_count} records ({len(unmatched_skus)} unique SKUs) without manufacturing time")
            print("Removing unmatched SKUs from dataset...")
            df = df[df['manufacturing_time'].notna()]

        print(f"Final dataset: {len(df):,} transactions")
        print(f"Unique products with manufacturing time: {df['SKU'].nunique()}")

        return df

    def aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate transactions to daily level per SKU

        Args:
            df: Merged sales data with manufacturing times

        Returns:
            pd.DataFrame: Daily aggregated data
        """
        print("\nAggregating to daily time series per SKU...")

        # Calculate price per unit
        df['price_per_unit'] = df['Gesamtpreis €'] / df['Auftragspositionen/Menge']

        # Normalize dates to daily level
        df['date'] = df['Auftragsdatum'].dt.normalize()

        # Aggregate by SKU and date
        df_daily = df.groupby(['SKU', 'date', 'manufacturing_time', 'product_type']).agg({
            'Auftragspositionen/Menge': 'sum',
            'price_per_unit': 'mean',
            'Gesamtpreis €': 'sum'
        }).reset_index()

        df_daily.columns = ['SKU', 'date', 'manufacturing_time', 'product_type',
                            'quantity', 'price_per_unit', 'total_price']

        print(f"Daily aggregated data: {len(df_daily):,} records")

        return df_daily

    def create_daily_skeleton(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Create complete date skeleton for all SKUs
        Ensures every SKU has an entry for every day (with 0 sales on days without transactions)

        Args:
            df_daily: Daily aggregated sales data

        Returns:
            pd.DataFrame: Complete time series with all dates
        """
        print("\nCreating complete date skeleton for all SKUs...")

        # Get date range
        min_date = df_daily['date'].min()
        max_date = df_daily['date'].max()
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D')

        # Get unique SKU metadata
        sku_metadata = df_daily[['SKU', 'manufacturing_time', 'product_type']].drop_duplicates()

        # Create skeleton: all combinations of SKU × Date
        skeleton_data = []
        for _, row in sku_metadata.iterrows():
            for date in all_dates:
                skeleton_data.append({
                    'SKU': row['SKU'],
                    'date': date,
                    'manufacturing_time': row['manufacturing_time'],
                    'product_type': row['product_type']
                })

        df_skeleton = pd.DataFrame(skeleton_data)

        # Merge skeleton with actual sales data
        df_complete = df_skeleton.merge(
            df_daily[['SKU', 'date', 'quantity', 'price_per_unit']],
            on=['SKU', 'date'],
            how='left'
        )

        # Fill missing quantities with 0
        df_complete['quantity'] = df_complete['quantity'].fillna(0)

        # Fill missing prices with SKU average price
        sku_avg_price = df_daily.groupby('SKU')['price_per_unit'].mean()
        df_complete['price_per_unit'] = df_complete.apply(
            lambda row: row['price_per_unit'] if pd.notna(row['price_per_unit'])
            else sku_avg_price.get(row['SKU'], 0),
            axis=1
        )

        print(f"Complete time series: {len(df_complete):,} rows")
        print(f"({df_complete['SKU'].nunique()} SKUs × {len(all_dates)} days)")

        self.df_complete = df_complete
        return df_complete

    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Complete pipeline: load, merge, aggregate, and create skeleton

        Returns:
            pd.DataFrame: Complete prepared dataset ready for feature engineering
        """
        print("="*80)
        print("DATA LOADING AND PREPARATION PIPELINE")
        print("="*80)

        # Step 1: Load sales data
        df_sales = self.load_sales_data()

        # Step 2: Load manufacturing times
        df_manufacturing = self.load_manufacturing_times()

        # Step 3: Merge datasets
        df_merged = self.merge_data()

        # Step 4: Aggregate to daily
        df_daily = self.aggregate_to_daily(df_merged)

        # Step 5: Create complete skeleton
        df_complete = self.create_daily_skeleton(df_daily)

        print("\n" + "="*80)
        print("DATA PREPARATION COMPLETE")
        print("="*80)

        return df_complete


# Convenience function for quick loading
def load_data() -> pd.DataFrame:
    """
    Convenience function to load and prepare data in one call

    Returns:
        pd.DataFrame: Complete prepared dataset
    """
    loader = DataLoader()
    return loader.load_and_prepare_data()
