#!/usr/bin/env python3
"""
Feature Backfill Pipeline - Collect historical data

Works in both local and production modes:
  --mode local       : Save to local Parquet files (fast, no cloud)
  --mode production  : Save to Hopsworks Feature Store (requires API key)

Usage:
    python pipelines/feature_backfill.py --mode local --start-date 2023-01-01 --end-date 2024-12-31
    python pipelines/feature_backfill.py --mode production --start-date 2020-01-01
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import argparse
import pandas as pd
from datetime import datetime, timedelta
from functions.util import get_historical_weather, get_electricity_prices
from functions.storage_factory import get_storage, detect_mode


def engineer_features(weather_df, price_df):
    """
    Engineer features from raw weather and price data

    Args:
        weather_df: Weather data with columns [date, temperature_2m_mean, ...]
        price_df: Price data with columns [date, price_sek_kwh_mean, ...]

    Returns:
        DataFrame with engineered features
    """
    # Merge weather and price data
    merged_df = pd.merge(weather_df, price_df, on='date', how='inner')
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df = merged_df.sort_values('date')

    # Temporal features (daily granularity)
    merged_df['day_of_week'] = merged_df['date'].dt.dayofweek
    merged_df['month'] = merged_df['date'].dt.month
    merged_df['is_weekend'] = (merged_df['day_of_week'] >= 5).astype('int32')
    merged_df['day_of_year'] = merged_df['date'].dt.dayofyear

    # Weather features
    merged_df['temp_squared'] = merged_df['temperature_2m_mean'] ** 2
    merged_df['wind_temp_interaction'] = (
        merged_df['wind_speed_10m_max'] * merged_df['temperature_2m_mean']
    )

    # Lag features (price history) - DAILY DATA
    merged_df['price_lag_1d'] = merged_df['price_sek_kwh_mean'].shift(1)  # 1 day ago
    merged_df['price_lag_7d'] = merged_df['price_sek_kwh_mean'].shift(7)  # 7 days ago

    # Rolling statistics - DAILY DATA
    merged_df['price_rolling_mean_7d'] = (
        merged_df['price_sek_kwh_mean'].rolling(window=7, min_periods=1).mean()
    )
    merged_df['price_rolling_std_7d'] = (
        merged_df['price_sek_kwh_mean'].rolling(window=7, min_periods=1).std()
    )

    # Drop rows with NaN only in critical columns
    # Allow NaN in long-term lags (price_lag_7d) for small datasets
    critical_cols = ['price_sek_kwh_mean', 'temperature_2m_mean', 'price_lag_1d']
    merged_df = merged_df.dropna(subset=critical_cols)

    # Fill remaining NaN in lag features with forward fill
    merged_df = merged_df.ffill()

    return merged_df


def main():
    parser = argparse.ArgumentParser(description='Feature Backfill Pipeline')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['local', 'production'],
        default=None,
        help='Storage mode: local (Parquet) or production (Hopsworks). Auto-detects if not specified.'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2023-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD). Defaults to yesterday.'
    )
    parser.add_argument(
        '--location',
        type=str,
        default='Stockholm',
        help='City name for metadata'
    )
    parser.add_argument(
        '--latitude',
        type=float,
        default=59.33,
        help='Latitude for weather data'
    )
    parser.add_argument(
        '--longitude',
        type=float,
        default=18.07,
        help='Longitude for weather data'
    )

    args = parser.parse_args()

    # Auto-detect mode if not specified
    mode = args.mode if args.mode else detect_mode()
    print(f"\n{'='*70}")
    print(f"FEATURE BACKFILL PIPELINE - Mode: {mode.upper()}")
    print(f"{'='*70}")

    # Default end date to yesterday
    if not args.end_date:
        args.end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"\n📅 Date Range: {args.start_date} to {args.end_date}")
    print(f"📍 Location: {args.location} ({args.latitude}, {args.longitude})")

    # Step 1: Fetch weather data
    print(f"\n[1/4] Fetching weather data...")
    weather_df = get_historical_weather(
        start_date=args.start_date,
        end_date=args.end_date,
        latitude=args.latitude,
        longitude=args.longitude
    )

    if weather_df.empty:
        print("❌ No weather data retrieved. Exiting.")
        sys.exit(1)

    print(f"  ✅ Retrieved {len(weather_df)} weather records")

    # Step 2: Fetch electricity prices
    print(f"\n[2/4] Fetching electricity prices...")
    price_df = get_electricity_prices(
        start_date=args.start_date,
        end_date=args.end_date,
        region='SE3'
    )

    if price_df.empty:
        print("❌ No price data retrieved. Exiting.")
        sys.exit(1)

    print(f"  ✅ Retrieved {len(price_df)} price records")

    # Step 3: Engineer features
    print(f"\n[3/4] Engineering features...")
    featured_df = engineer_features(weather_df, price_df)
    print(f"  ✅ Engineered {len(featured_df)} samples with {len(featured_df.columns)} features")

    # Step 4: Save to storage
    print(f"\n[4/4] Saving to {mode} storage...")
    storage = get_storage(mode)
    fs = storage.get_feature_store()

    # Create feature groups
    electricity_fg = fs.get_or_create_feature_group(
        name="electricity_price",
        version=1,
        description=f"Electricity prices for {args.location} with engineered features",
        primary_key=['date'],
        event_time='date',
    )
  
    # Ensure feature types match Feature Group schema
    INT_COLS = [
        "is_weekend",
        "day_of_week",
        "day_of_year",
    ]
    
    for col in INT_COLS:
        if col in featured_df.columns:
            featured_df[col] = featured_df[col].astype(int)


    # Insert data
    electricity_fg.insert(featured_df, overwrite=False)

    print(f"  ✅ Saved {len(featured_df)} records to feature group 'electricity_price'")

    print(f"\n{'='*70}")
    print(f"✅ BACKFILL COMPLETE!")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    if mode == 'local':
        print(f"  - Train model: python pipelines/training_pipeline.py --mode local")
        print(f"  - View data: ls -lh data/processed/")
    else:
        print(f"  - Check Hopsworks UI for feature group 'electricity_price'")
        print(f"  - Train model: python pipelines/training_pipeline.py --mode production")


if __name__ == '__main__':
    main()
