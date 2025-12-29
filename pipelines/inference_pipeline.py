#!/usr/bin/env python3
"""
Inference Pipeline - Generate 7-day electricity price forecast

Works in both local and production modes:
  --mode local       : Load model from local files
  --mode production  : Load from Hopsworks Model Registry

Usage:
    python pipelines/inference_pipeline.py --mode local --days 7
    python pipelines/inference_pipeline.py --mode production --days 10
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from functions.util import get_weather_forecast
from functions.storage_factory import get_storage, detect_mode


def load_model_local(experiment_name='default'):
    """Load model from local filesystem"""
    model_dir = f"data/models/electricity_price_xgboost_{experiment_name}"

    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model not found: {model_dir}\n"
            f"Train a model first: python pipelines/training_pipeline.py --mode local"
        )

    model = xgb.XGBRegressor()
    model.load_model(os.path.join(model_dir, "model.json"))

    with open(os.path.join(model_dir, "feature_names.json"), 'r') as f:
        feature_names = json.load(f)

    print(f"  ✅ Model loaded from: {model_dir}")
    return model, feature_names


def load_model_hopsworks(storage):
    """Load model from Hopsworks Model Registry"""
    mr = storage.get_model_registry()

    model_name = "electricity_price_xgboost"
    model = mr.get_model(model_name, version=1)

    # Download model files
    model_dir = model.download()

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(model_dir, "model.json"))

    with open(os.path.join(model_dir, "feature_names.json"), 'r') as f:
        feature_names = json.load(f)

    print(f"  ✅ Model loaded from Hopsworks: {model_name}")
    return xgb_model, feature_names


def prepare_forecast_features(weather_forecast_df, historical_df, feature_names):
    """
    Prepare features for forecast

    Args:
        weather_forecast_df: Future weather data
        historical_df: Historical data for lag features
        feature_names: List of features expected by model

    Returns:
        DataFrame with all required features
    """
    forecast_df = weather_forecast_df.copy()
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])

    # Temporal features
    forecast_df['hour'] = forecast_df['date'].dt.hour
    forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek
    forecast_df['month'] = forecast_df['date'].dt.month
    forecast_df['is_weekend'] = (forecast_df['day_of_week'] >= 5).astype(int)

    # Weather features
    forecast_df['temp_squared'] = forecast_df['temperature_2m_mean'] ** 2
    forecast_df['wind_temp_interaction'] = (
        forecast_df['wind_speed_10m_max'] * forecast_df['temperature_2m_mean']
    )

    # Lag features from historical data
    recent_prices = historical_df['price_sek_kwh_mean'].tail(200).values

    if len(recent_prices) >= 24:
        forecast_df['price_lag_1d'] = recent_prices[-24]
    else:
        forecast_df['price_lag_1d'] = recent_prices[-1]

    if len(recent_prices) >= 168:
        forecast_df['price_lag_7d'] = recent_prices[-168]
    else:
        forecast_df['price_lag_7d'] = recent_prices[-1]

    # Rolling statistics
    if len(recent_prices) >= 168:
        forecast_df['price_rolling_mean_7d'] = np.mean(recent_prices[-168:])
        forecast_df['price_rolling_std_7d'] = np.std(recent_prices[-168:])
    else:
        forecast_df['price_rolling_mean_7d'] = np.mean(recent_prices)
        forecast_df['price_rolling_std_7d'] = np.std(recent_prices)

    # Ensure all required features exist
    for feature in feature_names:
        if feature not in forecast_df.columns:
            forecast_df[feature] = 0  # Default value for missing features

    return forecast_df[feature_names]


def create_forecast_visualization(forecast_df, output_path='outputs/forecast.png'):
    """Create forecast-only visualization (next 7 days)"""
    os.makedirs('outputs', exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Forecast only
    ax.plot(forecast_df['date'], forecast_df['predicted_price'],
            label='7-Day Forecast', color='#E63946', linewidth=3,
            marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2)

    # Styling
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price (SEK/kWh)', fontsize=13, fontweight='bold')
    ax.set_title('Electricity Price Forecast - Stockholm (SE3)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')

    # Add value labels on points
    for i, (date, price) in enumerate(zip(forecast_df['date'], forecast_df['predicted_price'])):
        ax.text(date, price + 0.01, f'{price:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Forecast chart saved: {output_path}")

    return output_path


def create_comparison_visualization(historical_df, output_path='outputs/predicted_vs_actual.png'):
    """
    Create predicted vs actual comparison over time
    Single line graph with two lines: predicted and actual prices
    """
    os.makedirs('outputs', exist_ok=True)

    # Load prediction tracking file
    tracking_file = 'outputs/prediction_tracking.csv'
    if not os.path.exists(tracking_file):
        print(f"  ℹ️  No tracking data yet. Run daily to build comparison history.")
        return None

    tracking_df = pd.read_csv(tracking_file)

    tracking_df['target_date'] = (
        pd.to_datetime(tracking_df['target_date'], utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    historical_df['date'] = (
        pd.to_datetime(historical_df['date'], utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    actual_prices = historical_df[['date', 'price_sek_kwh_mean']].copy()
    actual_prices.columns = ['target_date', 'actual_price']

    comparison_df = tracking_df.merge(
        actual_prices,
        on='target_date',
        how='inner'
    )


    # Merge predictions with actuals (only for dates where we have actuals)
    comparison_df = tracking_df.merge(actual_prices, on='target_date', how='inner')

    if comparison_df.empty:
        print(f"  ℹ️  No actual data available yet for comparison.")
        return None

    # Keep only latest prediction for each date
    comparison_df = comparison_df.sort_values(['target_date', 'prediction_date'])
    comparison_df = comparison_df.drop_duplicates('target_date', keep='last')
    comparison_df = comparison_df.sort_values('target_date')

    # Calculate metrics
    comparison_df['error'] = comparison_df['predicted_price'] - comparison_df['actual_price']
    comparison_df['abs_error'] = comparison_df['error'].abs()
    mae = comparison_df['abs_error'].mean()
    rmse = np.sqrt((comparison_df['error'] ** 2).mean())

    # Create single line graph
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot predicted and actual lines
    ax.plot(comparison_df['target_date'], comparison_df['predicted_price'],
            label='Predicted', color='#E63946', linewidth=3,
            marker='o', markersize=8, alpha=0.9)
    ax.plot(comparison_df['target_date'], comparison_df['actual_price'],
            label='Actual', color='#2E86AB', linewidth=3,
            marker='s', markersize=8, alpha=0.9)

    # Styling
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price (SEK/kWh)', fontsize=13, fontweight='bold')
    ax.set_title('Predicted vs Actual Electricity Prices - Stockholm (SE3)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')

    # Add metrics text box
    metrics_text = f'MAE: {mae:.4f} SEK/kWh | RMSE: {rmse:.4f} SEK/kWh | Days: {len(comparison_df)}'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Comparison chart saved: {output_path}")
    print(f"     MAE: {mae:.4f} SEK/kWh | RMSE: {rmse:.4f} SEK/kWh | {len(comparison_df)} days")

    return output_path


def save_predictions_for_tracking(forecast_df, prediction_date):
    """
    Save predictions to tracking file for future comparison

    Args:
        forecast_df: DataFrame with predictions
        prediction_date: Date when predictions were made
    """
    tracking_file = 'outputs/prediction_tracking.csv'

    # Prepare tracking data
    tracking_data = forecast_df[['date', 'predicted_price']].copy()
    tracking_data.columns = ['target_date', 'predicted_price']
    tracking_data['prediction_date'] = prediction_date
    tracking_data['prediction_date'] = pd.to_datetime(tracking_data['prediction_date'])
    tracking_data['target_date'] = pd.to_datetime(tracking_data['target_date'])

    # Append to tracking file
    if os.path.exists(tracking_file):
        existing_df = pd.read_csv(tracking_file)
        existing_df['prediction_date'] = pd.to_datetime(existing_df['prediction_date'])
        existing_df['target_date'] = pd.to_datetime(existing_df['target_date'])

        # Remove old predictions for same target dates (keep latest)
        existing_df = existing_df[~existing_df['target_date'].isin(tracking_data['target_date'])]

        # Combine and sort
        combined_df = pd.concat([existing_df, tracking_data], ignore_index=True)
        combined_df = combined_df.sort_values(['target_date', 'prediction_date'])
        combined_df.to_csv(tracking_file, index=False)
    else:
        tracking_data.to_csv(tracking_file, index=False)

    print(f"  ✅ Predictions saved to tracking file: {tracking_file}")


def main():
    parser = argparse.ArgumentParser(description='Inference Pipeline')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['local', 'production'],
        default=None,
        help='Storage mode. Auto-detects if not specified.'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to forecast (excluding today)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='default',
        help='Experiment name (for local mode)'
    )
    parser.add_argument(
        '--latitude',
        type=float,
        default=59.33,
        help='Latitude for weather forecast'
    )
    parser.add_argument(
        '--longitude',
        type=float,
        default=18.07,
        help='Longitude for weather forecast'
    )

    args = parser.parse_args()

    # Auto-detect mode
    mode = args.mode if args.mode else detect_mode()
    print(f"\n{'='*70}")
    print(f"INFERENCE PIPELINE - Mode: {mode.upper()}")
    print(f"{'='*70}")
    print(f"Forecast horizon: {args.days} days")

    # Step 1: Load model
    print(f"\n[1/5] Loading model from {mode} storage...")

    if mode == 'local':
        model, feature_names = load_model_local(args.experiment_name)
    else:
        storage = get_storage(mode)
        model, feature_names = load_model_hopsworks(storage)

    # Step 2: Load historical data for lag features
    print(f"\n[2/5] Loading historical data...")
    storage = get_storage(mode)
    fs = storage.get_feature_store()

    electricity_fg = fs.get_or_create_feature_group(name="electricity_price", version=1)
    historical_df = electricity_fg.read()
    historical_df['date'] = pd.to_datetime(historical_df['date'])
    historical_df = historical_df.sort_values('date')

    print(f"  ✅ Loaded {len(historical_df)} historical records")
    print(f"  Latest data: {historical_df['date'].max()}")

    # Step 3: Get weather forecast
    print(f"\n[3/5] Fetching weather forecast...")
    weather_forecast = get_weather_forecast(
        days_ahead=args.days + 3,  # Get extra to ensure we have enough
        latitude=args.latitude,
        longitude=args.longitude
    )
    weather_forecast['date'] = pd.to_datetime(weather_forecast['date'])

    # Filter to exclude today
    tomorrow = (datetime.now() + timedelta(days=1)).date()
    weather_forecast = weather_forecast[weather_forecast['date'].dt.date >= tomorrow].head(args.days)

    print(f"  ✅ Retrieved {len(weather_forecast)} days of forecast")

    # Step 4: Prepare features and predict
    print(f"\n[4/5] Generating predictions...")
    forecast_features = prepare_forecast_features(weather_forecast, historical_df, feature_names)
    predictions = model.predict(forecast_features)

    forecast_df = weather_forecast[['date']].copy()
    forecast_df['predicted_price'] = predictions

    print(f"  ✅ Generated {len(forecast_df)} predictions")
    print(f"\n  📊 Forecast Summary:")
    print(f"     Date range: {forecast_df['date'].min().date()} to {forecast_df['date'].max().date()}")
    print(f"     Avg price:  {forecast_df['predicted_price'].mean():.3f} SEK/kWh")
    print(f"     Min price:  {forecast_df['predicted_price'].min():.3f} SEK/kWh")
    print(f"     Max price:  {forecast_df['predicted_price'].max():.3f} SEK/kWh")

    # Step 5: Save results
    print(f"\n[5/5] Saving results...")

    os.makedirs("outputs", exist_ok=True)

    # Save CSV
    today_str = datetime.now().strftime('%Y%m%d')
    csv_path = f"outputs/forecast_{today_str}.csv"
    forecast_df.to_csv(csv_path, index=False)
    print(f"  ✅ Forecast saved: {csv_path}")

    # Save predictions for tracking (for future comparison with actuals)
    prediction_date = datetime.now().date()
    save_predictions_for_tracking(forecast_df, prediction_date)

    # Create forecast visualization (only future predictions)
    forecast_chart_path = f"outputs/forecast_{today_str}.png"
    create_forecast_visualization(forecast_df, forecast_chart_path)

    # Create comparison visualization (predicted vs actual over time)
    comparison_chart_path = f"outputs/predicted_vs_actual_{today_str}.png"
    create_comparison_visualization(historical_df, comparison_chart_path)

    print(f"\n{'='*70}")
    print(f"✅ INFERENCE COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  - Forecast data: {csv_path}")
    print(f"  - Forecast chart: {forecast_chart_path}")
    if os.path.exists(comparison_chart_path):
        print(f"  - Comparison chart: {comparison_chart_path}")
    print(f"\nℹ️  Run daily to build prediction vs actual comparison history!")


if __name__ == '__main__':
    main()
