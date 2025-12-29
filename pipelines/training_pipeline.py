#!/usr/bin/env python3
"""
Training Pipeline - Train XGBoost model

Works in both local and production modes:
  --mode local       : Load from local Parquet, save model locally
  --mode production  : Load from Hopsworks, save to Model Registry

Usage:
    python pipelines/training_pipeline.py --mode local
    python pipelines/training_pipeline.py --mode production --experiment-name weekly_retrain
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import json
import os
from datetime import datetime
from functions.storage_factory import get_storage, detect_mode


def prepare_training_data(df):
    """
    Prepare data for training

    Args:
        df: DataFrame with features and target

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    # Target variable
    target_col = 'price_sek_kwh_mean'

    # Features (exclude target and date)
    feature_cols = [col for col in df.columns if col not in [target_col, 'date']]

    X = df[feature_cols]
    y = df[target_col]

    # Time-based split (no shuffle to prevent data leakage)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols


def train_model(X_train, y_train, X_test, y_test):
    """
    Train XGBoost model

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        Trained model, metrics dict
    """
    print("  Training XGBoost model...")

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=8,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    metrics = {
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
        'train_r2': float(r2_score(y_train, y_pred_train)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
        'test_r2': float(r2_score(y_test, y_pred_test)),
        'n_training_samples': float(len(X_train)),
        'n_test_samples': float(len(X_test)),
        'best_iteration': float(
            model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
        ),
    }


    return model, metrics


def save_model_local(model, feature_names, metrics, experiment_name='default'):
    """Save model to local filesystem"""
    model_dir = f"data/models/electricity_price_xgboost_{experiment_name}"
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model.save_model(os.path.join(model_dir, "model.json"))

    # Save feature names
    with open(os.path.join(model_dir, "feature_names.json"), 'w') as f:
        json.dump(feature_names, f)

    # Save metrics
    with open(os.path.join(model_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  ✅ Model saved to: {model_dir}")
    return model_dir


def save_model_hopsworks(model, feature_names, metrics, storage, experiment_name='default'):
    """Save model to Hopsworks Model Registry"""
    mr = storage.get_model_registry()

    # Create model directory
    import tempfile
    import shutil
    model_dir = tempfile.mkdtemp()

    try:
        # Save model files
        model.save_model(os.path.join(model_dir, "model.json"))

        with open(os.path.join(model_dir, "feature_names.json"), 'w') as f:
            json.dump(feature_names, f)

        with open(os.path.join(model_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)

        # Register model
        model_name = f"electricity_price_xgboost"
        trained_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        registered_model = mr.python.create_model(
            name=model_name,
            metrics=metrics,
            description=(
                f"XGBoost model for electricity price prediction - {experiment_name} "
                f"(trained_at={trained_at})"
            ),
        )


        registered_model.save(model_dir)
        print(f"  ✅ Model saved to Hopsworks Model Registry: {model_name}")

    finally:
        shutil.rmtree(model_dir)


def main():
    parser = argparse.ArgumentParser(description='Training Pipeline')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['local', 'production'],
        default=None,
        help='Storage mode: local or production. Auto-detects if not specified.'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='default',
        help='Experiment name for organizing models'
    )

    args = parser.parse_args()

    # Auto-detect mode
    mode = args.mode if args.mode else detect_mode()
    print(f"\n{'='*70}")
    print(f"TRAINING PIPELINE - Mode: {mode.upper()}")
    print(f"{'='*70}")
    print(f"Experiment: {args.experiment_name}")

    # Step 1: Load data from storage
    print(f"\n[1/4] Loading data from {mode} storage...")
    storage = get_storage(mode)
    fs = storage.get_feature_store()

    electricity_fg = fs.get_or_create_feature_group(
        name="electricity_price",
        version=1
    )

    df = electricity_fg.read()
    print(f"  ✅ Loaded {len(df)} records")

    # Step 2: Prepare training data
    print(f"\n[2/4] Preparing training data...")
    X_train, X_test, y_train, y_test, feature_names = prepare_training_data(df)

    # Step 3: Train model
    print(f"\n[3/4] Training model...")
    model, metrics = train_model(X_train, y_train, X_test, y_test)

    print(f"\n  📊 Model Performance:")
    print(f"     Train RMSE: {metrics['train_rmse']:.4f} SEK/kWh")
    print(f"     Test RMSE:  {metrics['test_rmse']:.4f} SEK/kWh")
    print(f"     Test MAE:   {metrics['test_mae']:.4f} SEK/kWh")
    print(f"     Test R²:    {metrics['test_r2']:.4f}")

    # Step 4: Save model
    print(f"\n[4/4] Saving model to {mode} storage...")

    if mode == 'local':
        model_path = save_model_local(model, feature_names, metrics, args.experiment_name)
    else:
        save_model_hopsworks(model, feature_names, metrics, storage, args.experiment_name)

    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    if mode == 'local':
        print(f"  - Generate forecast: python pipelines/inference_pipeline.py --mode local")
        print(f"  - View model: ls -lh {model_path}")
    else:
        print(f"  - Check Hopsworks Model Registry")
        print(f"  - Generate forecast: python pipelines/inference_pipeline.py --mode production")


if __name__ == '__main__':
    main()
