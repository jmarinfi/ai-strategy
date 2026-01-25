from pathlib import Path
from typing import Any
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import warnings

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.ensemble import RandomForestRegressor

from ...models.base import BaseModel

warnings.filterwarnings("ignore")


class RandomForest(BaseModel):
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else None
        self.model: Any | None = None
        self.feature_cols: list[str] = []
        self.config: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}

        # Load the model if path is provided
        if self.model_path and self.model_path.exists():
            self.load()
            print(f"✅ Model loaded from {self.model_path}")

    def prepare_data_for_training(
        self, df: pd.DataFrame, n_lags: int = 6, test_size: float = 0.2, **kwargs: Any
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare data for training using sliding window approach (Machine Learning Mastery method).

        This follows the EXACT approach from:
        https://machinelearningmastery.com/random-forest-for-time-series-forecasting/

        Uses n_lags previous time steps to predict the NEXT value (one-step-ahead).

        Args:
            df: DataFrame with OHLCV data and technical indicators
            n_lags: Number of previous time steps to use as features (default: 6)
            test_size: Proportion of data to use for testing (default: 0.2)

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        print(
            "\n📊 Preparing data using sliding window approach (ML Mastery method)..."
        )
        print(f"   Initial shape: {df.shape}")
        print(f"   Lag window: {n_lags} previous steps")
        print("   Prediction: next value (one-step-ahead)")

        # Select only technical indicators (exclude timestamp, OHLC)
        # Following ML Mastery: transform series using lag features
        exclude_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        indicator_cols = [col for col in df.columns if col not in exclude_cols]

        print(f"\n   📈 Using {len(indicator_cols)} technical indicators:")
        print(f"   {indicator_cols}")

        # Create supervised learning dataset using sliding window
        X_list = []
        y_list = []

        # We need at least n_lags+1 data points (n_lags for input, 1 for output)
        for i in range(n_lags, len(df)):
            # Create feature vector: concatenate n_lags previous values for each indicator
            feature_vector = []

            # Add lagged values for each technical indicator
            for lag in range(n_lags, 0, -1):  # From oldest to most recent
                lag_idx = i - lag
                for col in indicator_cols:
                    feature_vector.append(df[col].iloc[lag_idx])

            X_list.append(feature_vector)

            # Target: close price at current time step (next value after the lag window)
            y_list.append(df["close"].iloc[i])

        # Convert to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)

        # Store feature names for interpretability
        self.feature_cols = []
        for lag in range(n_lags, 0, -1):
            for col in indicator_cols:
                self.feature_cols.append(f"{col}_lag_{lag}")

        # Store n_lags in config for later use in predict()
        self.config["n_lags"] = n_lags

        print("\n   📊 Transformed dataset:")
        print(f"   X shape: {X.shape} (samples, features)")
        print(f"   y shape: {y.shape}")
        print(f"   Total features: {len(self.feature_cols)}")
        print(f"   Features = {n_lags} lags × {len(indicator_cols)} indicators")

        # TEMPORAL SPLIT (no shuffling!) - respects time order
        # First (1-test_size) for training, last test_size for testing
        split_idx = int(len(X) * (1 - test_size))

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        print(f"\n   ✅ Train set: {X_train.shape[0]} samples")
        print(f"   ✅ Test set:  {X_test.shape[0]} samples")
        print(f"   Train period: index 0 to {split_idx - 1}")
        print(f"   Test period:  index {split_idx} to {len(X) - 1}\n")

        return X_train, y_train, X_test, y_test

    def load(self, model_path: str | Path | None = None) -> None:
        if model_path:
            self.model_path = Path(model_path)

        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        model_data = joblib.load(self.model_path)

        self.model = model_data["model"]
        self.feature_cols = model_data["feature_cols"]
        self.config = model_data.get("config", {})
        self.metrics = model_data.get("metrics", {})

    def train(self, X_train, y_train, X_test, y_test, **kwargs):
        print("\n🔧 Training Random Forest model...")

        # Initialize and train the model
        regressor = RandomForestRegressor(
            n_estimators=100, random_state=42, oob_score=True, verbose=2
        )
        regressor.fit(X_train, y_train)

        # Save the trained model
        self.model = regressor

        # Make predictions
        y_train_pred = regressor.predict(X_train)
        y_test_pred = regressor.predict(X_test)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        train_rmse = np.sqrt(train_mse)  # Fixed: RMSE = sqrt(MSE)
        test_rmse = np.sqrt(test_mse)  # Fixed: RMSE = sqrt(MSE)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

        oob_score = regressor.oob_score_

        # Store metrics
        self.metrics = {
            "train": {
                "mae": train_mae,
                "mse": train_mse,
                "rmse": train_rmse,
                "r2": train_r2,
                "mape": train_mape,
            },
            "test": {
                "mae": test_mae,
                "mse": test_mse,
                "rmse": test_rmse,
                "r2": test_r2,
                "mape": test_mape,
            },
            "oob_score": oob_score,
        }

        # Print metrics
        print("\n" + "=" * 60)
        print("📊 TRAINING METRICS")
        print("=" * 60)
        print(f"  MAE:  {train_mae:.6f}")
        print(f"  MSE:  {train_mse:.6f}")
        print(f"  RMSE: {train_rmse:.6f}")
        print(f"  R²:   {train_r2:.6f}")
        print(f"  MAPE: {train_mape:.4f}%")
        print("\n" + "=" * 60)
        print("📊 TEST METRICS")
        print("=" * 60)
        print(f"  MAE:  {test_mae:.6f}")
        print(f"  MSE:  {test_mse:.6f}")
        print(f"  RMSE: {test_rmse:.6f}")
        print(f"  R²:   {test_r2:.6f}")
        print(f"  MAPE: {test_mape:.4f}%")
        print("\n" + "=" * 60)
        print(f"📊 OOB SCORE: {oob_score:.6f}")
        print("=" * 60 + "\n")

    def save(
        self, output_path: str | Path, config: dict[str, Any] | None = None
    ) -> None:
        # Merge incoming config with existing config (don't overwrite)
        if config:
            self.config.update(config)

        # Prepare model package
        model_data = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "metrics": self.metrics,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
            "n_features": len(self.feature_cols),
        }

        # Ensure directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save
        joblib.dump(model_data, output_file)
        self.model_path = output_file

    def predict(
        self, df: pd.DataFrame, n_lags: int | None = None, **kwargs: Any
    ) -> float:
        """Make a prediction for the next close price.

        Args:
            df: DataFrame with at least n_lags rows of technical indicators
                (should contain the same indicators used during training)
            n_lags: Number of lag steps (if None, will try to get from config)

        Returns:
            Predicted close price for the next candle
        """
        if self.model is None:
            raise ValueError(
                "Model not trained or loaded. Call train() or load() first."
            )

        # Get n_lags from parameter or config
        if n_lags is None:
            n_lags = self.config.get("n_lags")
            if n_lags is None:
                raise ValueError("n_lags not provided and not found in config")

        # Check we have enough data
        if len(df) < n_lags:
            raise ValueError(f"Need at least {n_lags} rows of data, got {len(df)}")

        # Get indicator columns (exclude OHLCV)
        exclude_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        indicator_cols = [col for col in df.columns if col not in exclude_cols]

        # Take last n_lags rows
        recent_data = df.iloc[-n_lags:]

        # Create feature vector (same as in training)
        feature_vector = []
        for lag in range(n_lags, 0, -1):  # From oldest to most recent
            lag_idx = n_lags - lag
            for col in indicator_cols:
                feature_vector.append(recent_data[col].iloc[lag_idx])

        # Reshape for prediction
        X = np.array([feature_vector])

        # Make prediction
        prediction = self.model.predict(X)[0]

        return float(prediction)
