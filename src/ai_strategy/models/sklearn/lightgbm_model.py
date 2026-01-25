"""LightGBM model wrapper for trading predictions."""

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from ...models.base import BaseModel


class LightGBMModel(BaseModel):
    """Wrapper for LightGBM model with training, prediction, and data processing capabilities."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        """Initialize the LightGBM model wrapper.

        Args:
            model_path: Optional path to load a trained model from.
        """
        self.model_path = Path(model_path) if model_path else None
        self.model: Any | None = None
        self.config: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}

        # Load the model if path is provided
        if self.model_path and self.model_path.exists():
            self.load()

    def prepare_data_for_training(
        self,
        df: pd.DataFrame,
        horizon: int = 16,
        min_movement: float = 0.0,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare data for LightGBM training.

        1. Generate target based on future returns.
        2. Split into Train/Test sets.

        Args:
            df: DataFrame with OHLCV and technical indicators.
            horizon: Number of candles to look ahead for target.
            min_movement: Minimum price movement to consider as UP (default: 0.0).

        Returns:
            Tuple of (X_train, y_train, X_test, y_test).
        """
        print("\n🎯 Preparing data for training...")
        print(f"Horizon: {horizon} candles, Min Movement: {min_movement * 100}%")
        print(f"📊 Initial data shape: {df.shape}")

        # Make a copy to avoid modifying the original
        df = df.copy()

        # Ensure data is sorted chronologically by timestamp
        print("\n🔍 Ensuring chronological order...")
        if "timestamp" in df.columns:
            print(f"   First timestamp before sort: {df['timestamp'].iloc[0]}")
            print(f"   Last timestamp before sort: {df['timestamp'].iloc[-1]}")
            df = df.sort_values("timestamp").reset_index(drop=True)
            print("   ✅ Data sorted by timestamp")
            print(f"   First timestamp after sort: {df['timestamp'].iloc[0]}")
            print(f"   Last timestamp after sort: {df['timestamp'].iloc[-1]}")
        else:
            print("   ⚠️  No 'timestamp' column found, assuming data is already sorted")

        # Create Target
        print(f"\n📈 Creating target with horizon={horizon}...")

        df["future_price"] = df["close"].shift(-horizon)
        df["future_return"] = (df["future_price"] - df["close"]) / df["close"]

        # Binary Target: 1 if return > min_movement, else 0
        df["target"] = (df["future_return"] > min_movement).astype(int)

        print("\n📊 Target distribution (before dropping NaN):")
        print(f"   Total rows: {len(df)}")
        print(f"   Target=1 (UP): {(df['target'] == 1).sum()}")
        print(f"   Target=0 (DOWN): {(df['target'] == 0).sum()}")
        print(f"   NaN: {df['target'].isna().sum()}")

        # Drop rows with NaN targets (last 'horizon' rows)
        before_drop = len(df)
        df.dropna(subset=["future_return"], inplace=True)
        after_drop = len(df)

        print(f"\n🧹 Dropped {before_drop - after_drop} rows with NaN targets")
        print(f"   Remaining rows: {after_drop}")

        # Select features (exclude target and auxiliary columns)
        exclude_cols = [
            "target",
            "future_return",
            "future_price",
            "timestamp",
            "symbol",
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        print("\n🔢 Feature selection:")
        print(f"   Total columns: {len(df.columns)}")
        print(f"   Excluded columns: {[c for c in exclude_cols if c in df.columns]}")
        print(f"   Feature columns: {len(feature_cols)}")

        X = df[feature_cols]
        y = df["target"]

        # Time Series Split (no random shuffle)
        split_point = int(len(df) * 0.8)

        print("\n✂️  Train/Test Split (80/20):")
        print(f"   Split point: {split_point}")

        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]

        print("\n✅ Data Split Summary:")
        print(
            f"   Train: {X_train.shape} | Positive Rate: {y_train.mean():.2%} | UP={y_train.sum()} DOWN={len(y_train) - y_train.sum()}"
        )
        print(
            f"   Test:  {X_test.shape}  | Positive Rate: {y_test.mean():.2%} | UP={y_test.sum()} DOWN={len(y_test) - y_test.sum()}"
        )

        # Store feature columns for later use
        self.feature_cols = feature_cols

        return X_train, y_train, X_test, y_test

    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 50,
        n_splits: int = 3,
    ) -> dict[str, Any]:
        """Optimize LightGBM hyperparameters using Optuna with TimeSeriesSplit.

        Args:
            X_train: Training features.
            y_train: Training target.
            n_trials: Number of optimization trials.
            n_splits: Number of time series splits for CV.

        Returns:
            Dictionary with best hyperparameters.
        """
        print("\n" + "=" * 80)
        print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
        print("=" * 80)
        print(f"Trials: {n_trials} | CV Splits: {n_splits}")

        def objective(trial):
            """Objective function for Optuna."""

            # Sample hyperparameters
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }

            # Time Series Cross-Validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            f1_scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
                )

                y_pred = model.predict(X_val)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                f1_scores.append(f1)

            # Return mean F1 score across all folds
            mean_f1 = sum(f1_scores) / len(f1_scores)
            return mean_f1

        # Create study and optimize
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")

        print("\n🔍 Starting optimization...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Results
        print("\n" + "=" * 80)
        print("OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"✅ Best F1 Score: {study.best_value:.4f}")
        print("\n📋 Best Hyperparameters:")
        for key, value in study.best_params.items():
            print(f"   {key:20s}: {value}")
        print("=" * 80)

        return study.best_params

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        params: dict[str, Any] | None = None,
        optimize: bool = False,
        n_trials: int = 50,
        **kwargs: Any,
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        """Train a LightGBM classifier for binary prediction.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_test: Test features.
            y_test: Test target.
            params: Custom hyperparameters (if None, uses defaults).
            optimize: If True, run Optuna optimization.
            n_trials: Number of trials for optimization.

        Returns:
            Tuple of (model, y_pred, y_proba).
        """
        print("\n" + "=" * 80)
        print("TRAINING LIGHTGBM CLASSIFIER")
        print("=" * 80)

        print("\n📊 Training data:")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Train samples: {len(X_train):,}")
        print(f"   Test samples: {len(X_test):,}")

        # Hyperparameter optimization if requested
        if optimize:
            print("\n🔧 Running hyperparameter optimization...")
            params = self._optimize_hyperparameters(X_train, y_train, n_trials=n_trials)
        elif params is None:
            # Default parameters
            params = {
                "n_estimators": 200,
                "max_depth": 7,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 50,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 1.0,
                "reg_lambda": 1.0,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }
            print("\n🔧 Using default hyperparameters")
        else:
            print("\n🔧 Using custom hyperparameters")

        print("\n📋 Model Configuration:")
        for key, value in params.items():
            if key not in ["random_state", "verbose", "n_jobs"]:
                print(f"   {key:20s}: {value}")

        # Model configuration
        self.model = lgb.LGBMClassifier(**params)

        print("\n🚀 Training started...")

        # Train with early stopping
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )

        print("✅ Training completed!")
        print(f"   Best iteration: {self.model.best_iteration_}")

        # Predictions
        print("\n📈 Generating predictions...")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]  # Probability of class 1 (UP)

        # Metrics
        print("\n" + "=" * 80)
        print("CLASSIFICATION METRICS")
        print("=" * 80)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        print("\n📊 Prediction Distribution:")
        print(
            f"   Predicted UP:   {(y_pred == 1).sum():,} ({(y_pred == 1).mean():.2%})"
        )
        print(
            f"   Predicted DOWN: {(y_pred == 0).sum():,} ({(y_pred == 0).mean():.2%})"
        )

        print("\n📊 Probability Statistics:")
        print(f"   Mean: {y_proba.mean():.4f}")
        print(f"   Std:  {y_proba.std():.4f}")
        print(f"   Min:  {y_proba.min():.4f}")
        print(f"   Max:  {y_proba.max():.4f}")

        print("=" * 80)

        # Store metrics
        self.metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "n_test_samples": len(y_test),
            "predicted_up_pct": (y_pred == 1).mean(),
        }

        return self.model, y_pred, y_proba

    def save(
        self, output_path: str | Path, config: dict[str, Any] | None = None
    ) -> None:
        """Save the trained model with metadata.

        Args:
            output_path: Path to save the model.
            config: Optional configuration dictionary (horizon, min_movement, etc.).
        """
        from pathlib import Path

        print("\n" + "=" * 80)
        print("SAVING MODEL")
        print("=" * 80)

        if config:
            self.config = config

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

        print(f"✅ Model saved to: {output_file}")
        print("\n📦 Model Package Contents:")
        print(f"   Model type: {type(self.model).__name__}")
        print(f"   Features: {len(self.feature_cols)}")
        print(f"   Timestamp: {model_data['timestamp']}")
        print("\n📊 Saved Metrics:")
        for key, value in self.metrics.items():
            print(
                f"   {key}: {value:.4f}"
                if isinstance(value, float)
                else f"   {key}: {value}"
            )
        print("\n⚙️  Configuration:")
        for key, value in self.config.items():
            print(f"   {key}: {value}")
        print("=" * 80)

    def load(self, model_path: str | Path | None = None) -> None:
        """Load the model from disk.

        Args:
            model_path: Optional path to load from (uses self.model_path if not provided).

        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        if model_path:
            self.model_path = Path(model_path)

        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        model_data = joblib.load(self.model_path)

        self.model = model_data["model"]
        self.feature_cols = model_data["feature_cols"]
        self.config = model_data.get("config", {})
        self.metrics = model_data.get("metrics", {})

    def prepare_features(self, features: pd.Series) -> pd.DataFrame:
        """Prepare features in the correct order for model prediction.

        Args:
            features: Series with feature values (e.g., from indicator calculation).

        Returns:
            DataFrame with features in the correct order expected by the model.
        """
        feature_values = []

        for col in self.feature_cols:
            if col in features.index:
                feature_values.append(features[col])
            else:
                # Use 0.0 as default for missing features
                feature_values.append(0.0)

        return pd.DataFrame([feature_values], columns=self.feature_cols)

    def predict_proba(self, X: pd.DataFrame) -> dict[str, float]:
        """Make probability predictions using the loaded model.

        Args:
            X: Features for prediction (DataFrame with correct columns).

        Returns:
            Dictionary with 'prob_down' and 'prob_up' probabilities.

        Raises:
            ValueError: If model is not loaded.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        # Get probabilities [prob_class_0, prob_class_1]
        # class_0 = DOWN (price will not go up)
        # class_1 = UP (price will go up)
        y_proba = self.model.predict_proba(X)[0]

        return {
            "prob_down": float(y_proba[0]),
            "prob_up": float(y_proba[1]),
        }

    def predict(self, features: pd.Series, **kwargs: Any) -> dict[str, float]:
        """End-to-end prediction from raw features.

        This method combines feature preparation and prediction.

        Args:
            features: Series with raw feature values.

        Returns:
            Dictionary with 'prob_down' and 'prob_up' probabilities.
        """
        X = self.prepare_features(features)
        return self.predict_proba(X)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded, False otherwise.
        """
        return self.model is not None

    @property
    def num_features(self) -> int:
        """Get the number of features expected by the model.

        Returns:
            Number of features.
        """
        return len(self.feature_cols)

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "loaded" if self.is_loaded else "not loaded"
        return (
            f"LightGBMModel(path={self.model_path}, "
            f"features={self.num_features}, status={status})"
        )
