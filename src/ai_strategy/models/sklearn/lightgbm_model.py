"""LightGBM model wrapper for trading predictions."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class LightGBMModel:
    """Wrapper for LightGBM model with prediction capabilities.

    This class encapsulates model loading, feature preparation,
    and prediction logic for LightGBM-based trading models.
    """

    def __init__(self, model_path: str | Path) -> None:
        """Initialize the LightGBM model wrapper.

        Args:
            model_path: Path to the trained model file (.pkl).
        """
        self.model_path = Path(model_path)
        self.model: Any | None = None
        self.feature_cols: list[str] = []
        self.config: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}

        # Load the model
        self.load()

    def load(self) -> None:
        """Load the model from disk.

        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        if not self.model_path.exists():
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

    def predict(self, features: pd.Series) -> dict[str, float]:
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
