from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all strategy models.

    This interface defines the common methods that must be implemented
    by any model used in the strategy (e.g., LightGBM, RandomForest).
    """

    @abstractmethod
    def prepare_data_for_training(
        self, df: pd.DataFrame, **kwargs: Any
    ) -> tuple[Any, Any, Any, Any]:
        """Prepare data for training.

        Args:
            df: Input DataFrame with raw data.
            **kwargs: Additional arguments specific to the model data preparation.

        Returns:
            Tuple containing (X_train, y_train, X_test, y_test).
        """
        pass

    @abstractmethod
    def train(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        **kwargs: Any,
    ) -> Any:
        """Train the model.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_test: Test features.
            y_test: Test target.
            **kwargs: Model-specific training parameters.

        Returns:
            Model-specific training results or metadata.
        """
        pass

    @abstractmethod
    def predict(self, data: Any, **kwargs: Any) -> Any:
        """Make predictions.

        Args:
            data: Input data for prediction (DataFrame or Series).
            **kwargs: Additional prediction arguments.

        Returns:
            Prediction result (could be class probabilities, regression value, etc.).
        """
        pass

    @abstractmethod
    def save(
        self, output_path: str | Path, config: dict[str, Any] | None = None
    ) -> None:
        """Save the model and configuration.

        Args:
            output_path: Path where the model will be saved.
            config: Optional configuration dictionary to save with the model.
        """
        pass

    @abstractmethod
    def load(self, model_path: str | Path | None = None) -> None:
        """Load the model from disk.

        Args:
            model_path: Path to the model file.
        """
        pass
