from pathlib import Path
from typing import Any
from datetime import datetime
import csv

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
import joblib


class RandomForestMLM:
    def __init__(
        self,
        data: pd.DataFrame,
        n_lags_in: int,
        n_lags_out: int,
        prediction_horizon: int,
        name_targets: list[str],
        swing_length: int,
        model=RandomForestRegressor(verbose=1, n_jobs=-1),
    ):
        self.data = data
        self.n_lags_in = n_lags_in
        self.n_lags_out = n_lags_out
        self.prediction_horizon = prediction_horizon
        self.name_targets = name_targets
        self.swing_length = swing_length
        self.model = model

    def _series_to_supervised_multivariate(
        self, data: pd.DataFrame, dropnan: bool = True
    ) -> np.ndarray:
        cols = []
        df_input_cols = data.iloc[:, : -len(self.name_targets)]
        df_output_cols = data.iloc[:, -len(self.name_targets) :]

        for i in range(self.n_lags_in, 0, -1):
            # Crear copia del lote desplazado para evitar look-ahead bias
            shifted_input = df_input_cols.shift(i).copy()

            # Poner a 0 los últimos swing_length registros de la columna "HighLow"
            if "HighLow" in shifted_input.columns:
                shifted_input.iloc[
                    -self.swing_length :, shifted_input.columns.get_loc("HighLow")
                ] = 0

            # Poner a 0 el último registro de la columna "fvg_signal"
            if "fvg_signal" in shifted_input.columns:
                shifted_input.iloc[-1, shifted_input.columns.get_loc("fvg_signal")] = 0

            cols.append(shifted_input)

        for i in range(self.n_lags_out):
            cols.append(df_output_cols.shift(-self.prediction_horizon - i))

        agg = pd.concat(cols, axis=1)
        if dropnan:
            agg.dropna(inplace=True)

        return agg.values

    def _train_test_split(
        self, data: np.ndarray, train_size: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_train = int(len(data) * train_size)
        train, test = data[:n_train], data[n_train:]
        X_train, y_train = (
            train[:, : -len(self.name_targets)],
            train[:, -len(self.name_targets) :],
        )
        X_test, y_test = (
            test[:, : -len(self.name_targets)],
            test[:, -len(self.name_targets) :],
        )
        return X_train, y_train, X_test, y_test

    def prepare_data_for_training(
        self, train_size: float = 0.8
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print("\n📊 PREPARING DATA FOR TRAINING")
        print("=" * 30)

        columns = [
            c
            for c in self.data.columns
            if c not in ["timestamp", "open", "high", "low"]
        ] + self.name_targets
        series = self.data[columns]

        print(f"Original DataFrame shape: {self.data.shape}")
        print(f"Features: {columns[: -len(self.name_targets)]}")
        print(f"Targets: {self.name_targets}")

        data = self._series_to_supervised_multivariate(series)
        print(f"\nTransformed Data shape (supervised): {data.shape}")

        X_train, y_train, X_test, y_test = self._train_test_split(data, train_size)

        print("\nDATA SPLIT SUMMARY")
        print("-" * 20)
        print(f"Train Set: X={X_train.shape}, y={y_train.shape} ({train_size:.0%})")
        print(f"Test Set:  X={X_test.shape}, y={y_test.shape} ({1 - train_size:.0%})")
        print("=" * 30 + "\n")

        return X_train, y_train, X_test, y_test

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        self.model.fit(X_train, y_train)
        print("\n✅ Model trained successfully!")

        score = self.model.score(X_test, y_test)
        print("\n✅ Model score: ", score)

        predicted_test = self.model.predict(X_test)
        print("\nPredicted test: ", predicted_test)
        print("\nPredicted test shape: ", predicted_test.shape)

        # metrics
        mae = mean_absolute_error(y_test, predicted_test)
        mape = mean_absolute_percentage_error(y_test, predicted_test)
        mse = mean_squared_error(y_test, predicted_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predicted_test)
        print("\nMetrics: ")
        print("MAE: ", mae)
        print("MAPE: ", mape)
        print("MSE: ", mse)
        print("RMSE: ", rmse)
        print("R2: ", r2)

    def walk_forward_validation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        block_size: int,
        results_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        history_X = X_train.tolist()
        history_y = y_train.tolist()
        predictions = []

        # Configurar CSV si se solicita
        csv_file = None
        writer = None
        if results_path:
            results_path = Path(results_path)
            results_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = open(results_path, mode="w", newline="")
            writer = csv.writer(csv_file)

            # Cabeceras
            headers = ["iteration", "timestamp", "train_r2"]
            for target in self.name_targets:
                headers.extend(
                    [f"expected_{target}", f"predicted_{target}", f"error_{target}"]
                )
            writer.writerow(headers)

        print(f"\n🚶 Iniciando Walk-Forward Validation ({len(X_test)} pasos)...")

        for i in range(0, len(X_test), block_size):
            current_test_X = X_test[i : i + block_size]
            current_test_y = y_test[i : i + block_size]

            # Re-entrenar con historial actualizado
            X_hist_arr = np.array(history_X)
            y_hist_arr = np.array(history_y)
            self.model.fit(X_hist_arr, y_hist_arr)

            # Predicción
            yhat = self.model.predict(current_test_X)
            predictions.extend(yhat)

            # Guardar métricas de la iteración en CSV y Logs
            train_r2 = 0.0
            if writer:
                train_r2 = self.model.score(X_hist_arr, y_hist_arr)

            for j in range(len(current_test_X)):
                idx_global = i + j
                real_val = current_test_y[j]
                pred_val = yhat[j]

                # CSV Logging
                if writer:
                    row = [idx_global + 1, datetime.now().isoformat(), train_r2]

                    real_vals = np.atleast_1d(real_val)
                    pred_vals = np.atleast_1d(pred_val)

                    for real, pred in zip(real_vals, pred_vals):
                        row.extend([real, pred, real - pred])

                    writer.writerow(row)
                    csv_file.flush()

                # Console Log
                print(
                    f"> Step {idx_global + 1}: Expected={real_val}, Predicted={pred_val}"
                )

            # Actualizar historial
            history_X.extend(current_test_X.tolist())
            history_y.extend(current_test_y.tolist())

            if output_path:
                self.save(output_path)

        if csv_file:
            csv_file.close()
            print(f"\n✅ Resultados detallados guardados en {results_path}")

        # metrics globales
        predictions = np.array(predictions)
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        print("\n📊 WALK-FORWARD VALIDATION METRICS")
        print("=" * 30)
        print(f"MAE:  {mae:.6f}")
        print(f"MAPE: {mape:.2%}")
        print(f"MSE:  {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R2:   {r2:.6f}")
        print("=" * 30 + "\n")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        print("\nPredicting...")
        print("X_test shape: ", X_test.shape)
        return self.model.predict(X_test)

    def save(
        self, output_path: str | Path, config: dict[str, Any] | None = None
    ) -> None:
        model_data = {
            "model": self.model,
            "feature_cols": self.data.columns[: -len(self.name_targets)],
            "target_cols": self.name_targets,
            "n_lags_in": self.n_lags_in,
            "n_lags_out": self.n_lags_out,
            "prediction_horizon": self.prediction_horizon,
            "timestamp": datetime.now().isoformat(),
        }
        print("\nModel data: ")
        print(model_data)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model_data, output_file)
        print(f"\n✅ Model saved to {output_file}")

    def load(self, model_path: str | Path | None = None) -> None:
        pass
