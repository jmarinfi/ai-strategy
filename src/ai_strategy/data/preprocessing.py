import pandas as pd
import numpy as np
from ta.momentum import AwesomeOscillatorIndicator, StochRSIIndicator, RSIIndicator
from ta.volume import AccDistIndexIndicator, ForceIndexIndicator, MFIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import ADXIndicator, EMAIndicator, IchimokuIndicator, MACD


class Preprocessor:
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate and add comprehensive technical indicators to OHLCV dataframe.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume, timestamp)

        Returns:
            DataFrame with added technical indicators (momentum, volume, volatility, trend)
        """
        print("\n📊 Calculating technical indicators...")
        print(f"   Initial rows: {len(df)}")

        # Sort by timestamp if column exists, otherwise assume data is already ordered
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        # Momentum indicators
        print("   📈 Momentum indicators...")
        ao = AwesomeOscillatorIndicator(df["high"], df["low"])
        df["ao"] = ao.awesome_oscillator()

        stoch_rsi = StochRSIIndicator(df["close"])
        df["stoch_rsi"] = stoch_rsi.stochrsi()
        df["stoch_rsi_d"] = stoch_rsi.stochrsi_d()
        df["stoch_rsi_k"] = stoch_rsi.stochrsi_k()

        rsi = RSIIndicator(df["close"])
        df["rsi"] = rsi.rsi()

        # Volume indicators
        print("   📊 Volume indicators...")
        adi = AccDistIndexIndicator(df["high"], df["low"], df["close"], df["volume"])
        df["adi"] = adi.acc_dist_index()

        fi = ForceIndexIndicator(df["close"], df["volume"])
        df["fi"] = fi.force_index()

        mfi = MFIIndicator(df["high"], df["low"], df["close"], df["volume"])
        df["mfi"] = mfi.money_flow_index()

        # Volatility indicators
        print("   📉 Volatility indicators...")
        atr = AverageTrueRange(df["high"], df["low"], df["close"])
        df["atr"] = atr.average_true_range()

        bb = BollingerBands(df["close"])
        df["b_hband"] = bb.bollinger_hband()
        df["b_hband_ind"] = bb.bollinger_hband_indicator()
        df["b_lband"] = bb.bollinger_lband()
        df["b_lband_ind"] = bb.bollinger_lband_indicator()
        df["b_mavg"] = bb.bollinger_mavg()
        df["b_pband"] = bb.bollinger_pband()
        df["b_wband"] = bb.bollinger_wband()

        # Trend indicators
        print("   📐 Trend indicators...")
        adx = ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx.adx()
        df["adx_neg"] = adx.adx_neg()
        df["adx_pos"] = adx.adx_pos()

        ema = EMAIndicator(df["close"])
        df["ema"] = ema.ema_indicator()

        ichimoku = IchimokuIndicator(df["high"], df["low"])
        df["ichimoku_a"] = ichimoku.ichimoku_a()
        df["ichimoku_b"] = ichimoku.ichimoku_b()
        df["ichimoku_base_line"] = ichimoku.ichimoku_base_line()
        df["ichimoku_conversion_line"] = ichimoku.ichimoku_conversion_line()

        macd = MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_diff"] = macd.macd_diff()
        df["macd_signal"] = macd.macd_signal()

        print(f"   ✅ Added {len(df.columns) - 6} technical indicators")

        return df.dropna(ignore_index=True)

    @staticmethod
    def parse_timeframe_to_minutes(timeframe: str) -> int:
        """Convert timeframe string (e.g. '15m', '1h', '4h', '1d') to minutes."""
        unit = timeframe[-1]
        value = int(timeframe[:-1])

        if unit == "m":
            return value
        elif unit == "h":
            return value * 60
        elif unit == "d":
            return value * 24 * 60
        elif unit == "w":
            return value * 24 * 60 * 7
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

    @staticmethod
    def split_sequences(sequences: np.ndarray, n_steps: int):
        X, y = [], []

        for i in range(len(sequences)):
            end_ix = i + n_steps
            if end_ix > len(sequences) - 1:
                break
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix, -1]
            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y)

    @staticmethod
    def series_to_supervised(
        data: pd.DataFrame, n_in: int = 1, n_out: int = 1, dropnan: bool = True
    ):
        """Convert a time series to a supervised learning dataset.

        Args:
            data: DataFrame with time series data
            n_in: Number of previous time steps to use as input features
            n_out: Number of future time steps to predict
            dropnan: Whether to drop rows with NaN values

        Returns:
            DataFrame with supervised learning data
        """
        # Eliminar columnas "timestamp", "open", "high", "low", "volume"
        if "timestamp" in data.columns:
            data = data.drop("timestamp", axis=1)
        if "open" in data.columns:
            data = data.drop("open", axis=1)
        if "high" in data.columns:
            data = data.drop("high", axis=1)
        if "low" in data.columns:
            data = data.drop("low", axis=1)
        if "volume" in data.columns:
            data = data.drop("volume", axis=1)
        print("\nDataframe without timestamp, open, high, low, volume:")
        print(data.head())

        # Reorganizar columnas con "close" al final
        cols = [col for col in data.columns if col != "close"] + ["close"]
        data = data[cols]
        print("\nDataframe with close column at the end:")
        print(data.head())

        # Transformar a array numpy
        values = data.values
        print("\nArray numpy:")
        print(values)
        print(values.shape)
