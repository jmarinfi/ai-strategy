"""Random Forest-based trading strategy implementation."""

from pathlib import Path
from datetime import datetime

import pandas as pd
from joblib import load

from src.ai_strategy.data import BaseStreamer, CCXTFetcher, Preprocessor
from src.ai_strategy.models import Candle, Signal, SignalType
from src.ai_strategy.strategies.base import BaseStrategy


class RandomForestStrategy(BaseStrategy):
    """Trading strategy using Random Forest model for price predictions.

    This strategy:
    1. Processes real-time candlestick data
    2. Calculates technical indicators
    3. Uses Random Forest to predict next candle's close price
    4. Generates BUY/SELL signals based on predicted price movement
    """

    def __init__(
        self,
        symbols: list[str],
        streamer: BaseStreamer,
        webhook_url: str,
        model_path: str | Path,
        timeframe: str,
        n_lags: int,  # Number of lag steps used in training
        long_bot_uuid: str | None = None,
        short_bot_uuid: str | None = None,
        exchange: str = "bitget",
        price_threshold: float = 0.001,  # 0.1% minimum price movement
        historical_candles: int = 60,
    ) -> None:
        """Initialize the Random Forest strategy.

        Args:
            symbols: List of trading pair symbols.
            streamer: Data streamer for real-time market data.
            webhook_url: URL for sending webhooks.
            model_path: Path to the trained Random Forest model file.
            timeframe: Candle timeframe (e.g., '5m', '15m', '1h').
            n_lags: Number of lag steps used in model training.
            long_bot_uuid: UUID of the LONG bot (optional).
            short_bot_uuid: UUID of the SHORT bot (optional).
            exchange: Exchange name (default: 'bitget').
            price_threshold: Minimum price movement % to trigger signal (default: 0.001 = 0.1%).
            historical_candles: Number of historical candles to fetch (default: 60).
        """
        super().__init__(
            symbols, streamer, webhook_url, long_bot_uuid, short_bot_uuid, timeframe
        )

        self.exchange = exchange
        self.price_threshold = price_threshold
        self.historical_candles = historical_candles
        self.n_lags = n_lags

        # Load the Random Forest model
        self.model = load(model_path)["model"]
        print(f"✅ Random Forest model loaded from {model_path}")

        # Track last processed candle
        self._last_candle_timestamp: int | None = None
        self._previous_candle: Candle | None = None

    async def on_candle(self, candle: Candle) -> None:
        """Process a new candlestick from the market.

        Args:
            candle: Current market candle data (OHLCV).
        """
        current_timestamp = candle.timestamp
        dt_str = datetime.fromtimestamp(current_timestamp / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Log every candle update (in-progress candle)
        print(f"📥 Candle update: {dt_str} | Close: {candle.close:.6f}", end="\r")

        # Detect new candle (timestamp changed)
        if current_timestamp != self._last_candle_timestamp:
            # Process the PREVIOUS candle (which is now complete)
            if self._previous_candle is not None:
                await self._process_complete_candle(self._previous_candle)

            # Update tracking
            self._last_candle_timestamp = current_timestamp

        # Always update previous_candle with the latest data
        self._previous_candle = candle

    async def _process_complete_candle(self, candle: Candle) -> None:
        """Process a complete candle and make trading decisions.

        Args:
            candle: Complete candle to process.
        """
        dt_str = datetime.fromtimestamp(candle.timestamp / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        print(f"\n{'=' * 80}")
        print(f"🕯️  PROCESSING COMPLETE CANDLE: {dt_str}")
        print(f"{'=' * 80}")
        print(f"   O: {candle.open:.6f}  H: {candle.high:.6f}")
        print(f"   L: {candle.low:.6f}   C: {candle.close:.6f}")
        print(f"   V: {candle.volume:.4f}")

        try:
            # 1. Fetch historical candles for indicators
            print("\n📊 Fetching historical data...")
            historical_df = await self._fetch_historical_data(candle)

            if historical_df is None or len(historical_df) < self.historical_candles:
                print(
                    f"   ⚠️  Not enough data ({len(historical_df) if historical_df is not None else 0} candles)"
                )
                return

            print(f"   ✅ Fetched {len(historical_df)} historical candles")

            # 2. Calculate technical indicators
            print("📈 Calculating technical indicators...")
            df_with_indicators = Preprocessor.add_technical_indicators(historical_df)
            columns = df_with_indicators.columns
            df_with_indicators = df_with_indicators[
                [col for col in columns if col != "timestamp"]
            ]

            if len(df_with_indicators) == 0:
                print("   ⚠️  No data after indicator calculation")
                return

            features_window = df_with_indicators[-96:]
            print(f"Longitud del window: {len(features_window)}")
            X_pred = features_window.values.flatten().reshape(1, -1)
            print(f"Shape de X_pred: {X_pred.shape}")

            # 3. Make prediction
            print("🤖 Making prediction...")
            predicted_close = self.model.predict(X_pred)[0]
            current_close = candle.close

            # Calculate predicted movement
            price_change = predicted_close - current_close
            price_change_pct = (price_change / current_close) * 100

            print("\n💡 Model Prediction:")
            print(f"   Current Close:    {current_close:.6f}")
            print(f"   Predicted Close:  {predicted_close:.6f}")
            print(
                f"   Expected Change:  {price_change:+.6f} ({price_change_pct:+.2f}%)"
            )
            print(f"   Threshold:        ±{self.price_threshold * 100:.2f}%")

            # 4. Generate signal
            signal = self.generate_signal(candle, predicted_close)

            print(f"\n🎯 Signal Generated: {signal.signal_type}")
            if signal.reason:
                print(f"   Reason: {signal.reason}")

            # 5. Execute signal if actionable
            if signal.is_actionable:
                print("   ✅ Signal is actionable, executing...")
                await self._execute_signal(signal)
            else:
                print("   ⏸️  Signal is NOT actionable (HOLD)")

            print(f"{'=' * 80}\n")

        except Exception as e:
            # Log error but don't crash the strategy
            print(f"\n❌ Error processing candle: {e}")
            import traceback

            traceback.print_exc()
            print(f"{'=' * 80}\n")

    async def _fetch_historical_data(self, candle: Candle) -> pd.DataFrame | None:
        """Fetch historical candles needed for indicator calculation.

        Args:
            candle: Reference candle for end timestamp.

        Returns:
            DataFrame with historical OHLCV data, or None if fetch failed.
        """
        try:
            fetcher = CCXTFetcher(self.exchange, sandbox=False)

            end_ts = candle.timestamp
            start_ts = end_ts - (
                60
                * 1000
                * Preprocessor.parse_timeframe_to_minutes(self.timeframe)
                * self.historical_candles
            )

            historical_df = await fetcher.fetch_ohlcv_range(
                candle.symbol, self.timeframe, start_ts, end_ts
            )
            await fetcher.close()

            return historical_df

        except Exception:
            return None

    def generate_signal(
        self, candle: Candle, predicted_close: float | None = None
    ) -> Signal:
        """Generate a trading signal based on predicted price movement.

        Args:
            candle: Current market candle data (OHLCV).
            predicted_close: Predicted close price for next candle.

        Returns:
            Signal indicating whether to buy, sell, or hold.
        """
        if predicted_close is None:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=candle.symbol,
                price=candle.close,
                timestamp=candle.datetime_,
            )

        current_close = candle.close
        price_change_pct = (predicted_close - current_close) / current_close

        # Generate signal based on predicted movement
        if price_change_pct > self.price_threshold:
            # Price expected to go UP
            return Signal(
                signal_type=SignalType.BUY,
                symbol=candle.symbol,
                price=candle.close,
                confidence=abs(price_change_pct),
                model_prediction=predicted_close,
                reason=f"Predicted price increase: {current_close:.6f} → {predicted_close:.6f} ({price_change_pct * 100:+.2f}%)",
            )
        elif price_change_pct < -self.price_threshold:
            # Price expected to go DOWN
            return Signal(
                signal_type=SignalType.SELL,
                symbol=candle.symbol,
                price=candle.close,
                confidence=abs(price_change_pct),
                model_prediction=predicted_close,
                reason=f"Predicted price decrease: {current_close:.6f} → {predicted_close:.6f} ({price_change_pct * 100:+.2f}%)",
            )
        else:
            # Movement too small
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=candle.symbol,
                price=candle.close,
                confidence=abs(price_change_pct),
                model_prediction=predicted_close,
                reason=f"Predicted movement too small ({price_change_pct * 100:+.2f}% < ±{self.price_threshold * 100:.2f}%)",
            )

    async def _execute_signal(self, signal: Signal) -> None:
        """Execute a trading signal by sending webhooks.

        Args:
            signal: Signal to execute.
        """
        if signal.signal_type == SignalType.BUY:
            # BUY signal = Open LONG position
            await self.open_position(signal.symbol, position_type="long")
        elif signal.signal_type == SignalType.SELL:
            # SELL signal = Open SHORT position
            await self.open_position(signal.symbol, position_type="short")
        # HOLD does nothing
