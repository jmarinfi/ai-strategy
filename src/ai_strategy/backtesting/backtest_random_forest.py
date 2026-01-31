from datetime import datetime

from backtesting import Strategy
from joblib import load

from src.ai_strategy.data import CCXTFetcher, Preprocessor


class RandomForestStrategyBacktest(Strategy):
    price_pct_threshold = 0.02
    minutes_stop = 15 * 4 * 4 * 60
    take_profit_pct = 0.005
    stop_loss_pct = -0.003

    def init(self):
        self.fetcher = CCXTFetcher(exchange_id="bitget")
        self.model = load("models/random_forest_ada_usdt_15m_lags96.pkl")["model"]

    def next(self):
        # Skip if not enough data for indicator calculation
        if len(self.data.df) < 150:
            return

        # Time-based stop: Close positions open for more than n_candles_stop + 1 candles
        if self.position:
            # Get entry timestamp from trade tag
            for trade in self.trades:
                # Check if trade is still open (exit_bar is None means trade is open)
                if trade.exit_bar is None and trade.tag:
                    try:
                        entry_time = datetime.strptime(trade.tag, "%Y-%m-%d %H:%M:%S")
                        current_time = self.data.index[-1]

                        # Calculate time difference in candles (assuming 15min candles)
                        time_diff = (
                            current_time - entry_time
                        ).total_seconds() / 60  # minutes

                        if time_diff >= self.minutes_stop:
                            trade.close()

                    except (ValueError, AttributeError):
                        pass

        # Get last 150 candles for indicator calculation
        recent_data = self.data.df.iloc[-150:].copy()

        # Convert backtesting.py format (uppercase) to expected format (lowercase)
        historical_df = recent_data.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        historical_df = Preprocessor.add_technical_indicators(historical_df)

        if len(historical_df) == 0:
            print("   ⚠️  No data after indicator calculation")
            return

        if len(historical_df) < 96:
            return

        features_window = historical_df.iloc[-96:]
        X_pred = features_window.values.flatten().reshape(1, -1)

        predicted_close = self.model.predict(X_pred)
        current_close = self.data.Close[-1]
        print("Predicted close: ", predicted_close)
        print("Current close: ", current_close)

        price_change = predicted_close - current_close
        price_change_pct = (price_change / current_close) * 100

        current_timestamp = self.data.index[-1].strftime("%Y-%m-%d %H:%M:%S")

        if price_change_pct > self.price_pct_threshold and not (
            self.position and self.position.is_long
        ):
            self.buy(
                sl=current_close + self.stop_loss_pct * current_close,
                tp=current_close + self.take_profit_pct * current_close,
                tag=current_timestamp,
            )
        elif price_change_pct < -self.price_pct_threshold and not (
            self.position and self.position.is_short
        ):
            self.sell(
                sl=current_close - self.stop_loss_pct * current_close,
                tp=current_close - self.take_profit_pct * current_close,
                tag=current_timestamp,
            )
