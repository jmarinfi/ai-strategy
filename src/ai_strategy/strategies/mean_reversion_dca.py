from datetime import datetime

import numpy as np
import pandas as pd
from ta.volatility import BollingerBands

from src.ai_strategy.models.candle import Candle
from src.ai_strategy.models.signal import Signal
from src.ai_strategy.strategies.base import BaseStrategy
from src.ai_strategy.data import BaseStreamer
from src.ai_strategy.data.fetchers.ccxt_fetcher import CCXTFetcher
from src.ai_strategy.data.preprocessing import Preprocessor


class MeanReversionDCA(BaseStrategy):
    def __init__(
        self,
        symbols: list[str],
        streamer: BaseStreamer,
        webhook_url: str,
        long_bot_uuid: str,
        short_bot_uuid: str,
        timeframe: str,
        window_bb: int,
        window_dev_bb: int,
        max_orders_dca: int,
        min_pct_objective: float,
    ):
        super().__init__(
            symbols=symbols,
            streamer=streamer,
            webhook_url=webhook_url,
            long_bot_uuid=long_bot_uuid,
            short_bot_uuid=short_bot_uuid,
            timeframe=timeframe,
        )
        self.window_bb = window_bb
        self.window_dev_bb = window_dev_bb
        self.max_orders_dca = max_orders_dca
        self.min_pct_objective = min_pct_objective

        self._last_candle_timestamp: int | None = None
        self._historical: pd.DataFrame | None = None
        self._last_bb_high: float | None = None
        self._last_bb_low: float | None = None
        self._last_bb_mid: float | None = None
        self._is_long_position: bool = False
        self._is_short_position: bool = False
        self._num_dca_long: int = 0
        self._num_dca_short: int = 0
        self._last_long_price: float | None = None
        self._last_short_price: float | None = None

    async def on_candle(self, candle: Candle) -> None:
        current_timestamp = candle.timestamp
        dt_str = datetime.fromtimestamp(current_timestamp / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        print(f"📥 Candle update: {dt_str} | Close: {candle.close}")

        await self._process_candle(candle)

    async def _fetch_historical_data(self, candle: Candle) -> pd.DataFrame | None:
        try:
            fetcher = CCXTFetcher(exchange_id="bitget", sandbox=False)

            end_ts = candle.timestamp
            start_ts = (
                end_ts
                - (self.window_bb + 1)
                * Preprocessor.parse_timeframe_to_minutes(self.timeframe)
                * 60
                * 1000
            )
            historical_df = await fetcher.fetch_ohlcv_range(
                candle.symbol, self.timeframe, start_ts, end_ts
            )
            await fetcher.close()

            return historical_df

        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None

    def _update_bb_indicators(self, close: float) -> None:
        all_closes = np.append(self._historical["close"].values, close)
        closes = all_closes[-self.window_bb :]

        mavg = np.mean(closes)
        mstd = np.std(closes)  # ddof=0 by default, same as ta library

        self._last_bb_mid = mavg
        self._last_bb_high = mavg + self.window_dev_bb * mstd
        self._last_bb_low = mavg - self.window_dev_bb * mstd

    async def _process_candle(self, candle: Candle) -> None:
        try:
            # Obtener histórico para calcular los indicadores si detectamos nueva candle y añadir indicadores
            if (
                candle.timestamp != self._last_candle_timestamp
                or self._historical is None
            ):
                self._historical = await self._fetch_historical_data(candle)
                self._last_candle_timestamp = candle.timestamp
                if self._historical is None or len(self._historical) < self.window_bb:
                    print("Not enough data to process candle")
                    return

                bb = BollingerBands(
                    close=self._historical["close"],
                    window=self.window_bb,
                    window_dev=self.window_dev_bb,
                )
                self._historical["bb_hband"] = bb.bollinger_hband()
                self._historical["bb_lband"] = bb.bollinger_lband()
                self._historical["bb_mavg"] = bb.bollinger_mavg()

                self._last_bb_high = self._historical["bb_hband"].iloc[-1]
                self._last_bb_low = self._historical["bb_lband"].iloc[-1]
                self._last_bb_mid = self._historical["bb_mavg"].iloc[-1]

            if (
                self._last_bb_high is not None
                and self._last_bb_low is not None
                and self._last_bb_mid is not None
            ):
                self._update_bb_indicators(close=candle.close)
                print(f"BB High: {self._last_bb_high}")
                print(f"BB Low: {self._last_bb_low}")
                print(f"BB Mid: {self._last_bb_mid}")
                print(f"Close: {candle.close}")
                await self.generate_signal(candle)

        except Exception as e:
            print(f"Error processing candle: {e}")

    async def generate_signal(self, candle: Candle) -> Signal | None:
        # Definir entrada DCA en long
        if (
            self._is_long_position
            and self._num_dca_long < self.max_orders_dca
            and candle.close < self._last_long_price * (1 - self.min_pct_objective)
            and candle.close < self._last_bb_low
        ):
            self._num_dca_long += 1
            self._last_long_price = candle.close
            print(f"DCA en long num {self._num_dca_long}")

            await self.add_funds_quote(
                symbol=candle.symbol,
                position_type="long",
                perc=1 / (self.max_orders_dca * 2 + 2),
            )
            return

        # Definir entrada DCA en short
        if (
            self._is_short_position
            and self._num_dca_short < self.max_orders_dca
            and candle.close > self._last_short_price * (1 + self.min_pct_objective)
            and candle.close > self._last_bb_high
        ):
            self._num_dca_short += 1
            self._last_short_price = candle.close
            print(f"DCA en short num {self._num_dca_short}")

            await self.add_funds_quote(
                symbol=candle.symbol,
                position_type="short",
                perc=1 / (self.max_orders_dca * 2 + 2),
            )
            return

        # Definir salida long
        if self._is_long_position and candle.close > self._last_bb_mid:
            self._is_long_position = False
            self._num_dca_long = 0
            self._last_long_price = None
            print("Salida en long")

            await self.close_position(symbol=candle.symbol, position_type="long")
            return

        # Definir salida short
        if self._is_short_position and candle.close < self._last_bb_mid:
            self._is_short_position = False
            self._num_dca_long = 0
            self._last_short_price = None
            print("Salida en short")

            await self.close_position(symbol=candle.symbol, position_type="short")
            return

        pct_close_to_bb_mid = (candle.close - self._last_bb_mid) / self._last_bb_mid

        # Definir entrada en long
        if (
            not self._is_long_position
            and self._last_bb_low is not None
            and candle.close < self._last_bb_low
        ):
            if abs(pct_close_to_bb_mid) < self.min_pct_objective:
                print("No entra en long porque está demasiado cerca del objetivo.")
                return

            print("Entrada en long")
            self._is_long_position = True
            self._last_long_price = candle.close

            await self.open_position(symbol=candle.symbol, position_type="long")
            return

        # Definir entrada en short
        if (
            not self._is_short_position
            and self._last_bb_high is not None
            and candle.close > self._last_bb_high
        ):
            if abs(pct_close_to_bb_mid) < self.min_pct_objective:
                print("No entra en short porque está demasiado cerca del objetivo.")
                return

            print("Entrada en short")
            self._is_short_position = True
            self._last_short_price = candle.close

            await self.open_position(symbol=candle.symbol, position_type="short")
            return
