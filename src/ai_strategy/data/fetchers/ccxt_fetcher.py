"""CCXT-based data fetcher implementation."""

from typing import Any, Optional

import ccxt.async_support as ccxt
import pandas as pd

from src.ai_strategy.data.fetchers.base import BaseFetcher


class CCXTFetcher(BaseFetcher):
    """CCXT implementation of the data fetcher."""

    def __init__(self, exchange_id: str, sandbox: bool = False):
        """Initialize the CCXT fetcher.

        Args:
            exchange_id: The ccxt exchange identifier (e.g., 'bitget', 'binance').
            sandbox: Whether to use the exchange's sandbox/testnet mode.
        """
        super().__init__(exchange_id, sandbox)
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({"enableRateLimit": True})
        if sandbox:
            self.exchange.set_sandbox_mode(True)

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT').
            timeframe: Candlestick timeframe (e.g., '1m', '1h', '1d').
            since: Start timestamp in milliseconds.
            limit: Maximum number of candles to retrieve.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        candles = await self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        return self._to_dataframe(candles)

    async def fetch_ohlcv_range(
        self, symbol: str, timeframe: str, start_ts: int, end_ts: int
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a specific time range.

        Uses manual pagination to fetch all candles in the range.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT').
            timeframe: Candlestick timeframe (e.g., '1m', '1h', '1d').
            start_ts: Start timestamp in milliseconds.
            end_ts: End timestamp in milliseconds.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        all_candles: list[Any] = []
        current_since = start_ts
        limit = 200  # Bitget's actual limit per request

        while current_since < end_ts:
            candles = await self.exchange.fetch_ohlcv(
                symbol, timeframe, current_since, limit
            )
            
            if not candles:
                break

            # Filter candles within range and add to results
            filtered = [c for c in candles if start_ts <= c[0] <= end_ts]
            all_candles.extend(filtered)

            # Get the last timestamp to advance pagination
            last_ts = candles[-1][0]
            
            # If we've reached or passed the end, stop
            if last_ts >= end_ts:
                break
                
            # Move to next batch (advance past last candle)
            current_since = last_ts + 1

        return self._to_dataframe(all_candles)

    def _to_dataframe(self, candles: list[Any]) -> pd.DataFrame:
        """Convert candles list to DataFrame."""
        return pd.DataFrame(
            candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    async def get_available_symbols(self) -> list[str]:
        """Get list of available trading symbols on the exchange."""
        await self.exchange.load_markets()
        return list(self.exchange.symbols)

    async def close(self) -> None:
        """Close the exchange connection and release resources."""
        await self.exchange.close()
