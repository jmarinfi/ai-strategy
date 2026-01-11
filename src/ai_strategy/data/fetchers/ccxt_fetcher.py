"""CCXT-based data fetcher implementation."""

import asyncio
from typing import Any, Optional

import ccxt.async_support as ccxt
import pandas as pd

from src.ai_strategy.data.fetchers.base import BaseFetcher


class CCXTFetcher(BaseFetcher):
    """CCXT implementation of the data fetcher.

    This fetcher uses the ccxt library to connect to cryptocurrency exchanges
    and retrieve OHLCV (candlestick) data.

    Note:
        The API parameters used in fetch_ohlcv and fetch_ohlcv_range are currently
        configured specifically for **Bitget**. When adding support for other exchanges,
        these parameters will need to be adapted accordingly:
        - `uta`: Bitget-specific parameter (unified trading account).
        - `paginate`: ccxt pagination feature, behavior may vary per exchange.
        - `until`: Bitget-specific parameter for end timestamp in range queries.
    """

    def __init__(self, exchange_id: str, sandbox: bool = False):
        """Initialize the CCXT fetcher.

        Args:
            exchange_id: The ccxt exchange identifier (e.g., 'bitget', 'binance').
            sandbox: Whether to use the exchange's sandbox/testnet mode.
        """
        super().__init__(exchange_id, sandbox)
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(
            {
                "enableRateLimit": True,
            }
        )
        if sandbox:
            self.exchange.set_sandbox_mode(True)
        self.symbols: list[str] = []

    async def _ensure_markets_loaded(self) -> None:
        """Ensure markets are loaded before operations."""
        if not self.symbols:
            await self.exchange.load_markets()
            self.symbols = self.exchange.symbols or []

    def _get_bitget_params(
        self, paginate: bool = True, until: Optional[int] = None
    ) -> dict:
        """Get Bitget-specific parameters.

        Args:
            paginate: Whether to enable ccxt automatic pagination.
            until: End timestamp for range queries (Bitget-specific).

        Returns:
            Dictionary of exchange-specific parameters.

        Note:
            These parameters are Bitget-specific. For other exchanges,
            this method should be extended or overridden.
        """
        params: dict = {"uta": False, "paginate": paginate}
        if until is not None:
            params["until"] = until
        return params

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data with automatic pagination.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT').
            timeframe: Candlestick timeframe (e.g., '1m', '1h', '1d').
            since: Start timestamp in milliseconds.
            limit: Maximum number of candles to retrieve.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.

        Raises:
            ValueError: If symbol is not found or exchange doesn't support fetchOHLCV.

        Note:
            The params `uta=False` and `paginate=True` are Bitget-specific.
            For other exchanges, these parameters may need to be modified.
        """
        if not self.exchange.has["fetchOHLCV"]:
            raise ValueError("Exchange does not support fetchOHLCV")

        await self._ensure_markets_loaded()

        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not found on {self.exchange_id}")

        params = self._get_bitget_params(paginate=True)
        candles = await self.exchange.fetch_ohlcv(
            symbol, timeframe, since, limit, params=params
        )
        return self._to_dataframe(candles)

    async def fetch_ohlcv_range(
        self, symbol: str, timeframe: str, start_ts: int, end_ts: int
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a specific time range with manual pagination.

        This method implements manual pagination to ensure all candles in the
        specified range are fetched, regardless of exchange limits.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT').
            timeframe: Candlestick timeframe (e.g., '1m', '1h', '1d').
            start: Start timestamp in milliseconds.
            end: End timestamp in milliseconds.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.

        Raises:
            ValueError: If symbol is not found or exchange doesn't support fetchOHLCV.

        Note:
            Uses manual pagination loop because Bitget's `paginate=True` returns
            empty results when combined with `until` parameter.
            The `uta=False` and `until` params are Bitget-specific.
        """
        if not self.exchange.has["fetchOHLCV"]:
            raise ValueError("Exchange does not support fetchOHLCV")

        await self._ensure_markets_loaded()

        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not found on {self.exchange_id}")

        all_candles: list = []
        current_since = start_ts
        batch_limit = 1000  # Safe limit for most exchanges

        while current_since < end_ts:
            params = self._get_bitget_params(paginate=False, until=end_ts)
            candles: list[Any] = await self.exchange.fetch_ohlcv(
                symbol, timeframe, current_since, batch_limit, params=params
            )

            if not candles:
                break

            # Filter candles within the range
            candles = [c for c in candles if c[0] <= end_ts]
            if not candles:
                break

            all_candles.extend(candles)

            # Move to next batch
            last_timestamp = candles[-1][0]
            if len(candles) < batch_limit or last_timestamp >= end_ts:
                break

            # Advance past the last candle to avoid duplicates
            current_since = last_timestamp + 1

            # Respect rate limits
            await asyncio.sleep(self.exchange.rateLimit / 1000)

        # Remove duplicates based on timestamp
        seen_timestamps: set[int] = set()
        unique_candles = []
        for candle in all_candles:
            if candle[0] not in seen_timestamps:
                unique_candles.append(candle)
                seen_timestamps.add(candle[0])

        return self._to_dataframe(unique_candles)

    def _to_dataframe(self, candles: list[Any]) -> pd.DataFrame:
        """Convert candles list to DataFrame.

        Args:
            candles: List of OHLCV candles from ccxt.

        Returns:
            DataFrame with standard OHLCV columns.
        """
        return pd.DataFrame(
            data=candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

    async def get_available_symbols(self) -> list[str]:
        """Get list of available trading symbols on the exchange.

        Returns:
            List of symbol strings (e.g., ['BTC/USDT', 'ETH/USDT']).
        """
        await self._ensure_markets_loaded()
        return self.symbols

    async def close(self) -> None:
        """Close the exchange connection and release resources."""
        await self.exchange.close()
