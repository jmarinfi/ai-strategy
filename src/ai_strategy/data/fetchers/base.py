from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class BaseFetcher(ABC):
    """Abstract base class for cryptocurrency data fetchers.

    Attributes:
        exchange_id (str): The ID of the exchange (e.g., 'binance').
        sandbox (bool): Whether to use the exchange's testnet/sandbox.
    """

    def __init__(self, exchange_id: str, sandbox: bool = False):
        """Initialize the fetcher.

        Args:
            exchange_id: The unique identifier for the exchange.
            sandbox: If True, uses the exchange's testnet environment.
        """
        self.exchange_id = exchange_id
        self.sandbox = sandbox

    @abstractmethod
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for a specific symbol.

        Args:
            symbol: The trading pair symbol (e.g., 'BTC/USDT').
            timeframe: The candlestick timeframe (e.g., '1h', '1d').
            since: The start timestamp in milliseconds.
            limit: The maximum number of candles to return. If None, uses exchange default.

        Returns:
            pd.DataFrame: A DataFrame containing 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
        """
        pass

    @abstractmethod
    async def fetch_ohlcv_range(
        self, symbol: str, timeframe: str, start_ts: int, end_ts: int
    ) -> pd.DataFrame:
        """Fetch OHLCV data covering a specific time range.

        Implementations should handle pagination/multiple requests if the range
        exceeds the exchange's limit per request.

        Args:
            symbol: The trading pair symbol.
            timeframe: The candlestick timeframe.
            start_ts: The start timestamp in milliseconds (inclusive).
            end_ts: The end timestamp in milliseconds (inclusive).

        Returns:
            pd.DataFrame: A DataFrame with the complete dataset for the range.
        """
        pass

    @abstractmethod
    async def get_available_symbols(self) -> list[str]:
        """Get a list of all available trading symbols on the exchange.

        Returns:
            list[str]: A list of symbol strings (e.g., ['BTC/USDT', 'ETH/USDT']).
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by the fetcher (e.g., HTTP sessions)."""
        pass
