"""CCXT-based data fetcher implementation."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import ccxt.async_support as ccxt
import pandas as pd

from src.ai_strategy.data.fetchers.base import BaseFetcher


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

    async def get_training_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        output_dir: Path = Path("data/raw"),
        output_file: Path | None = None,
    ) -> pd.DataFrame:
        """Get training data with automatic caching.

        If the file exists, load it. Otherwise, download from exchange and save.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT').
            timeframe: Candlestick timeframe (e.g., '1m', '1h', '1d').
            limit: Number of candles to fetch.
            output_dir: Directory to save the data (default: 'data/raw').
            output_file: Optional custom output file path.

        Returns:
            DataFrame with OHLCV data.

        Raises:
            ValueError: If no data is fetched from exchange.
        """
        # Determine output file path
        if output_file is None:
            safe_symbol = symbol.replace("/", "_").lower()
            output_file = output_dir / f"{safe_symbol}_{timeframe}.parquet"

        # Load from cache if exists
        if output_file.exists():
            print(f"Loading data from {output_file}...")
            return pd.read_parquet(output_file)

        # Download from exchange
        print(f"Data file {output_file} not found. Downloading...")

        # Calculate start and end times based on the limit
        end_ts = int(datetime.now().timestamp() * 1000)

        # Dynamic duration calculation
        minutes_per_candle = parse_timeframe_to_minutes(timeframe)
        duration_minutes = limit * minutes_per_candle

        start_ts = int(
            (datetime.now() - timedelta(minutes=duration_minutes)).timestamp() * 1000
        )

        print(f"Fetching {limit} candles for {symbol} ({timeframe})")
        print(
            f"Range: {datetime.fromtimestamp(start_ts/1000)} -> {datetime.fromtimestamp(end_ts/1000)}"
        )

        df = await self.fetch_ohlcv_range(symbol, timeframe, start_ts, end_ts)

        if df.empty:
            raise ValueError("No data fetched from exchange.")

        print(f"Downloaded {len(df)} candles.")
        print(f"First: {pd.to_datetime(df.iloc[0]['timestamp'], unit='ms')}")
        print(f"Last: {pd.to_datetime(df.iloc[-1]['timestamp'], unit='ms')}")

        # Ensure directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to cache
        df.to_parquet(output_file)
        print(f"Saved data to {output_file}")

        return df

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

