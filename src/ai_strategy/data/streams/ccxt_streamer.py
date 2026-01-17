"""CCXT-based streamer implementation for real-time market data."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import ccxt.pro as ccxtpro

from src.ai_strategy.data.streams.streamer_base import BaseStreamer
from src.ai_strategy.models import Ticker

logger = logging.getLogger(__name__)


class CCXTStreamer(BaseStreamer):
    """CCXT Pro implementation of the real-time data streamer.

    This streamer uses ccxt.pro WebSocket functionality to receive
    real-time ticker updates from cryptocurrency exchanges.

    Note:
        Currently configured for Bitget. The `uta=False` parameter is
        Bitget-specific (disables unified trading account mode).
        For other exchanges, this may need adjustment.
    """

    def __init__(
        self, exchange_id: str, symbols: list[str], sandbox: bool = False
    ) -> None:
        """Initialize the CCXT streamer.

        Args:
            exchange_id: The ccxt exchange identifier (e.g., 'bitget').
            symbols: List of trading pair symbols (e.g., ['BTC/USDT', 'ETH/USDT']).
            sandbox: Whether to use the exchange's sandbox/testnet mode.
        """
        super().__init__(exchange_id, symbols, sandbox)
        exchange_class = getattr(ccxtpro, exchange_id)
        self.exchange: Any = exchange_class(
            {
                "enableRateLimit": True,
            }
        )
        if sandbox:
            self.exchange.set_sandbox_mode(True)
        self._connected: bool = False
        self._max_retries: int = 100
        self._base_delay: float = 1.0  # Initial delay in seconds
        self._max_delay: float = 60.0  # Maximum delay in seconds

    async def connect(self) -> None:
        """Establish connection by loading markets.

        This prepares the exchange for WebSocket operations by
        loading available markets and symbols.
        """
        if not self._connected:
            await self.exchange.load_markets()
            # Validate that all requested symbols exist
            for symbol in self.symbols:
                if symbol not in self.exchange.symbols:
                    raise ValueError(f"Symbol {symbol} not found on {self.exchange_id}")
            self._connected = True

    async def watch_tickers(self) -> AsyncGenerator[dict[str, Any], None]:
        """Watch real-time ticker updates for all configured symbols.

        Continuously polls for ticker updates and yields them as they arrive.
        Each call to watch_tickers on the exchange returns the latest ticker(s).

        Yields:
            dict containing ticker data with fields:
                - symbol: The trading pair symbol
                - last: Last traded price
                - bid: Best bid price
                - ask: Best ask price
                - high: 24h high price
                - low: 24h low price
                - baseVolume: 24h base volume
                - quoteVolume: 24h quote volume
                - timestamp: Update timestamp in milliseconds
                - datetime: ISO8601 datetime string
                - info: Raw exchange response

        Raises:
            ConnectionError: If not connected.
        """
        if not self._connected:
            raise ConnectionError(
                "Streamer not connected. Call connect() or use async with."
            )

        params = {"uta": False}  # Bitget-specific: disable unified trading account

        retries = 0
        delay = self._base_delay

        while True:
            try:
                # watch_tickers returns a dict of tickers keyed by symbol
                tickers = await self.exchange.watch_tickers(self.symbols, params)

                # Reset retry counter on successful fetch
                retries = 0
                delay = self._base_delay

                # Yield each ticker individually
                for symbol, ticker in tickers.items():
                    if symbol in self.symbols:
                        yield ticker

            except Exception as e:
                retries += 1
                error_msg = str(e).lower()

                if retries > self._max_retries:
                    logger.error(
                        f"Max retries ({self._max_retries}) exceeded. Giving up."
                    )
                    raise ConnectionError(
                        f"Failed to recover after {self._max_retries} retries: {error_msg}"
                    ) from e

                logger.warning(
                    f"Error in watch_tickers (attempt {retries}/{self._max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )

                await asyncio.sleep(delay)

                # Exponential backoff with cap
                delay = min(delay * 2, self._max_delay)

                try:
                    self._connected = False
                    await self.exchange.close()
                except Exception:
                     pass  # Ignore errors when closing broken connection
                await self.connect()

    async def watch_ohlcv(
        self, symbol: str, timeframe: str = "1m"
    ) -> AsyncGenerator[list[Any], None]:
        """Watch real-time OHLCV (candlestick) updates for a specific symbol.

        Continuously streams candlestick data as new candles are formed or updated.

        Args:
            symbol: The trading pair symbol (e.g., 'BTC/USDT').
            timeframe: Candlestick time interval (e.g., '1m', '5m', '15m', '1h', '4h', '1d').

        Yields:
            list containing OHLCV data: [timestamp, open, high, low, close, volume]

        Raises:
            ConnectionError: If not connected.
            ValueError: If timeframe is not supported.

        Example:
            async with CCXTStreamer("bitget", ["BTC/USDT"]) as streamer:
                async for candle in streamer.watch_ohlcv("BTC/USDT", "5m"):
                    print(candle)  # [timestamp, o, h, l, c, v]
        """
        if not self._connected:
            raise ConnectionError(
                "Streamer not connected. Call connect() or use async with."
            )

        # Validate symbol
        if symbol not in self.symbols:
            raise ValueError(
                f"Symbol {symbol} not in configured symbols: {self.symbols}"
            )

        params = {"uta": False}  # Bitget-specific: disable unified trading account

        retries = 0
        delay = self._base_delay

        while True:
            try:
                # watch_ohlcv returns an ArrayCache of candles
                ohlcv = await self.exchange.watch_ohlcv(symbol, timeframe, params=params)

                # Reset retry counter on successful fetch
                retries = 0
                delay = self._base_delay

                # Yield the most recent candle
                # ohlcv is an ArrayCache, get the last element
                if len(ohlcv) > 0:
                    yield ohlcv[-1]

            except Exception as e:
                retries += 1
                error_msg = str(e).lower()

                if retries > self._max_retries:
                    logger.error(
                        f"Max retries ({self._max_retries}) exceeded. Giving up."
                    )
                    raise ConnectionError(
                        f"Failed to recover after {self._max_retries} retries: {error_msg}"
                    ) from e

                logger.warning(
                    f"Error in watch_ohlcv (attempt {retries}/{self._max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )

                await asyncio.sleep(delay)

                # Exponential backoff with cap
                delay = min(delay * 2, self._max_delay)

                try:
                    self._connected = False
                    await self.exchange.close()
                except Exception:
                    pass  # Ignore errors when closing broken connection
                await self.connect()

    def parse_ticker(self, raw_ticker: dict[str, Any]) -> Ticker:

        """Parse raw CCXT ticker data into a Ticker model.

        Args:
            raw_ticker: Raw ticker dictionary from CCXT.

        Returns:
            Ticker model with standardized fields.
        """
        return Ticker.model_validate(raw_ticker)

    async def unsubscribe(self, symbols: list[str] | None = None) -> None:
        """Unsubscribe from ticker updates for specified symbols.

        Args:
            symbols: List of symbols to unsubscribe from.
                     If None, unsubscribes from all symbols.

        Note:
            Uses Bitget-specific un_watch_ticker method.
            For other exchanges, this may need adjustment.
        """
        targets = symbols if symbols is not None else self.symbols.copy()

        for symbol in targets:
            try:
                await self.exchange.un_watch_ticker(symbol, {"uta": False})
                if symbol in self.symbols:
                    self.symbols.remove(symbol)
            except Exception:
                # Some exchanges may not support unsubscribe, ignore errors
                pass

    async def un_watch_ohlcv(self, symbol: str, timeframe: str = "1m") -> None:
        """Unsubscribe from OHLCV updates for a specific symbol and timeframe.

        Args:
            symbol: The trading pair symbol to unsubscribe from.
            timeframe: The candlestick timeframe to unsubscribe from.

        Note:
            Uses Bitget-specific un_watch_ohlcv method.
            For other exchanges, this may need adjustment.
        """
        try:
            await self.exchange.un_watch_ohlcv(symbol, timeframe, {"uta": False})
        except Exception as e:
            # Some exchanges may not support unsubscribe, log but don't raise
            logger.warning(f"Failed to unsubscribe from OHLCV for {symbol} {timeframe}: {e}")

    async def close(self) -> None:

        """Close the WebSocket connection and release resources."""
        self._connected = False
        await self.exchange.close()
