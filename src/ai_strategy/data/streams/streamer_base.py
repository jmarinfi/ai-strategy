"""Base streamer interface for real-time market data."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


class BaseStreamer(ABC):
    """Abstract base class for real-time market data streaming.

    A single streamer instance handles multiple symbols to minimize
    the number of open WebSocket connections.

    Attributes:
        exchange_id: The ID of the exchange (e.g., 'bitget').
        symbols: List of trading pair symbols to stream.
        sandbox: Whether to use the exchange's testnet/sandbox.
    """

    def __init__(
        self, exchange_id: str, symbols: list[str], sandbox: bool = False
    ) -> None:
        """Initialize the streamer.

        Args:
            exchange_id: The unique identifier for the exchange.
            symbols: List of trading pair symbols (e.g., ['BTC/USDT', 'ETH/USDT']).
            sandbox: If True, uses the exchange's testnet environment.
        """
        self.exchange_id = exchange_id
        self.symbols = symbols
        self.sandbox = sandbox

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the exchange WebSocket.

        This method should be called before starting to watch tickers.
        It may load markets and prepare the connection.
        """
        pass

    @abstractmethod
    def watch_tickers(self) -> AsyncGenerator[dict[str, Any], None]:
        """Watch real-time ticker updates for all configured symbols.

        Yields ticker data as it arrives from the exchange. Each ticker
        contains price and volume information for a single symbol.

        Yields:
            dict containing ticker data with at least:
                - symbol: The trading pair symbol
                - last: Last traded price
                - bid: Best bid price
                - ask: Best ask price
                - high: 24h high price
                - low: 24h low price
                - volume: 24h volume
                - timestamp: Update timestamp in milliseconds

        Raises:
            ConnectionError: If not connected or connection is lost.
        """
        # Implementations should use `async def` with `yield`
        # to create an async generator
        ...

    @abstractmethod
    async def unsubscribe(self, symbols: list[str] | None = None) -> None:
        """Unsubscribe from ticker updates for specified symbols.

        Args:
            symbols: List of symbols to unsubscribe from.
                     If None, unsubscribes from all symbols.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the WebSocket connection and release resources."""
        pass

    async def __aenter__(self) -> "BaseStreamer":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()
