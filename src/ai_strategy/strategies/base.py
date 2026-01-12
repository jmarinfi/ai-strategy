from abc import ABC, abstractmethod
from enum import Enum

from src.ai_strategy.data import BaseStreamer
from src.ai_strategy.models import (
    CloseDealWebhook,
    Signal,
    StartDealWebhook,
    Ticker,
    WebhookRequest,
)


class BaseStrategy(ABC):

    def __init__(
        self,
        symbols: list[str],
        streamer: BaseStreamer,
        webhook_url: str,
        bot_uuid: str

    ) -> None:
        self.symbols = symbols
        self.streamer = streamer
        self.webhook_url = webhook_url
        self.bot_uuid = bot_uuid

        self._running: bool = False

    @abstractmethod
    async def on_tick(self, ticker: Ticker) -> None:
        """Process a new price tick from the market.

        Called each time a new ticker update is received from the streamer.
        Implementations should analyze the ticker and potentially generate
        signals or update internal state.

        Args:
            ticker: Current market ticker data.
        """
        ...

    @abstractmethod
    def generate_signal(self, ticker: Ticker) -> Signal:
        """Generate a trading signal based on the ticker data.

        Args:
            ticker: Current market ticker data.

        Returns:
            Signal indicating whether to buy, sell, or hold.
        """
        ...

    async def run(self) -> None:
        """Main loop that processes tickers and generates signals.

        Starts the strategy by connecting to the streamer and processing
        each ticker update through on_tick. Runs until stop() is called.
        """
        self._running = True

        async for ticker_data in self.streamer.watch_tickers():
            if not self._running:
                break

            ticker = self.streamer.parse_ticker(ticker_data)
            await self.on_tick(ticker)

    async def stop(self) -> None:
        """Stop the strategy gracefully.

        Sets the running flag to False, causing the main loop
        in run() to exit after processing the current tick.
        """
        self._running = False

    def _convert_symbol(self, symbol: str) -> str:
        """Convert symbol from exchange format to bot format.

        Args:
            symbol: Symbol in exchange format (e.g., 'BTC/USDT')

        Returns:
            Symbol in bot format (e.g., 'BTC_USDT')
        """
        return symbol.replace("/", "_")

    async def open_position(self, symbol: str) -> None:
        """Open a new position and send webhook to bot.

        The bot will execute the order at market price and track the position.

        Args:
            symbol: Trading pair symbol.
        """
        # Send webhook to bot
        webhook = StartDealWebhook(
            uuid=self.bot_uuid,
            symbol=self._convert_symbol(symbol),
        )
        await self.send_webhook(webhook)

    async def close_position(self, symbol: str) -> None:
        """Close an existing position and send webhook to bot.

        The bot will execute the close order at market price.

        Args:
            symbol: Trading pair symbol.
        """
        # Send webhook to bot
        webhook = CloseDealWebhook(
            uuid=self.bot_uuid,
            symbol=self._convert_symbol(symbol),
        )
        await self.send_webhook(webhook)

    @abstractmethod
    async def send_webhook(self, webhook: WebhookRequest) -> None:
        """Send a webhook to the trading bot.

        Args:
            webhook: Webhook request model to send.
        """
        ...
