from abc import ABC, abstractmethod

from src.ai_strategy.data import BaseStreamer
from src.ai_strategy.models import (
    Candle,
    CloseDealWebhook,
    Signal,
    StartDealWebhook,
    WebhookRequest,
)


class BaseStrategy(ABC):
    def __init__(
        self,
        symbols: list[str],
        streamer: BaseStreamer,
        webhook_url: str,
        bot_uuid: str,
    ) -> None:
        self.symbols = symbols
        self.streamer = streamer
        self.webhook_url = webhook_url
        self.bot_uuid = bot_uuid

        self._running: bool = False

    @abstractmethod
    async def on_candle(self, candle: Candle) -> None:
        """Process a new candlestick from the market.

        Called each time a new candle is received from the streamer.
        Implementations should analyze the candle and potentially generate
        signals or update internal state.

        Args:
            candle: Current market candle data (OHLCV).
        """
        ...

    @abstractmethod
    def generate_signal(self, candle: Candle) -> Signal:
        """Generate a trading signal based on the candle data.

        Args:
            candle: Current market candle data (OHLCV).

        Returns:
            Signal indicating whether to buy, sell, or hold.
        """
        ...

    async def run(self) -> None:
        """Main loop that processes candles and generates signals.

        Starts the strategy by connecting to the streamer and processing
        each candle through on_candle. Runs until stop() is called.
        """
        self._running = True

        async for candle_data in self.streamer.watch_ohlcv():
            if not self._running:
                break

            candle = self.streamer.parse_candle(candle_data)
            await self.on_candle(candle)

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

    async def send_webhook(self, webhook: WebhookRequest) -> bool:
        """Send a webhook to the trading bot.

        Args:
            webhook: Webhook request model to send.

        Returns:
            True if successful, False otherwise.
        """
        import httpx

        payload = {
            "action": webhook.action,
            "uuid": webhook.uuid,
        }

        # Add symbol if present (for StartDealWebhook and CloseDealWebhook)
        if hasattr(webhook, "symbol"):
            payload["symbol"] = webhook.symbol

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(self.webhook_url, json=payload)

                if response.status_code == 200:
                    return True
                else:
                    return False

        except Exception:
            return False
