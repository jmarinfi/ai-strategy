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
        long_bot_uuid: str | None = None,
        short_bot_uuid: str | None = None,
        timeframe: str = "5m",
    ) -> None:
        self.symbols = symbols
        self.streamer = streamer
        self.webhook_url = webhook_url
        self.long_bot_uuid = long_bot_uuid
        self.short_bot_uuid = short_bot_uuid
        self.timeframe = timeframe

        # At least one bot UUID must be provided
        if not long_bot_uuid and not short_bot_uuid:
            raise ValueError("At least one of long_bot_uuid or short_bot_uuid must be provided")

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

        # Use the first symbol in the list
        symbol = self.symbols[0] if self.symbols else None
        if not symbol:
            raise ValueError("No symbols provided to strategy")

        async for ohlcv_data in self.streamer.watch_ohlcv(symbol, self.timeframe):
            if not self._running:
                break

            # Create Candle from OHLCV data: [timestamp, open, high, low, close, volume]
            candle = Candle(
                symbol=symbol,
                timeframe=self.timeframe,
                timestamp=int(ohlcv_data[0]),
                open=float(ohlcv_data[1]),
                high=float(ohlcv_data[2]),
                low=float(ohlcv_data[3]),
                close=float(ohlcv_data[4]),
                volume=float(ohlcv_data[5]),
            )
            
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

    async def open_position(self, symbol: str, position_type: str = "long") -> None:
        """Open a new position and send webhook to bot.

        Args:
            symbol: Trading pair symbol.
            position_type: "long" or "short" (default: "long").
        """
        position_type = position_type.lower()
        
        if position_type == "long":
            if not self.long_bot_uuid:
                print(f"   ⚠️  Skipping LONG position - no long_bot_uuid configured")
                return
            uuid = self.long_bot_uuid
            print(f"\n📤 Opening LONG position for {symbol}...")
        elif position_type == "short":
            if not self.short_bot_uuid:
                print(f"   ⚠️  Skipping SHORT position - no short_bot_uuid configured")
                return
            uuid = self.short_bot_uuid
            print(f"\n📤 Opening SHORT position for {symbol}...")
        else:
            raise ValueError(f"Invalid position_type: {position_type}. Must be 'long' or 'short'")

        # Send webhook to bot
        webhook = StartDealWebhook(
            uuid=uuid,
            symbol=self._convert_symbol(symbol),
        )
        success = await self.send_webhook(webhook)
        if success:
            print(f"   ✅ {position_type.upper()} position opened successfully")
        else:
            print(f"   ❌ Failed to open {position_type.upper()} position")

    async def close_position(self, symbol: str, position_type: str = "long") -> None:
        """Close an existing position and send webhook to bot.

        Note: This method is kept for backward compatibility but should
        generally not be used as the bot manages position closing.

        Args:
            symbol: Trading pair symbol.
            position_type: "long" or "short" (default: "long").
        """
        position_type = position_type.lower()
        
        if position_type == "long":
            if not self.long_bot_uuid:
                return
            uuid = self.long_bot_uuid
        elif position_type == "short":
            if not self.short_bot_uuid:
                return
            uuid = self.short_bot_uuid
        else:
            raise ValueError(f"Invalid position_type: {position_type}. Must be 'long' or 'short'")

        print(f"\n📤 Closing {position_type.upper()} position for {symbol}...")
        # Send webhook to bot
        webhook = CloseDealWebhook(
            uuid=uuid,
            symbol=self._convert_symbol(symbol),
        )
        success = await self.send_webhook(webhook)
        if success:
            print(f"   ✅ {position_type.upper()} position closed successfully")
        else:
            print(f"   ❌ Failed to close {position_type.upper()} position")

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

        print(f"   🌐 Sending webhook to {self.webhook_url}")
        print(f"   📦 Payload: {payload}")

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(self.webhook_url, json=payload)

                print(f"   📡 Response status: {response.status_code}")
                if response.status_code == 200:
                    print(f"   📄 Response body: {response.text}")
                    return True
                else:
                    print(f"   ⚠️  Non-200 response: {response.text}")
                    return False

        except Exception as e:
            print(f"   ❌ Webhook error: {e}")
            return False
