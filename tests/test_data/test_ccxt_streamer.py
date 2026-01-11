"""Tests for CCXTStreamer with mocked WebSocket functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ai_strategy.data.streams.ccxt_streamer import CCXTStreamer


@pytest.fixture
def mock_exchange():
    """Create a mock exchange with common setup."""
    exchange = MagicMock()
    exchange.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    exchange.load_markets = AsyncMock()
    exchange.watch_tickers = AsyncMock()
    exchange.un_watch_ticker = AsyncMock()
    exchange.close = AsyncMock()
    exchange.set_sandbox_mode = MagicMock()
    return exchange


def generate_ticker(symbol: str, last_price: float = 50000.0) -> dict:
    """Generate a mock ticker for testing.

    Args:
        symbol: Trading pair symbol.
        last_price: Last traded price.

    Returns:
        Mock ticker dictionary.
    """
    return {
        "symbol": symbol,
        "timestamp": 1700000000000,
        "datetime": "2023-11-14T22:13:20.000Z",
        "high": last_price * 1.05,
        "low": last_price * 0.95,
        "bid": last_price - 10,
        "bidVolume": 1.5,
        "ask": last_price + 10,
        "askVolume": 2.0,
        "open": last_price * 0.98,
        "close": last_price,
        "last": last_price,
        "change": last_price * 0.02,
        "percentage": 2.0,
        "baseVolume": 1000.0,
        "quoteVolume": 50000000.0,
        "info": {"raw": "data"},
    }


class TestCCXTStreamerInit:
    """Tests for CCXTStreamer initialization."""

    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    def test_init_creates_exchange(self, mock_ccxtpro):
        """Test that __init__ creates exchange instance."""
        mock_exchange_class = MagicMock()
        mock_ccxtpro.bitget = mock_exchange_class

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])

        mock_exchange_class.assert_called_once_with({"enableRateLimit": True})
        assert streamer.exchange_id == "bitget"
        assert streamer.symbols == ["BTC/USDT"]
        assert streamer.sandbox is False
        assert streamer._connected is False

    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    def test_init_with_sandbox_mode(self, mock_ccxtpro):
        """Test that sandbox mode is enabled when specified."""
        mock_exchange = MagicMock()
        mock_exchange_class = MagicMock(return_value=mock_exchange)
        mock_ccxtpro.bitget = mock_exchange_class

        streamer = CCXTStreamer("bitget", ["BTC/USDT"], sandbox=True)

        mock_exchange.set_sandbox_mode.assert_called_once_with(True)
        assert streamer.sandbox is True

    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    def test_init_with_multiple_symbols(self, mock_ccxtpro):
        """Test initialization with multiple symbols."""
        mock_exchange_class = MagicMock()
        mock_ccxtpro.bitget = mock_exchange_class

        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        streamer = CCXTStreamer("bitget", symbols)

        assert streamer.symbols == symbols


class TestConnect:
    """Tests for connect method."""

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_connect_loads_markets(self, mock_ccxtpro, mock_exchange):
        """Test that connect loads markets."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange

        await streamer.connect()

        mock_exchange.load_markets.assert_called_once()
        assert streamer._connected is True

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_connect_validates_symbols(self, mock_ccxtpro, mock_exchange):
        """Test that connect validates all symbols exist."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)
        mock_exchange.symbols = ["BTC/USDT"]  # Only BTC available

        streamer = CCXTStreamer("bitget", ["BTC/USDT", "INVALID/USDT"])
        streamer.exchange = mock_exchange

        with pytest.raises(ValueError, match="Symbol INVALID/USDT not found"):
            await streamer.connect()

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_connect_skips_if_already_connected(
        self, mock_ccxtpro, mock_exchange
    ):
        """Test that connect doesn't reload markets if already connected."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange
        streamer._connected = True

        await streamer.connect()

        mock_exchange.load_markets.assert_not_called()


class TestWatchTickers:
    """Tests for watch_tickers method."""

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_watch_tickers_raises_if_not_connected(
        self, mock_ccxtpro, mock_exchange
    ):
        """Test that watch_tickers raises ConnectionError if not connected."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange

        with pytest.raises(ConnectionError, match="not connected"):
            async for _ in streamer.watch_tickers():
                pass

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_watch_tickers_yields_normalized_tickers(
        self, mock_ccxtpro, mock_exchange
    ):
        """Test that watch_tickers yields properly normalized tickers."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        ticker = generate_ticker("BTC/USDT", 50000.0)
        mock_exchange.watch_tickers.return_value = {"BTC/USDT": ticker}

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange
        streamer._connected = True

        received_tickers = []
        async for t in streamer.watch_tickers():
            received_tickers.append(t)
            if len(received_tickers) >= 1:
                break

        assert len(received_tickers) == 1
        assert received_tickers[0]["symbol"] == "BTC/USDT"
        assert received_tickers[0]["last"] == 50000.0

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_watch_tickers_filters_unwanted_symbols(
        self, mock_ccxtpro, mock_exchange
    ):
        """Test that watch_tickers only yields tickers for requested symbols."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        # Exchange returns extra symbol we didn't request
        mock_exchange.watch_tickers.return_value = {
            "BTC/USDT": generate_ticker("BTC/USDT"),
            "XRP/USDT": generate_ticker("XRP/USDT"),  # Not requested
        }

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange
        streamer._connected = True

        received_tickers = []
        async for t in streamer.watch_tickers():
            received_tickers.append(t)
            if len(received_tickers) >= 1:
                break

        # Should only get BTC/USDT
        assert all(t["symbol"] == "BTC/USDT" for t in received_tickers)

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_watch_tickers_uses_bitget_params(self, mock_ccxtpro, mock_exchange):
        """Test that watch_tickers uses Bitget-specific params."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)
        mock_exchange.watch_tickers.return_value = {
            "BTC/USDT": generate_ticker("BTC/USDT")
        }

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange
        streamer._connected = True

        async for _ in streamer.watch_tickers():
            break

        mock_exchange.watch_tickers.assert_called_with(["BTC/USDT"], {"uta": False})

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_watch_tickers_yields_multiple_symbols(
        self, mock_ccxtpro, mock_exchange
    ):
        """Test that watch_tickers yields tickers for all requested symbols."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        mock_exchange.watch_tickers.return_value = {
            "BTC/USDT": generate_ticker("BTC/USDT", 50000.0),
            "ETH/USDT": generate_ticker("ETH/USDT", 3000.0),
        }

        streamer = CCXTStreamer("bitget", ["BTC/USDT", "ETH/USDT"])
        streamer.exchange = mock_exchange
        streamer._connected = True

        received_tickers = []
        async for t in streamer.watch_tickers():
            received_tickers.append(t)
            if len(received_tickers) >= 2:
                break

        symbols = {t["symbol"] for t in received_tickers}
        assert "BTC/USDT" in symbols
        assert "ETH/USDT" in symbols


class TestReconnection:
    """Tests for automatic reconnection behavior."""

    @pytest.mark.asyncio
    @patch(
        "src.ai_strategy.data.streams.ccxt_streamer.asyncio.sleep",
        new_callable=AsyncMock,
    )
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_watch_tickers_reconnects_on_error(
        self, mock_ccxtpro, mock_sleep, mock_exchange
    ):
        """Test that watch_tickers reconnects after an error."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        # First call fails, second succeeds
        mock_exchange.watch_tickers.side_effect = [
            Exception("Connection lost"),
            {"BTC/USDT": generate_ticker("BTC/USDT")},
        ]

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange
        streamer._connected = True

        received_tickers = []
        async for t in streamer.watch_tickers():
            received_tickers.append(t)
            if len(received_tickers) >= 1:
                break

        # Should have reconnected and received ticker
        assert len(received_tickers) == 1
        assert mock_sleep.called
        assert mock_exchange.close.called
        assert mock_exchange.load_markets.called

    @pytest.mark.asyncio
    @patch(
        "src.ai_strategy.data.streams.ccxt_streamer.asyncio.sleep",
        new_callable=AsyncMock,
    )
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_watch_tickers_exponential_backoff(
        self, mock_ccxtpro, mock_sleep, mock_exchange
    ):
        """Test that reconnection uses exponential backoff."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        # Fail 3 times, then succeed
        mock_exchange.watch_tickers.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            Exception("Error 3"),
            {"BTC/USDT": generate_ticker("BTC/USDT")},
        ]

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange
        streamer._connected = True
        streamer._base_delay = 1.0

        async for _ in streamer.watch_tickers():
            break

        # Check exponential backoff: 1, 2, 4 seconds
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert sleep_calls == [1.0, 2.0, 4.0]

    @pytest.mark.asyncio
    @patch(
        "src.ai_strategy.data.streams.ccxt_streamer.asyncio.sleep",
        new_callable=AsyncMock,
    )
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_watch_tickers_max_delay_cap(
        self, mock_ccxtpro, mock_sleep, mock_exchange
    ):
        """Test that backoff delay is capped at max_delay."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        # Fail many times, then succeed
        failures = [Exception(f"Error {i}") for i in range(10)]
        mock_exchange.watch_tickers.side_effect = [
            *failures,
            {"BTC/USDT": generate_ticker("BTC/USDT")},
        ]

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange
        streamer._connected = True
        streamer._base_delay = 1.0
        streamer._max_delay = 8.0  # Cap at 8 seconds for testing

        async for _ in streamer.watch_tickers():
            break

        # Delays should be: 1, 2, 4, 8, 8, 8, 8, 8, 8, 8 (capped at 8)
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert all(d <= 8.0 for d in sleep_calls)
        assert sleep_calls[-1] == 8.0

    @pytest.mark.asyncio
    @patch(
        "src.ai_strategy.data.streams.ccxt_streamer.asyncio.sleep",
        new_callable=AsyncMock,
    )
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_watch_tickers_resets_retry_counter_on_success(
        self, mock_ccxtpro, mock_sleep, mock_exchange
    ):
        """Test that retry counter resets after successful fetch."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        # Fail once, succeed twice, fail once, succeed
        mock_exchange.watch_tickers.side_effect = [
            Exception("Error 1"),
            {"BTC/USDT": generate_ticker("BTC/USDT")},
            {"BTC/USDT": generate_ticker("BTC/USDT")},
            Exception("Error 2"),
            {"BTC/USDT": generate_ticker("BTC/USDT")},
        ]

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange
        streamer._connected = True
        streamer._base_delay = 1.0

        count = 0
        async for _ in streamer.watch_tickers():
            count += 1
            if count >= 3:
                break

        # Both errors should have used base_delay (1.0) since counter resets
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert sleep_calls == [1.0, 1.0]

    @pytest.mark.asyncio
    @patch(
        "src.ai_strategy.data.streams.ccxt_streamer.asyncio.sleep",
        new_callable=AsyncMock,
    )
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_watch_tickers_raises_after_max_retries(
        self, mock_ccxtpro, mock_sleep, mock_exchange
    ):
        """Test that watch_tickers raises after max retries exceeded."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        # Always fail
        mock_exchange.watch_tickers.side_effect = Exception("Persistent error")

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange
        streamer._connected = True
        streamer._max_retries = 3

        with pytest.raises(ConnectionError, match="Failed to recover after 3 retries"):
            async for _ in streamer.watch_tickers():
                pass

    @pytest.mark.asyncio
    @patch(
        "src.ai_strategy.data.streams.ccxt_streamer.asyncio.sleep",
        new_callable=AsyncMock,
    )
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_watch_tickers_handles_close_error_during_reconnect(
        self, mock_ccxtpro, mock_sleep, mock_exchange
    ):
        """Test that errors during close are ignored during reconnection."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        mock_exchange.watch_tickers.side_effect = [
            Exception("Connection error"),
            {"BTC/USDT": generate_ticker("BTC/USDT")},
        ]
        mock_exchange.close.side_effect = Exception("Close failed")

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange
        streamer._connected = True

        # Should not raise despite close failing
        async for _ in streamer.watch_tickers():
            break

        assert mock_exchange.close.called


class TestNormalizeTicker:
    """Tests for _normalize_ticker method."""

    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    def test_normalize_ticker_extracts_all_fields(self, mock_ccxtpro):
        """Test that _normalize_ticker extracts all expected fields."""
        mock_ccxtpro.bitget = MagicMock()

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        ticker = generate_ticker("BTC/USDT", 50000.0)

        result = streamer._normalize_ticker(ticker)

        assert result["symbol"] == "BTC/USDT"
        assert result["timestamp"] == 1700000000000
        assert result["datetime"] == "2023-11-14T22:13:20.000Z"
        assert result["high"] == 50000.0 * 1.05
        assert result["low"] == 50000.0 * 0.95
        assert result["bid"] == 49990.0
        assert result["ask"] == 50010.0
        assert result["last"] == 50000.0
        assert result["baseVolume"] == 1000.0
        assert result["info"] == {"raw": "data"}

    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    def test_normalize_ticker_handles_missing_fields(self, mock_ccxtpro):
        """Test that _normalize_ticker handles missing fields gracefully."""
        mock_ccxtpro.bitget = MagicMock()

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        ticker = {"symbol": "BTC/USDT", "last": 50000.0}  # Minimal ticker

        result = streamer._normalize_ticker(ticker)

        assert result["symbol"] == "BTC/USDT"
        assert result["last"] == 50000.0
        assert result["high"] is None
        assert result["low"] is None
        assert result["bid"] is None


class TestUnsubscribe:
    """Tests for unsubscribe method."""

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_unsubscribe_specific_symbols(self, mock_ccxtpro, mock_exchange):
        """Test unsubscribing from specific symbols."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        streamer = CCXTStreamer("bitget", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        streamer.exchange = mock_exchange

        await streamer.unsubscribe(["ETH/USDT"])

        mock_exchange.un_watch_ticker.assert_called_once_with(
            "ETH/USDT", {"uta": False}
        )
        assert "ETH/USDT" not in streamer.symbols
        assert "BTC/USDT" in streamer.symbols
        assert "SOL/USDT" in streamer.symbols

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_unsubscribe_all_symbols(self, mock_ccxtpro, mock_exchange):
        """Test unsubscribing from all symbols when None is passed."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        streamer = CCXTStreamer("bitget", ["BTC/USDT", "ETH/USDT"])
        streamer.exchange = mock_exchange

        await streamer.unsubscribe(None)

        assert mock_exchange.un_watch_ticker.call_count == 2
        assert len(streamer.symbols) == 0

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_unsubscribe_ignores_errors(self, mock_ccxtpro, mock_exchange):
        """Test that unsubscribe errors are silently ignored."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)
        mock_exchange.un_watch_ticker.side_effect = Exception("Unsubscribe failed")

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange

        # Should not raise
        await streamer.unsubscribe(["BTC/USDT"])


class TestClose:
    """Tests for close method."""

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_close_closes_exchange(self, mock_ccxtpro, mock_exchange):
        """Test that close properly closes the exchange connection."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange
        streamer._connected = True

        await streamer.close()

        mock_exchange.close.assert_called_once()
        assert streamer._connected is False


class TestContextManager:
    """Tests for async context manager functionality."""

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_context_manager_connects_and_closes(
        self, mock_ccxtpro, mock_exchange
    ):
        """Test that async with connects on entry and closes on exit."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange

        async with streamer:
            assert streamer._connected is True
            mock_exchange.load_markets.assert_called_once()

        mock_exchange.close.assert_called_once()
        assert streamer._connected is False

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.streams.ccxt_streamer.ccxtpro")
    async def test_context_manager_closes_on_exception(
        self, mock_ccxtpro, mock_exchange
    ):
        """Test that async with closes even when exception occurs."""
        mock_ccxtpro.bitget = MagicMock(return_value=mock_exchange)

        streamer = CCXTStreamer("bitget", ["BTC/USDT"])
        streamer.exchange = mock_exchange

        with pytest.raises(ValueError):
            async with streamer:
                raise ValueError("Test error")

        mock_exchange.close.assert_called_once()
