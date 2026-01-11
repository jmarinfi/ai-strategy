"""Tests for CCXTFetcher with mocked pagination."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.ai_strategy.data.fetchers.ccxt_fetcher import CCXTFetcher


@pytest.fixture
def mock_exchange():
    """Create a mock exchange with common setup."""
    exchange = MagicMock()
    exchange.has = {"fetchOHLCV": True}
    exchange.symbols = ["BTC/USDT", "ETH/USDT"]
    exchange.rateLimit = 100
    exchange.load_markets = AsyncMock()
    exchange.fetch_ohlcv = AsyncMock()
    exchange.close = AsyncMock()
    exchange.set_sandbox_mode = MagicMock()
    return exchange


def generate_candles(start_ts: int, count: int, interval_ms: int = 60000) -> list:
    """Generate mock OHLCV candles.

    Args:
        start_ts: Starting timestamp in milliseconds.
        count: Number of candles to generate.
        interval_ms: Time interval between candles (default 1 minute).

    Returns:
        List of [timestamp, open, high, low, close, volume] candles.
    """
    candles = []
    for i in range(count):
        ts = start_ts + (i * interval_ms)
        price = 50000 + i
        candles.append([ts, price, price + 10, price - 10, price + 5, 1.0])
    return candles


class TestCCXTFetcherInit:
    """Tests for CCXTFetcher initialization."""

    @patch("src.ai_strategy.data.fetchers.ccxt_fetcher.ccxt")
    def test_init_creates_exchange(self, mock_ccxt):
        """Test that __init__ creates exchange instance."""
        mock_exchange_class = MagicMock()
        mock_ccxt.bitget = mock_exchange_class

        fetcher = CCXTFetcher("bitget", sandbox=False)

        mock_exchange_class.assert_called_once_with({"enableRateLimit": True})
        assert fetcher.exchange_id == "bitget"
        assert fetcher.sandbox is False

    @patch("src.ai_strategy.data.fetchers.ccxt_fetcher.ccxt")
    def test_init_with_sandbox_mode(self, mock_ccxt):
        """Test that sandbox mode is enabled when specified."""
        mock_exchange = MagicMock()
        mock_exchange_class = MagicMock(return_value=mock_exchange)
        mock_ccxt.bitget = mock_exchange_class

        fetcher = CCXTFetcher("bitget", sandbox=True)

        mock_exchange.set_sandbox_mode.assert_called_once_with(True)
        assert fetcher.sandbox is True


class TestFetchOHLCV:
    """Tests for fetch_ohlcv method."""

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.fetchers.ccxt_fetcher.ccxt")
    async def test_fetch_ohlcv_returns_dataframe(self, mock_ccxt, mock_exchange):
        """Test that fetch_ohlcv returns a properly formatted DataFrame."""
        mock_ccxt.bitget = MagicMock(return_value=mock_exchange)
        candles = generate_candles(1000000, 100)
        mock_exchange.fetch_ohlcv.return_value = candles

        fetcher = CCXTFetcher("bitget")
        fetcher.exchange = mock_exchange
        fetcher.symbols = ["BTC/USDT"]

        result = await fetcher.fetch_ohlcv("BTC/USDT", "1m", 1000000, limit=100)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert list(result.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.fetchers.ccxt_fetcher.ccxt")
    async def test_fetch_ohlcv_calls_with_bitget_params(self, mock_ccxt, mock_exchange):
        """Test that fetch_ohlcv uses Bitget-specific params."""
        mock_ccxt.bitget = MagicMock(return_value=mock_exchange)
        mock_exchange.fetch_ohlcv.return_value = generate_candles(1000000, 10)

        fetcher = CCXTFetcher("bitget")
        fetcher.exchange = mock_exchange
        fetcher.symbols = ["BTC/USDT"]

        await fetcher.fetch_ohlcv("BTC/USDT", "1m", 1000000, limit=10)

        mock_exchange.fetch_ohlcv.assert_called_once_with(
            "BTC/USDT", "1m", 1000000, 10, params={"uta": False, "paginate": True}
        )

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.fetchers.ccxt_fetcher.ccxt")
    async def test_fetch_ohlcv_raises_on_unknown_symbol(self, mock_ccxt, mock_exchange):
        """Test that fetch_ohlcv raises ValueError for unknown symbols."""
        mock_ccxt.bitget = MagicMock(return_value=mock_exchange)

        fetcher = CCXTFetcher("bitget")
        fetcher.exchange = mock_exchange
        fetcher.symbols = ["BTC/USDT"]

        with pytest.raises(ValueError, match="Symbol UNKNOWN/USDT not found"):
            await fetcher.fetch_ohlcv("UNKNOWN/USDT", "1m", 1000000, limit=10)

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.fetchers.ccxt_fetcher.ccxt")
    async def test_fetch_ohlcv_raises_when_not_supported(
        self, mock_ccxt, mock_exchange
    ):
        """Test that fetch_ohlcv raises ValueError when exchange doesn't support it."""
        mock_exchange.has = {"fetchOHLCV": False}
        mock_ccxt.bitget = MagicMock(return_value=mock_exchange)

        fetcher = CCXTFetcher("bitget")
        fetcher.exchange = mock_exchange
        fetcher.symbols = ["BTC/USDT"]

        with pytest.raises(ValueError, match="does not support fetchOHLCV"):
            await fetcher.fetch_ohlcv("BTC/USDT", "1m", 1000000, limit=10)


class TestFetchOHLCVRange:
    """Tests for fetch_ohlcv_range method with pagination."""

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.fetchers.ccxt_fetcher.ccxt")
    async def test_fetch_ohlcv_range_single_batch(self, mock_ccxt, mock_exchange):
        """Test fetch_ohlcv_range with data fitting in one batch."""
        mock_ccxt.bitget = MagicMock(return_value=mock_exchange)
        start_ts = 1000000
        end_ts = start_ts + (100 * 60000)  # 100 minutes
        candles = generate_candles(start_ts, 100)
        mock_exchange.fetch_ohlcv.return_value = candles

        fetcher = CCXTFetcher("bitget")
        fetcher.exchange = mock_exchange
        fetcher.symbols = ["BTC/USDT"]

        result = await fetcher.fetch_ohlcv_range("BTC/USDT", "1m", start_ts, end_ts)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100

    @pytest.mark.asyncio
    @patch(
        "src.ai_strategy.data.fetchers.ccxt_fetcher.asyncio.sleep",
        new_callable=AsyncMock,
    )
    @patch("src.ai_strategy.data.fetchers.ccxt_fetcher.ccxt")
    async def test_fetch_ohlcv_range_pagination_multiple_batches(
        self, mock_ccxt, mock_sleep, mock_exchange
    ):
        """Test fetch_ohlcv_range correctly paginates across multiple batches."""
        mock_ccxt.bitget = MagicMock(return_value=mock_exchange)

        start_ts = 1000000
        interval_ms = 60000
        batch_size = 1000
        total_candles = 2500  # Requires 3 batches
        end_ts = start_ts + (total_candles * interval_ms)

        # Generate candles for each batch
        batch1 = generate_candles(start_ts, batch_size, interval_ms)
        batch2_start = start_ts + (batch_size * interval_ms)
        batch2 = generate_candles(batch2_start, batch_size, interval_ms)
        batch3_start = batch2_start + (batch_size * interval_ms)
        batch3 = generate_candles(batch3_start, 500, interval_ms)  # Partial batch

        mock_exchange.fetch_ohlcv.side_effect = [batch1, batch2, batch3]

        fetcher = CCXTFetcher("bitget")
        fetcher.exchange = mock_exchange
        fetcher.symbols = ["BTC/USDT"]

        result = await fetcher.fetch_ohlcv_range("BTC/USDT", "1m", start_ts, end_ts)

        # Should have called fetch_ohlcv 3 times
        assert mock_exchange.fetch_ohlcv.call_count == 3
        assert len(result) == total_candles

        # Verify rate limiting sleep was called between batches
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    @patch(
        "src.ai_strategy.data.fetchers.ccxt_fetcher.asyncio.sleep",
        new_callable=AsyncMock,
    )
    @patch("src.ai_strategy.data.fetchers.ccxt_fetcher.ccxt")
    async def test_fetch_ohlcv_range_removes_duplicates(
        self, mock_ccxt, mock_sleep, mock_exchange
    ):
        """Test that fetch_ohlcv_range removes duplicate candles."""
        mock_ccxt.bitget = MagicMock(return_value=mock_exchange)

        start_ts = 1000000
        interval_ms = 60000
        end_ts = start_ts + (1500 * interval_ms)

        # Create overlapping batches (last candle of batch1 = first candle of batch2)
        batch1 = generate_candles(start_ts, 1000, interval_ms)
        batch2_start = start_ts + (999 * interval_ms)  # Overlap by 1
        batch2 = generate_candles(batch2_start, 501, interval_ms)

        mock_exchange.fetch_ohlcv.side_effect = [batch1, batch2]

        fetcher = CCXTFetcher("bitget")
        fetcher.exchange = mock_exchange
        fetcher.symbols = ["BTC/USDT"]

        result = await fetcher.fetch_ohlcv_range("BTC/USDT", "1m", start_ts, end_ts)

        # Should have 1500 unique candles, not 1501
        assert len(result) == 1500
        # Verify no duplicate timestamps
        assert result["timestamp"].nunique() == len(result)

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.fetchers.ccxt_fetcher.ccxt")
    async def test_fetch_ohlcv_range_filters_beyond_end(self, mock_ccxt, mock_exchange):
        """Test that candles beyond end timestamp are filtered out."""
        mock_ccxt.bitget = MagicMock(return_value=mock_exchange)

        start_ts = 1000000
        interval_ms = 60000
        end_ts = start_ts + (50 * interval_ms)  # Only want 50 candles

        # Exchange returns more candles than requested range
        candles = generate_candles(start_ts, 100, interval_ms)
        mock_exchange.fetch_ohlcv.return_value = candles

        fetcher = CCXTFetcher("bitget")
        fetcher.exchange = mock_exchange
        fetcher.symbols = ["BTC/USDT"]

        result = await fetcher.fetch_ohlcv_range("BTC/USDT", "1m", start_ts, end_ts)

        # Should only have candles up to end_ts
        assert len(result) == 51  # 0 to 50 inclusive
        assert result["timestamp"].max() <= end_ts

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.fetchers.ccxt_fetcher.ccxt")
    async def test_fetch_ohlcv_range_handles_empty_response(
        self, mock_ccxt, mock_exchange
    ):
        """Test that fetch_ohlcv_range handles empty responses gracefully."""
        mock_ccxt.bitget = MagicMock(return_value=mock_exchange)
        mock_exchange.fetch_ohlcv.return_value = []

        fetcher = CCXTFetcher("bitget")
        fetcher.exchange = mock_exchange
        fetcher.symbols = ["BTC/USDT"]

        result = await fetcher.fetch_ohlcv_range("BTC/USDT", "1m", 1000000, 2000000)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.fetchers.ccxt_fetcher.ccxt")
    async def test_fetch_ohlcv_range_uses_correct_params(
        self, mock_ccxt, mock_exchange
    ):
        """Test that fetch_ohlcv_range uses Bitget params with paginate=False and until."""
        mock_ccxt.bitget = MagicMock(return_value=mock_exchange)
        mock_exchange.fetch_ohlcv.return_value = generate_candles(1000000, 50)

        fetcher = CCXTFetcher("bitget")
        fetcher.exchange = mock_exchange
        fetcher.symbols = ["BTC/USDT"]

        start_ts = 1000000
        end_ts = 2000000
        await fetcher.fetch_ohlcv_range("BTC/USDT", "1m", start_ts, end_ts)

        mock_exchange.fetch_ohlcv.assert_called_with(
            "BTC/USDT",
            "1m",
            start_ts,
            1000,
            params={"uta": False, "paginate": False, "until": end_ts},
        )


class TestGetAvailableSymbols:
    """Tests for get_available_symbols method."""

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.fetchers.ccxt_fetcher.ccxt")
    async def test_get_available_symbols_loads_markets(self, mock_ccxt, mock_exchange):
        """Test that get_available_symbols loads markets if not cached."""
        mock_ccxt.bitget = MagicMock(return_value=mock_exchange)
        mock_exchange.symbols = ["BTC/USDT", "ETH/USDT"]

        fetcher = CCXTFetcher("bitget")
        fetcher.exchange = mock_exchange

        result = await fetcher.get_available_symbols()

        mock_exchange.load_markets.assert_called_once()
        assert result == ["BTC/USDT", "ETH/USDT"]

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.fetchers.ccxt_fetcher.ccxt")
    async def test_get_available_symbols_uses_cache(self, mock_ccxt, mock_exchange):
        """Test that get_available_symbols uses cached symbols."""
        mock_ccxt.bitget = MagicMock(return_value=mock_exchange)

        fetcher = CCXTFetcher("bitget")
        fetcher.exchange = mock_exchange
        fetcher.symbols = ["CACHED/USDT"]  # Pre-populated cache

        result = await fetcher.get_available_symbols()

        mock_exchange.load_markets.assert_not_called()
        assert result == ["CACHED/USDT"]


class TestClose:
    """Tests for close method."""

    @pytest.mark.asyncio
    @patch("src.ai_strategy.data.fetchers.ccxt_fetcher.ccxt")
    async def test_close_calls_exchange_close(self, mock_ccxt, mock_exchange):
        """Test that close properly closes the exchange connection."""
        mock_ccxt.bitget = MagicMock(return_value=mock_exchange)

        fetcher = CCXTFetcher("bitget")
        fetcher.exchange = mock_exchange

        await fetcher.close()

        mock_exchange.close.assert_called_once()
