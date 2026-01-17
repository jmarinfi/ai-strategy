"""Candle model for OHLCV candlestick data."""

from datetime import datetime

from pydantic import BaseModel, Field


class Candle(BaseModel):
    """Standardized OHLCV candlestick data from exchange.

    Based on CCXT unified OHLCV structure.
    See: https://docs.ccxt.com/#/?id=ohlcv-structure
    """

    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Candle timeframe (e.g., '1m', '5m', '1h')")
    timestamp: int = Field(..., description="Candle opening time (Unix timestamp in milliseconds)")
    datetime_: datetime | str | None = Field(
        None, alias="datetime", description="ISO8601 datetime string"
    )

    # OHLCV data
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price during the period")
    low: float = Field(..., description="Lowest price during the period")
    close: float = Field(..., description="Closing price")
    volume: float = Field(..., description="Trading volume in base currency")

    # Optional fields
    quote_volume: float | None = Field(
        None, alias="quoteVolume", description="Trading volume in quote currency"
    )
    trades: int | None = Field(None, description="Number of trades during the period")

    # Raw data
    info: dict | None = Field(None, description="Raw exchange response")

    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }
