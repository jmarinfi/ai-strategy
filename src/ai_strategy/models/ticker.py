"""Ticker model for real-time market data."""

from datetime import datetime

from pydantic import BaseModel, Field


class Ticker(BaseModel):
    """Standardized ticker data from exchange.

    Based on CCXT unified ticker structure.
    See: https://docs.ccxt.com/#/?id=ticker-structure
    """

    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    timestamp: int | None = Field(
        None, description="Unix timestamp in milliseconds"
    )
    datetime_: datetime | str | None = Field(
        None, alias="datetime", description="ISO8601 datetime string"
    )

    # Price data
    high: float | None = Field(None, description="Highest price in last 24h")
    low: float | None = Field(None, description="Lowest price in last 24h")
    open: float | None = Field(None, description="Opening price 24h ago")
    close: float | None = Field(None, description="Closing/current price")
    last: float | None = Field(None, description="Last traded price")

    # Bid/Ask
    bid: float | None = Field(None, description="Best bid price")
    bid_volume: float | None = Field(
        None, alias="bidVolume", description="Best bid volume"
    )
    ask: float | None = Field(None, description="Best ask price")
    ask_volume: float | None = Field(
        None, alias="askVolume", description="Best ask volume"
    )

    # Volume
    base_volume: float | None = Field(
        None, alias="baseVolume", description="24h volume in base currency"
    )
    quote_volume: float | None = Field(
        None, alias="quoteVolume", description="24h volume in quote currency"
    )

    # Change
    change: float | None = Field(None, description="Absolute price change")
    percentage: float | None = Field(None, description="Percentage price change")

    # Raw data
    info: dict | None = Field(None, description="Raw exchange response")

    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }
