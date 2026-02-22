"""Webhook models for trading bot communication."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class WebhookAction(str, Enum):
    """Available webhook actions."""

    START_DEAL = "startDeal"
    CLOSE_DEAL = "closeDeal"
    CLOSE_DEAL_SL = "closeDealSl"
    START_BOT = "startBot"
    STOP_BOT = "stopBot"
    ADD_FUNDS = "addFunds"
    REDUCE_FUNDS = "reduceFunds"
    CHANGE_PAIRS = "changePairs"


class AssetType(str, Enum):
    """Asset type for fund operations."""

    BASE = "base"
    QUOTE = "quote"


class FundType(str, Enum):
    """Type of fund adjustment."""

    PERC = "perc"
    FIXED = "fixed"


class CloseType(str, Enum):
    """Type of bot close operation."""

    LIMIT = "limit"
    MARKET = "market"
    LEAVE = "leave"
    CANCEL = "cancel"


class PairsMode(str, Enum):
    """Mode for changing bot pairs."""

    REPLACE = "replace"
    ADD = "add"
    REMOVE = "remove"


# Deal-related webhooks
class StartDealWebhook(BaseModel):
    """Start a new deal.

    Examples:
        All symbols: {"action": "startDeal", "uuid": "..."}
        Specific: {"action": "startDeal", "uuid": "...", "symbol": "BTC_USDT"}
    """

    action: Literal[WebhookAction.START_DEAL] = WebhookAction.START_DEAL
    uuid: str = Field(..., description="Bot UUID")
    symbol: str | None = Field(None, description="Trading symbol (BTC_USDT format)")

    model_config = {
        "use_enum_values": True,
    }


class CloseDealWebhook(BaseModel):
    """Close an open deal.

    Examples:
        All symbols: {"action": "closeDeal", "uuid": "..."}
        Specific: {"action": "closeDeal", "uuid": "...", "symbol": "BTC_USDT"}
    """

    action: Literal[WebhookAction.CLOSE_DEAL] = WebhookAction.CLOSE_DEAL
    uuid: str = Field(..., description="Bot UUID")
    symbol: str | None = Field(None, description="Trading symbol (BTC_USDT format)")

    model_config = {
        "use_enum_values": True,
    }


class CloseDealSlWebhook(BaseModel):
    """Close deal by stop loss.

    Examples:
        All symbols: {"action": "closeDealSl", "uuid": "..."}
        Specific: {"action": "closeDealSl", "uuid": "...", "symbol": "BTC_USDT"}
    """

    action: Literal[WebhookAction.CLOSE_DEAL_SL] = WebhookAction.CLOSE_DEAL_SL
    uuid: str = Field(..., description="Bot UUID")
    symbol: str | None = Field(None, description="Trading symbol (BTC_USDT format)")

    model_config = {
        "use_enum_values": True,
    }


# Bot control webhooks
class StartBotWebhook(BaseModel):
    """Start the bot.

    Example:
        {"action": "startBot", "uuid": "..."}
    """

    action: Literal[WebhookAction.START_BOT] = WebhookAction.START_BOT
    uuid: str = Field(..., description="Bot UUID")

    model_config = {
        "use_enum_values": True,
    }


class StopBotWebhook(BaseModel):
    """Stop the bot.

    Example:
        {"action": "stopBot", "uuid": "...", "closeType": "limit"}
    """

    action: Literal[WebhookAction.STOP_BOT] = WebhookAction.STOP_BOT
    uuid: str = Field(..., description="Bot UUID")
    close_type: CloseType = Field(..., alias="closeType")

    model_config = {
        "use_enum_values": True,
        "populate_by_name": True,
    }


# Fund adjustment webhooks
class AddFundsWebhook(BaseModel):
    """Add funds to deals.

    Examples:
        All symbols: {"action": "addFunds", "uuid": "...", "asset": "quote", "qty": "10", "type": "perc"}
        Specific: {"action": "addFunds", "uuid": "...", "asset": "base", "qty": "0.5", "symbol": "BTC_USDT", "type": "fixed"}
    """

    action: Literal[WebhookAction.ADD_FUNDS] = WebhookAction.ADD_FUNDS
    uuid: str = Field(..., description="Bot UUID")
    asset: AssetType
    qty: str = Field(..., description="Quantity as string")
    type: FundType = Field(..., description="Percentage or fixed")
    # symbol: str | None = Field(None, description="Trading symbol (BTC_USDT format)")

    model_config = {
        "use_enum_values": True,
    }


class ReduceFundsWebhook(BaseModel):
    """Reduce funds in deals.

    Examples:
        All symbols: {"action": "reduceFunds", "uuid": "...", "asset": "quote", "qty": "20", "type": "perc"}
        Specific: {"action": "reduceFunds", "uuid": "...", "asset": "base", "qty": "0.1", "symbol": "BTC_USDT", "type": "fixed"}
    """

    action: Literal[WebhookAction.REDUCE_FUNDS] = WebhookAction.REDUCE_FUNDS
    uuid: str = Field(..., description="Bot UUID")
    asset: AssetType
    qty: str = Field(..., description="Quantity as string")
    type: FundType = Field(..., description="Percentage or fixed")
    symbol: str | None = Field(None, description="Trading symbol (BTC_USDT format)")

    model_config = {
        "use_enum_values": True,
    }


# Pairs management webhook
class ChangePairsWebhook(BaseModel):
    """Change bot trading pairs.

    Examples:
        Replace: {"action": "changePairs", "uuid": "...", "pairsToSet": ["BTC_USDT"], "pairsToSetMode": "replace"}
        Add: {"action": "changePairs", "uuid": "...", "pairsToSet": ["ETH_USDT"], "pairsToSetMode": "add"}
        Remove: {"action": "changePairs", "uuid": "...", "pairsToSet": ["SOL_USDT"], "pairsToSetMode": "remove"}
    """

    action: Literal[WebhookAction.CHANGE_PAIRS] = WebhookAction.CHANGE_PAIRS
    uuid: str = Field(..., description="Bot UUID")
    pairs_to_set: list[str] = Field(..., alias="pairsToSet")
    pairs_to_set_mode: PairsMode = Field(..., alias="pairsToSetMode")

    model_config = {
        "use_enum_values": True,
        "populate_by_name": True,
    }


# Union type for all webhooks using discriminated union
WebhookRequest = Annotated[
    StartDealWebhook
    | CloseDealWebhook
    | CloseDealSlWebhook
    | StartBotWebhook
    | StopBotWebhook
    | AddFundsWebhook
    | ReduceFundsWebhook
    | ChangePairsWebhook,
    Field(discriminator="action"),
]
