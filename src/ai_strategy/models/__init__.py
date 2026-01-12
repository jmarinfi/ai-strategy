"""Models for ai-strategy."""

from src.ai_strategy.models.signal import Signal, SignalType
from src.ai_strategy.models.ticker import Ticker
from src.ai_strategy.models.webhook import (
    AddFundsWebhook,
    AssetType,
    ChangePairsWebhook,
    CloseDealSlWebhook,
    CloseDealWebhook,
    CloseType,
    FundType,
    PairsMode,
    ReduceFundsWebhook,
    StartBotWebhook,
    StartDealWebhook,
    StopBotWebhook,
    WebhookAction,
    WebhookRequest,
)

__all__ = [
    # Signal
    "Signal",
    "SignalType",
    # Ticker
    "Ticker",
    # Webhook
    "AddFundsWebhook",
    "AssetType",
    "ChangePairsWebhook",
    "CloseDealSlWebhook",
    "CloseDealWebhook",
    "CloseType",
    "FundType",
    "PairsMode",
    "ReduceFundsWebhook",
    "StartBotWebhook",
    "StartDealWebhook",
    "StopBotWebhook",
    "WebhookAction",
    "WebhookRequest",
]
