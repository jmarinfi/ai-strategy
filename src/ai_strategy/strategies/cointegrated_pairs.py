from datetime import datetime

import numpy as np
import pandas as pd

from src.ai_strategy.models.candle import Candle
from src.ai_strategy.models.signal import Signal
from src.ai_strategy.strategies.base import BaseStrategy
from src.ai_strategy.data import BaseStreamer
from src.ai_strategy.data.fetchers.ccxt_fetcher import CCXTFetcher
from src.ai_strategy.data.preprocessing import Preprocessor


class CointegratedPairs(BaseStrategy):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        window: int,
        zscore: float,
        streamer: BaseStreamer,
        webhook_url: str,
        long_bot_uuid: str,
        short_bot_uuid: str,
        timeframe: str,
    ):
        super().__init__(
            symbols=pairs,
            streamer=streamer,
            webhook_url=webhook_url,
            long_bot_uuid=long_bot_uuid,
            short_bot_uuid=short_bot_uuid,
            timeframe=timeframe,
        )
        self.pairs = pairs
        self.window = window
        self.zscore = zscore
