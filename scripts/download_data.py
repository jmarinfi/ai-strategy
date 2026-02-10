import sys
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import asyncio

root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from src.ai_strategy.data.fetchers.ccxt_fetcher import CCXTFetcher  # noqa: E402


class Timeframe(Enum):
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"

    @property
    def minutes(self) -> int:
        unit = self.value[-1]
        amount = int(self.value[:-1])

        if unit == "m":
            return amount
        if unit == "h":
            return amount * 60
        if unit == "d":
            return amount * 24 * 60

        raise ValueError(f"Unknown timeframe unit: {unit}")

    def __str__(self) -> str:
        return self.value


async def download_data(symbol: str, timeframe: Timeframe, since: datetime):
    fetcher = CCXTFetcher("bitget", sandbox=False)

    delta = datetime.now() - since
    total_minutes = int(delta.total_seconds() / 60)
    limit_candles = total_minutes // timeframe.minutes

    try:
        df = await fetcher.get_training_data(
            symbol=symbol,
            timeframe=timeframe.value,
            limit=limit_candles,
            output_dir=Path("data/raw"),
        )
        print(f"Data downloaded successfully: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"Error downloading data: {e}")
    finally:
        await fetcher.close()


async def main():
    now = datetime.now()
    tasks = [
        download_data(
            symbol="BTC/USDT",
            timeframe=Timeframe.M15,
            since=now - timedelta(days=70),
        ),
        download_data(
            symbol="ETH/USDT",
            timeframe=Timeframe.M15,
            since=now - timedelta(days=70),
        ),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
