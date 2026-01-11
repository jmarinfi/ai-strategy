import asyncio
from datetime import datetime, timedelta

from src.ai_strategy.data.fetchers.ccxt_fetcher import CCXTFetcher


async def main():
    fetcher = CCXTFetcher("bitget", sandbox=False)
    print(fetcher.exchange)
    print(await fetcher.get_available_symbols())
    since_ts = (datetime.now() - timedelta(days=30)).timestamp() * 1000
    limit = 31 * 24 * 60
    ohlcv = await fetcher.fetch_ohlcv("BTC/USDT", "1m", since=since_ts, limit=limit)
    print(ohlcv)
    ohlcv_range = await fetcher.fetch_ohlcv_range(
        "BTC/USDT", "1m", since_ts, since_ts + 24 * 60 * 60 * 1000
    )
    print(ohlcv_range)

    await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
