import asyncio
from datetime import datetime, timedelta

from src.ai_strategy.data import CCXTFetcher
from src.ai_strategy.data import CCXTStreamer


async def test_fetcher():
    """Test the CCXTFetcher against the real exchange."""
    print("=" * 60)
    print("Testing CCXTFetcher")
    print("=" * 60)

    fetcher = CCXTFetcher("bitget", sandbox=False)
    print(f"Exchange: {fetcher.exchange}")

    symbols = await fetcher.get_available_symbols()
    print(f"Available symbols: {len(symbols)} (showing first 10)")
    print(symbols[:10])

    since_ts = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
    limit = 31 * 24 * 60
    ohlcv = await fetcher.fetch_ohlcv("BTC/USDT", "1m", since=since_ts, limit=limit)
    print(f"\nfetch_ohlcv result: {len(ohlcv)} candles")
    print(ohlcv.head())

    ohlcv_range = await fetcher.fetch_ohlcv_range(
        "BTC/USDT", "1m", since_ts, since_ts + 24 * 60 * 60 * 1000
    )
    print(f"\nfetch_ohlcv_range result: {len(ohlcv_range)} candles")
    print(ohlcv_range.head())

    await fetcher.close()
    print("\nFetcher test completed!")


async def test_streamer():
    """Test the CCXTStreamer against the real exchange."""
    print("\n" + "=" * 60)
    print("Testing CCXTStreamer")
    print("=" * 60)

    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    print(f"Watching tickers for: {symbols}")
    print("Will receive 20 ticker updates then stop...\n")

    async with CCXTStreamer("bitget", symbols) as streamer:
        count = 0
        max_tickers = 20

        async for ticker in streamer.watch_tickers():
            count += 1
            print(
                f"[{count:3d}] {ticker['symbol']:12s} | "
                f"Last: {ticker['last']:>12} | "
                f"Bid: {ticker['bid']:>12} | "
                f"Ask: {ticker['ask']:>12} | "
                f"Vol: {ticker['baseVolume']}"
            )

            if count >= max_tickers:
                print(f"\nReceived {max_tickers} tickers, stopping...")
                break

    print("Streamer test completed!")


async def main():
    await test_fetcher()
    await test_streamer()


if __name__ == "__main__":
    asyncio.run(main())
