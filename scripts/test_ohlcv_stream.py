"""Test script for CCXTStreamer watch_ohlcv functionality."""

import asyncio
from src.ai_strategy.data import CCXTStreamer


async def test_watch_ohlcv():
    """Test watching OHLCV candles via websocket."""
    print("=" * 80)
    print("TESTING CCXT STREAMER - WATCH OHLCV")
    print("=" * 80)

    symbol = "BTC/USDT"
    timeframe = "5m"
    max_candles = 5

    print(f"\n📊 Symbol: {symbol}")
    print(f"⏱️  Timeframe: {timeframe}")
    print(f"🎯 Will receive {max_candles} candle updates then unsubscribe...\n")

    async with CCXTStreamer("bitget", [symbol]) as streamer:
        count = 0

        try:
            async for candle in streamer.watch_ohlcv(symbol, timeframe):
                count += 1
                timestamp, open_price, high, low, close, volume = candle
                
                # Convert timestamp to readable format
                from datetime import datetime
                dt = datetime.fromtimestamp(timestamp / 1000)
                
                print(f"[{count}] {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    O: {open_price:>12.2f}  H: {high:>12.2f}")
                print(f"    L: {low:>12.2f}  C: {close:>12.2f}")
                print(f"    V: {volume:>12.4f}")
                print()

                if count >= max_candles:
                    print(f"✅ Received {max_candles} candles. Unsubscribing...")
                    await streamer.un_watch_ohlcv(symbol, timeframe)
                    break

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_watch_ohlcv())
