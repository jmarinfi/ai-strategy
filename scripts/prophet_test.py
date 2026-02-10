from pathlib import Path
import sys
import asyncio

import pandas as pd
from prophet import Prophet

root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from src.ai_strategy.data.fetchers.ccxt_fetcher import CCXTFetcher  # noqa: E402
from src.ai_strategy.data import CCXTFetcher  # noqa: E402, F811
from src.ai_strategy.data import BaseFetcher  # noqa: E402


fetcher = CCXTFetcher("bitget", sandbox=False)


async def load_data(
    symbol: str, timeframe: str, output_dir: Path, fetcher: BaseFetcher
) -> pd.DataFrame:
    try:
        df = await fetcher.get_training_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=100_000,
            output_dir=output_dir,
        )
        df["y"] = (df["close"] - df["close"].shift(1)) / df["close"].shift(1)
        df = df.dropna()
        df["ds"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["ds", "y"]]
        print(df.head())
        print(df.size)
    except Exception as e:
        print(f"Error loading data: {e}")
    finally:
        await fetcher.close()
    return df


async def main():
    df = await load_data(
        symbol="BTC/USDT", timeframe="15m", output_dir=Path("data/raw"), fetcher=fetcher
    )

    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=90)
    print(future.tail())

    forecast = m.predict(future)
    print(forecast.head())
    print(forecast.tail())

    fig1 = m.plot(forecast)
    fig1.savefig("prophet_test.png")
    fig2 = m.plot_components(forecast)
    fig2.savefig("prophet_test_components.png")


if __name__ == "__main__":
    asyncio.run(main())
