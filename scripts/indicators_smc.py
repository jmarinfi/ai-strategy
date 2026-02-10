from pathlib import Path
import sys
import asyncio

import pandas as pd
from smartmoneyconcepts import smc
import numpy as np

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
        df = df[["open", "high", "low", "close", "volume"]]
        print(df.head())
        print(df.size)
    except Exception as e:
        print(f"Error loading data: {e}")
    finally:
        await fetcher.close()
    return df


def calculate_fvg(df: pd.DataFrame) -> pd.DataFrame:
    df_fvg = smc.fvg(df, join_consecutive=False)
    print(df_fvg.head())
    print(df_fvg.size)

    return df_fvg


def calculate_swing_highs_lows(df: pd.DataFrame, swing_length: int) -> pd.DataFrame:
    df_swing_highs_lows = smc.swing_highs_lows(df, swing_length=swing_length)
    print(df_swing_highs_lows.tail())
    print(df_swing_highs_lows.size)

    return df_swing_highs_lows


def add_labels(
    df: pd.DataFrame,
    df_fvg: pd.DataFrame,
    df_swing_highs_lows: pd.DataFrame,
    swing_length: int,
) -> pd.DataFrame:
    df["fvg_signal"] = df_fvg["FVG"]
    df["fvg_signal"] = df["fvg_signal"].fillna(0).astype(int)

    df["HighLow"] = df_swing_highs_lows["HighLow"]
    df["HighLow"] = df["HighLow"].fillna(0).astype(int)

    print(df.head())
    print(df.tail())
    print(df.size)

    return df


async def main():
    df = await load_data(
        symbol="BTC/USDT", timeframe="15m", output_dir=Path("data/raw"), fetcher=fetcher
    )
    swing_length = 50

    df_fvg = calculate_fvg(df)
    df_swing_highs_lows = calculate_swing_highs_lows(df, swing_length)
    df_swing_highs_lows.iloc[-swing_length:, :] = np.nan

    df = add_labels(
        df,
        df_fvg,
        df_swing_highs_lows,
        swing_length,
    )


if __name__ == "__main__":
    asyncio.run(main())
