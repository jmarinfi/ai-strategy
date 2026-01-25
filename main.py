"""Main script for training and running LightGBM trading strategies."""

import asyncio
from pathlib import Path

import pandas as pd
import numpy as np
from backtesting import Backtest

from src.ai_strategy.data import CCXTFetcher, CCXTStreamer, Preprocessor
from src.ai_strategy.models.mlm.random_forest_model import RandomForestMLM
from src.ai_strategy.strategies import RandomForestStrategy
from src.ai_strategy.backtesting import RandomForestStrategyBacktest

# Configuration
DATA_DIR = Path("data/raw")

SYMBOL = "ADA/USDT"
TIMEFRAME = "15m"
LIMIT_CANDLES = 100_000

# Webhook configuration
WEBHOOK_URL = "http://192.168.1.132:7503/trade_signal"
LONG_BOT_UUID = "d56aef35-8fb1-4cdf-b8c8-2d7f0cca2005"  # UUID for LONG positions
SHORT_BOT_UUID = "3bfa7130-5bc9-410e-ab3e-5f19152dec39"  # UUID for SHORT positions

# Random Forest Configuration (Machine Learning Mastery method)
N_LAGS = 96  # How many previous time steps to use as features (sliding window size)

# MODE: 'train' or 'live' or 'backtest'
MODE = "train"


async def main():
    """Main entry point for the application."""
    try:
        if MODE == "train":
            print("🎓 MODE: TRAINING\n")

            # 1. Get training data
            fetcher = CCXTFetcher("bitget", sandbox=False)
            try:
                df = await fetcher.get_training_data(
                    symbol=SYMBOL,
                    timeframe=TIMEFRAME,
                    limit=LIMIT_CANDLES,
                    output_dir=DATA_DIR,
                )
            finally:
                await fetcher.close()

            # 2. Add technical indicators
            print(f"\n🧪 Adding technical indicators to {len(df)} candles...")
            df = Preprocessor.add_technical_indicators(df)
            print(f"✅ Indicators added. New shape: {df.shape}")

            # 3. Create and train model
            print("\n🧠 Initializing RandomForestMLM...")
            model = RandomForestMLM(
                df, n_lags_in=N_LAGS, n_lags_out=1, name_targets=["close"]
            )
            X_train, y_train, X_test, y_test = model.prepare_data_for_training(
                train_size=0.95
            )

            # # Train model
            # model.train(
            #     X_train,
            #     y_train,
            #     X_test,
            #     y_test,
            # )

            output_path = f"models/random_forest_{SYMBOL.replace('/', '_').lower()}_{TIMEFRAME}_lags{N_LAGS}.pkl"
            model.walk_forward_validation(
                X_train,
                y_train,
                X_test,
                y_test,
                block_size=96,
                results_path="results/random_forest_walk_forward_validation.csv",
                output_path=output_path,
            )

            # Re-entrenar con todos los datos para producción
            print("\n🔄 Re-training model with FULL dataset for production...")
            X_full = np.concatenate((X_train, X_test))
            y_full = np.concatenate((y_train, y_test))
            model.model.fit(X_full, y_full)
            print("✅ Production model trained successfully")

            model.save(output_path)

        elif MODE == "live":
            print("🚀 MODE: LIVE TRADING\n")

            # Construct model path
            model_path = f"models/random_forest_{SYMBOL.replace('/', '_').lower()}_{TIMEFRAME}_lags{N_LAGS}.pkl"

            # Check if model exists
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. "
                    f"Please train the model first by setting MODE='train'"
                )

            # Create streamer
            async with CCXTStreamer("bitget", [SYMBOL]) as streamer:
                # Create strategy
                strategy = RandomForestStrategy(
                    symbols=[SYMBOL],
                    streamer=streamer,
                    webhook_url=WEBHOOK_URL,
                    model_path=model_path,
                    timeframe=TIMEFRAME,
                    n_lags=N_LAGS,
                    long_bot_uuid=LONG_BOT_UUID,
                    short_bot_uuid=SHORT_BOT_UUID,
                    exchange="bitget",
                    price_threshold=0.001,
                    historical_candles=120,  # Enough for indicators (~52) + N_LAGS (48) + margin
                )

                print("🚀 Starting Random Forest strategy...")
                print(f"   Model: {model_path}")
                print(f"   Symbol: {SYMBOL}")
                print(f"   Timeframe: {TIMEFRAME}")
                print(f"   N_Lags: {N_LAGS}")
                print("   Price Threshold: ±0.1%")
                print("\n👂 Listening for candles...")
                print("   Press Ctrl+C to stop\n")

                try:
                    # Run the strategy
                    await strategy.run()
                except (KeyboardInterrupt, asyncio.CancelledError):
                    print("\n⚠️  Received interrupt signal. Stopping...")
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    import traceback

                    traceback.print_exc()
                finally:
                    await strategy.stop()
                    print("✅ Strategy stopped cleanly")

        elif MODE == "backtest":
            print(" MODE: BACKTESTING\n")

            data = pd.read_parquet("data/raw/ada_usdt_5m.parquet")

            # Rename columns to backtesting.py format (capitalized)
            data = data.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )

            # Convert timestamp (ms) to datetime and set as index
            data.index = pd.to_datetime(data["timestamp"], unit="ms")
            data = data.drop(columns=["timestamp"])

            bt_random_forest = Backtest(
                data=data,
                strategy=RandomForestStrategyBacktest,
                commission=0.0008,
                hedging=True,
            )
            stats = bt_random_forest.run()
            print(stats)

            stats = bt_random_forest.optimize(
                price_pct_threshold=[0.0005, 0.001, 0.002, 0.004],
                n_candles_stop=range(1, 10, 2),
                take_profit_pct=[0.002, 0.003, 0.004, 0.005],
                stop_loss_pct=[-0.001, -0.002, -0.003, -0.004],
            )
            print(stats)

            bt_random_forest.plot()

        else:
            raise ValueError(f"Invalid MODE: {MODE}. Must be 'train' or 'live'")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
