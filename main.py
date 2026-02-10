"""Main script for training and running LightGBM trading strategies."""

import asyncio
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from backtesting import Backtest

from src.ai_strategy.data import CCXTFetcher, CCXTStreamer, Preprocessor
from src.ai_strategy.models.mlm.random_forest_model import RandomForestMLM
from src.ai_strategy.strategies import RandomForestStrategy, MeanReversionDCA
from src.ai_strategy.backtesting import RandomForestStrategyBacktest

# Configuration
DATA_DIR = Path("data/raw")

SYMBOL = "PEPE/USDT"
TIMEFRAME = "15m"
LIMIT_CANDLES = 100_000

# Webhook configuration
WEBHOOK_URL = "http://192.168.1.132:7503/trade_signal"
LONG_BOT_UUID = "59d6aa5c-6209-4c3d-90f1-65eda1457452"  # UUID for LONG positions
SHORT_BOT_UUID = "b0860ab2-36ee-423f-9d9a-a3202830d591"  # UUID for SHORT positions

# Random Forest Configuration (Machine Learning Mastery method)
N_LAGS = 48  # How many previous time steps to use as features (sliding window size)
PREDICTION_HORIZON = 4  # How many time steps to predict into the future

# MODE: 'train' or 'live' or 'backtest'
MODE = "live"


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
            swing_length = 10
            df = Preprocessor.add_smc_indicators(df, swing_length=swing_length)
            print(f"✅ Indicators added. New shape: {df.shape}")

            # 3. Transform prices to log-returns (stationarity for time series)
            print("\n📈 Transforming prices to log-returns...")
            # Calculate log-returns for target prediction
            df["close_return"] = np.log(
                df["close"] / df["close"].shift(PREDICTION_HORIZON)
            )
            # Remove original close column (now we use close_return as target)
            df = df.drop(columns=["close"])
            # Drop the first row with NaN from the shift
            df = df.dropna()
            print(f"✅ Log-returns calculated. New shape: {df.shape}")
            print(df.head())
            print(df.tail())

            # 4. Create and train model
            print("\n🧠 Initializing RandomForestMLM...")
            model = RandomForestMLM(
                df,
                n_lags_in=N_LAGS,
                n_lags_out=1,
                prediction_horizon=PREDICTION_HORIZON,
                name_targets=["close_return"],
                swing_length=swing_length,
            )

            X_train, y_train, X_test, y_test = model.prepare_data_for_training(
                train_size=0.95
            )
            print(f"✅ Data prepared for training. X_train shape: {X_train.shape}")
            print(f"✅ Data prepared for training. y_train shape: {y_train.shape}")
            print(f"✅ Data prepared for training. X_test shape: {X_test.shape}")
            print(f"✅ Data prepared for training. y_test shape: {y_test.shape}")

            # Train model
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

            # # Construct model path
            # model_path = f"models/random_forest_{SYMBOL.replace('/', '_').lower()}_{TIMEFRAME}_lags{N_LAGS}.pkl"

            # # Check if model exists
            # if not Path(model_path).exists():
            #     raise FileNotFoundError(
            #         f"Model not found at {model_path}. "
            #         f"Please train the model first by setting MODE='train'"
            #     )

            # Create streamer
            async with CCXTStreamer("bitget", [SYMBOL]) as streamer:
                # Create strategy
                # strategy = RandomForestStrategy(
                #     symbols=[SYMBOL],
                #     streamer=streamer,
                #     webhook_url=WEBHOOK_URL,
                #     model_path=model_path,
                #     timeframe=TIMEFRAME,
                #     n_lags=N_LAGS,
                #     long_bot_uuid=LONG_BOT_UUID,
                #     short_bot_uuid=SHORT_BOT_UUID,
                #     exchange="bitget",
                #     price_threshold=0.024783739559276756,
                #     historical_candles=150,  # Enough for indicators (~52) + N_LAGS (96) + margin
                # )
                strategy = MeanReversionDCA(
                    symbols=[SYMBOL],
                    streamer=streamer,
                    webhook_url=WEBHOOK_URL,
                    long_bot_uuid=LONG_BOT_UUID,
                    short_bot_uuid=SHORT_BOT_UUID,
                    timeframe=TIMEFRAME,
                    window_bb=6,
                    window_dev_bb=1.65,
                    max_orders_dca=9,
                    min_pct_objective=0.5 / 100,
                )

                print("🚀 Starting strategy...")
                print(f"   Symbol: {SYMBOL}")
                print(f"   Timeframe: {TIMEFRAME}")
                print(f"   Window BB: {strategy.window_bb}")
                print(f"   Window Dev BB: {strategy.window_dev_bb}")
                print(f"   Max Orders DCA: {strategy.max_orders_dca}")
                print(f"   Min Pct Objective: {strategy.min_pct_objective * 100}%")
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
            print("🚀 MODE: BACKTESTING\n")

            data = pd.read_parquet("data/raw/ada_usdt_15m.parquet")
            # Filter data from 2025-03-10 to final
            data = data[data["timestamp"] >= datetime(2025, 3, 10).timestamp() * 1000]
            print(f"Enncabezado de datos para backtesting:\n{data.head()}")
            print(f"Columna de datos para backtesting:\n{data.columns}")
            print(f"Cantidad de datos para backtesting: {len(data)}")

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
            print(f"Datos para backtesting:\n{data.head()}")
            print(f"Datos para backtesting:\n{data.tail()}")

            bt_random_forest = Backtest(
                data=data,
                strategy=RandomForestStrategyBacktest,
                commission=0.0008,
                hedging=True,
            )
            stats = bt_random_forest.run()
            print(stats)
            bt_random_forest.plot()

            stats = bt_random_forest.optimize(
                method="sambo",
                max_tries=25,
                price_pct_threshold=[0.01, 0.015, 0.02, 0.025, 0.03],
                minutes_stop=range(4 * 60, 5 * 60, 15),
                take_profit_pct=[0.005, 0.0075, 0.01, 0.0125, 0.015],
                stop_loss_pct=[-0.0025, -0.005, -0.0075, -0.01, -0.0125],
            )
            print(stats)

            print("\n✨ MEJORES PARÁMETROS ENCONTRADOS:")
            best_strat = stats["_strategy"]
            print(f"• price_pct_threshold: {best_strat.price_pct_threshold}")
            print(f"• minutes_stop:        {best_strat.minutes_stop}")
            print(f"• take_profit_pct:     {best_strat.take_profit_pct}")
            print(f"• stop_loss_pct:       {best_strat.stop_loss_pct}")

            bt_random_forest.plot()

        else:
            raise ValueError(f"Invalid MODE: {MODE}. Must be 'train' or 'live'")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
