"""Main script for training and running LightGBM trading strategies."""

import asyncio
from pathlib import Path

from src.ai_strategy.data import CCXTFetcher, CCXTStreamer
from src.ai_strategy.models.sklearn.lightgbm_model import (
    LightGBMModel,
    add_technical_indicators,
)
from src.ai_strategy.models.sklearn.xgboost_model import XGBoostModel
from src.ai_strategy.strategies import LightGBMStrategy

# Configuration
DATA_DIR = Path("data/raw")

SYMBOL = "ADA/USDT"
TIMEFRAME = "1m"
LIMIT_CANDLES = 100_000

# Webhook configuration
WEBHOOK_URL = "http://192.168.1.132:7503/trade_signal"
LONG_BOT_UUID = "75ec9a0c-f525-4560-a2e7-ccd4a6915093"  # UUID for LONG positions
SHORT_BOT_UUID = None                                   # UUID for SHORT positions

# Configuration
HORIZON = 30
MIN_MOVEMENT = 0.0
OPTIMIZE = True
N_TRIALS = 30
PROB_THRESHOLD = 0.51

# MODE: 'train' or 'live'
MODE = "live"  # Change to 'live' to run the live strategy


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
            df = add_technical_indicators(df)

            # 3. Create and train model
            model = LightGBMModel()

            # Prepare data for training
            X_train, y_train, X_test, y_test = model.prepare_data_for_training(
                df, horizon=HORIZON, min_movement=MIN_MOVEMENT
            )

            # Train model
            model.train(
                X_train,
                y_train,
                X_test,
                y_test,
                optimize=OPTIMIZE,
                n_trials=N_TRIALS,
            )

            # 4. Save model with configuration
            config = {
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME,
                "horizon": HORIZON,
                "min_movement": MIN_MOVEMENT,
                "optimized": OPTIMIZE,
                "n_trials": N_TRIALS if OPTIMIZE else None,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }

            output_path = f"models/lightgbm_{SYMBOL.replace('/', '_').lower()}_{TIMEFRAME}_h{HORIZON}.pkl"
            model.save(output_path, config=config)

            print("\n✅ Training completed successfully!")
            print(f"📦 Model saved and ready for predictions")

        elif MODE == "live":
            print("� MODE: LIVE TRADING\n")

            # Construct model path
            model_path = f"models/lightgbm_{SYMBOL.replace('/', '_').lower()}_{TIMEFRAME}_h{HORIZON}.pkl"

            # Check if model exists
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. "
                    f"Please train the model first by setting MODE='train'"
                )

            # Create streamer
            async with CCXTStreamer("bitget", [SYMBOL]) as streamer:
                # Create strategy
                strategy = LightGBMStrategy(
                    symbols=[SYMBOL],
                    streamer=streamer,
                    webhook_url=WEBHOOK_URL,
                    model_path=model_path,
                    timeframe=TIMEFRAME,
                    long_bot_uuid=LONG_BOT_UUID,
                    short_bot_uuid=SHORT_BOT_UUID,
                    exchange="bitget",
                    prob_threshold=PROB_THRESHOLD,
                    historical_candles=60,
                )

                print(f"🚀 Starting LightGBM strategy...")
                print(f"   Model: {model_path}")
                print(f"   Symbol: {SYMBOL}")
                print(f"   Timeframe: {TIMEFRAME}")
                print(f"   Threshold: {PROB_THRESHOLD}")
                print(f"   Features: {strategy.model.num_features}")
                print(f"\n👂 Listening for candles...")
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

        else:
            raise ValueError(f"Invalid MODE: {MODE}. Must be 'train' or 'live'")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
